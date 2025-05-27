from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn # Added for nn.GRU
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from torch.optim.lr_scheduler import LambdaLR # Changed from CosineAnnealingWarmRestarts

from src.models.mlp import MLP
from src.models.gru import GRUModel # Added for GRU


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs) # RMSE is sqrt(MSE)

class ActuatorModel(pl.LightningModule):
    """
    Lightning module for actuator torque prediction.
    Can use either an MLP or a GRU model.
    Can operate in direct prediction mode or residual prediction mode.
    Expects input sequences of length 2 with 5 features per step.
    Feature order for tau_phys: current_angle, target_angle, current_ang_vel, target_ang_vel.
    """
    
    def __init__(
        self,
        model_type: str,
        input_dim: int,         # Expect 4 from DataModule (ActuatorDataset.INPUT_DIM)
        mlp_hidden_dims: list[int] = [64, 128, 64],
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_use_batch_norm: bool = True,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        use_residual: bool = False,
        k_spring: float = 0.0,
        theta0: float = 0.0,
        kp_phys: float = 0.0,
        kd_phys: float = 0.0,
        loss_diff_weight: float = 0.1,
        warmup_epochs: int = 1,
        **kwargs
    ):
        """
        Initialize ActuatorModel.
        
        Args:
            model_type: 'mlp' or 'gru'
            input_dim: Dimension of input features (per timestep for GRU)
            mlp_hidden_dims: List of hidden layer dimensions (for MLP)
            mlp_activation: Activation function (for MLP)
            mlp_dropout: Dropout probability (for MLP)
            mlp_use_batch_norm: Whether to use batch normalization (for MLP)
            gru_hidden_dim: Hidden dimension for GRU
            gru_num_layers: Number of GRU layers
            gru_dropout: Dropout probability for GRU layers
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            use_residual: If True, predicts residual torque (tau_meas - tau_phys)
            k_spring: Spring constant for tau_phys calculation
            theta0: Spring equilibrium angle for tau_phys calculation
            kp_phys: Proportional gain for physical model part of tau_phys
            kd_phys: Derivative gain for physical model part of tau_phys
            loss_diff_weight: Weight for the first-difference of torque predictions in the loss
            warmup_epochs: Number of epochs for learning rate warmup in cosine scheduler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])

        if self.hparams.input_dim != 4: # Check for 4 features
            print(f"Warning: ActuatorModel input_dim={self.hparams.input_dim}, expected 4.")

        self.model_type = self.hparams.model_type.lower()
        self.use_residual = use_residual
        self.k_spring = k_spring
        self.theta0 = theta0
        self.kp_phys = kp_phys
        self.kd_phys = kd_phys
        self.loss_diff_weight = loss_diff_weight

        if self.model_type == "mlp":
            mlp_input_dim_effective = self.hparams.input_dim * 2 # seq_len=2 -> 4*2=8
            print(f"MLP model, effective input_dim (flattened sequence): {mlp_input_dim_effective}")
            self.model = MLP(
                input_dim=mlp_input_dim_effective,
                hidden_dims=self.hparams.mlp_hidden_dims,
                output_dim=1, activation=self.hparams.mlp_activation,
                dropout=self.hparams.mlp_dropout, use_batch_norm=self.hparams.mlp_use_batch_norm,
            )
        elif self.model_type == "gru":
            self.model = GRUModel(
                input_dim=self.hparams.input_dim, # GRU input_dim is per step (4)
                hidden_dim=self.hparams.gru_hidden_dim, num_layers=self.hparams.gru_num_layers,
                output_dim=1, dropout=self.hparams.gru_dropout
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.hparams.model_type}")

        # Metrics - adding RMSE
        base_metrics = {
            "mse": MeanSquaredError(),
            "rmse": RootMeanSquaredError(), # Added RMSE
            "mae": MeanAbsoluteError(),
            "r2": R2Score()
        }
        self.train_metrics = nn.ModuleDict({k: v.clone() for k, v in base_metrics.items()})
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in base_metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in base_metrics.items()})

        self.test_step_outputs = []
    
    def on_fit_start(self):
        device = next(self.parameters()).device
        for metric_dict in [self.train_metrics, self.val_metrics, self.test_metrics]:
            for name, metric in metric_dict.items():
                metric_dict[name] = metric.to(device)
    
    def on_test_start(self):
        device = next(self.parameters()).device
        for name, metric in self.test_metrics.items():
            self.test_metrics[name] = metric.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        For MLP, x shape: [batch_size, input_dim]
        For GRU, x shape: [batch_size, seq_len, input_dim] -> output [batch_size, 1] (last timestep prediction)
        """
        if self.model_type == "mlp":
            batch_size = x.size(0)
            x = x.view(batch_size, -1) # Flatten sequence and features for MLP
        return self.model(x)

    def _calculate_tau_phys(self, features_current_step: torch.Tensor) -> torch.Tensor:
        """
        Calculates physics-based torque (tau_phys) for the CURRENT time step k.
        Assumes features_current_step are [theta(k), theta_d(k), theta_dot(k), theta_d_dot(k), ...other_features]
        Indices for relevant features:
        - 0: current_angle_rad(k)
        - 1: target_angle_rad(k)
        - 2: current_ang_vel_rad_s(k)
        - 3: target_ang_vel_rad_s(k)
        """
        current_angle = features_current_step[:, 0]
        target_angle = features_current_step[:, 1]
        current_ang_vel = features_current_step[:, 2]
        target_ang_vel = features_current_step[:, 3]

        error = target_angle - current_angle
        error_dot = target_ang_vel - current_ang_vel
        
        tau_spring_comp = self.k_spring * (current_angle - self.theta0)
        tau_pd_comp = self.kp_phys * error + self.kd_phys * error_dot
        
        tau_phys = tau_spring_comp + tau_pd_comp
        return tau_phys.unsqueeze(1) # Shape [batch_size, 1]

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """
        General step function for training, validation, and testing.
        """
        x, y_measured_torque = batch # y_measured_torque is tau_meas

        # y_hat is the direct model output (either full torque or residual)
        y_hat = self(x).squeeze(-1) # Output shape: [batch_size]

        target_for_loss = y_measured_torque 
        
        if self.use_residual:
            # tau_phys is calculated based on input features x
            # The `_calculate_tau_phys` method needs to know the structure of x
            tau_phys = self._calculate_tau_phys(x).squeeze(-1) # Shape [batch_size]
            target_for_loss = y_measured_torque - tau_phys # Model learns residual
            # The final predicted torque for metrics is model_output (residual) + tau_phys
            y_pred_final_torque = y_hat + tau_phys
        else:
            # Model learns the full torque directly
            y_pred_final_torque = y_hat

        # Main MSE loss
        loss = F.mse_loss(y_hat, target_for_loss)
        
        # First-difference loss for smoothing
        if self.loss_diff_weight > 0 and len(y_hat) > 1:
            diff_y_hat = torch.diff(y_hat)
            diff_target = torch.diff(target_for_loss) # Smooth the target for the model
            loss_diff = F.mse_loss(diff_y_hat, diff_target)
            loss = loss + self.loss_diff_weight * loss_diff
        
        metrics = getattr(self, f"{stage}_metrics")
        # Update metrics using the final predicted torque vs measured torque
        for name, metric in metrics.items():
            metric.update(y_pred_final_torque, y_measured_torque) 
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'), sync_dist=True)
        # Also log tau_phys magnitude if used, for diagnostics
        if self.use_residual:
            self.log(f"{stage}_tau_phys_mean_abs", torch.mean(torch.abs(tau_phys)), on_epoch=True, on_step=False, sync_dist=True)

        return {"loss": loss, "preds": y_pred_final_torque, "targets": y_measured_torque}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")["loss"] # Return only loss for training

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "val")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, batch_idx, "test")
        self.test_step_outputs.append({'preds': step_output['preds'], 'targets': step_output['targets']})
        return step_output
    
    def on_train_epoch_end(self):
        self._log_metrics(self.train_metrics, "train")
    
    def on_validation_epoch_end(self):
        self._log_metrics(self.val_metrics, "val")
                
    def on_test_epoch_end(self):
        self._log_metrics(self.test_metrics, "test")
        if self.test_step_outputs:
            all_preds = torch.cat([out['preds'] for out in self.test_step_outputs]).cpu().numpy()
            all_targets = torch.cat([out['targets'] for out in self.test_step_outputs]).cpu().numpy()
            self.all_test_preds = all_preds
            self.all_test_targets = all_targets
        self.test_step_outputs.clear()
                
    def _log_metrics(self, metrics_dict: Dict[str, Any], stage: str): # Renamed metrics to metrics_dict
        for name, metric in metrics_dict.items(): # Use metrics_dict here
            try:
                metric = metric.to(self.device)
                computed_metric = metric.compute()
                if computed_metric is not None:
                    self.log(f"{stage}_{name}", computed_metric, prog_bar=True, sync_dist=True, logger=True) # sync_dist=True for multi-GPU
                metric.reset() # Reset metric at the end of epoch
            except Exception as e:
                 print(f"Error computing/logging metric {stage}_{name}: {e}")
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW( # Using AdamW as it's generally a good default
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        def lr_lambda_fn(current_step: int):
            # Ensure trainer and its attributes are available
            if not hasattr(self.trainer, 'num_training_batches') or not hasattr(self.trainer, 'max_epochs'):
                # Fallback if trainer attributes not ready (e.g. during initial check by PL)
                # This might happen if PL tries to call this too early. A simple linear decay or constant LR could be a fallback.
                # However, for a real run, these should be populated.
                print("Warning: Trainer attributes for LR scheduler not fully available yet. Using LR=1.0 for this step.")
                return 1.0 
                
            num_warmup_steps = self.hparams.warmup_epochs * self.trainer.num_training_batches
            if num_warmup_steps == 0: # Avoid division by zero if warmup_epochs is 0
                num_warmup_steps = 1 # Effectively no warmup or very short
            
            num_training_steps = self.trainer.max_epochs * self.trainer.num_training_batches
            if num_training_steps <= num_warmup_steps: # Avoid division by zero or negative in cosine if training is shorter than warmup
                return 1.0 # Or handle as error, or just linear decay

            if current_step < num_warmup_steps:
                return float(current_step) / float(num_warmup_steps)
            
            # Cosine decay part
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        } 