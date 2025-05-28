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
    Expects input sequences of length 2. Number of features per step is 3 after removing target_ang_vel.
    Feature order for tau_phys (if calculated from sequence): current_angle, target_angle, current_ang_vel.
    """
    
    def __init__(
        self,
        model_type: str,
        input_dim: int,         # Expect 3 from DataModule (ActuatorDataset.INPUT_DIM)
        # Normalization stats from DataModule
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        # sampling_frequency: float, # REMOVED - Not needed if target_ang_vel for physics is 0
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
            input_dim: Dimension of input features (per timestep for GRU). Should be 3.
            input_mean: Mean of input features for normalization.
            input_std: Standard deviation of input features for normalization.
            target_mean: Mean of target variable for normalization.
            target_std: Standard deviation of target variable for normalization.
            # sampling_frequency: REMOVED
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
        self.save_hyperparameters(ignore=["kwargs", "input_mean", "input_std", "target_mean", "target_std"])

        # Register normalization stats as buffers
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

        if self.hparams.input_dim != 3: # Check for 3 features now
            print(f"Warning: ActuatorModel input_dim={self.hparams.input_dim}, but expected 3 after removing target_ang_vel.")
        
        # self.dt = 1.0 / self.hparams.sampling_frequency # REMOVED

        self.model_type = self.hparams.model_type.lower()
        self.use_residual = use_residual
        self.k_spring = k_spring
        self.theta0 = theta0
        self.kp_phys = kp_phys
        self.kd_phys = kd_phys
        self.loss_diff_weight = loss_diff_weight

        if self.model_type == "mlp":
            mlp_input_dim_effective = self.hparams.input_dim * 2 # seq_len=2 -> 3*2=6
            print(f"MLP model, effective input_dim (flattened sequence): {mlp_input_dim_effective}")
            self.model = MLP(
                input_dim=mlp_input_dim_effective,
                hidden_dims=self.hparams.mlp_hidden_dims,
                output_dim=1, activation=self.hparams.mlp_activation,
                dropout=self.hparams.mlp_dropout, use_batch_norm=self.hparams.mlp_use_batch_norm,
            )
        elif self.model_type == "gru":
            self.model = GRUModel(
                input_dim=self.hparams.input_dim, # GRU input_dim is per step (3)
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
        x shape for GRU: [batch_size, seq_len, input_dim] -> output [batch_size, 1]
        x shape for MLP: [batch_size, seq_len, input_dim], will be flattened.
        """
        if self.model_type == "mlp":
            batch_size = x.size(0)
            # x has shape [batch, sequence_length, features_per_step]
            # MLP expects [batch, flattened_features]
            x = x.view(batch_size, -1) 
        return self.model(x)

    def _calculate_tau_phys(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """
        Calculates physics-based torque (tau_phys) using the input sequence x_sequence.
        x_sequence shape: [batch_size, sequence_length, features_per_step]
        Assumes features_per_step are [current_angle_rad, target_angle_rad, current_ang_vel_rad_s]
        Indices for relevant features within each step:
        - 0: current_angle_rad
        - 1: target_angle_rad
        - 2: current_ang_vel_rad_s
        For the kd_phys term, target angular velocity is assumed to be 0.
        """
        if not self.use_residual: # Should not be called if not in residual mode
            return torch.zeros(x_sequence.size(0), 1, device=x_sequence.device)

        # Features from the LATEST time step in the sequence (k)
        features_current_step = x_sequence[:, -1, :] 
        current_angle_k = features_current_step[:, 0]
        target_angle_k = features_current_step[:, 1]
        current_ang_vel_k = features_current_step[:, 2]

        error_pos_k = target_angle_k - current_angle_k
        
        # Target angular velocity for the physics model's kd_phys term is assumed to be 0.
        target_ang_vel_for_physics_term = torch.zeros_like(current_ang_vel_k)
        error_vel_k = target_ang_vel_for_physics_term - current_ang_vel_k # Effectively -current_ang_vel_k
        
        tau_spring_comp = self.k_spring * (current_angle_k - self.theta0)
        tau_pd_comp = self.kp_phys * error_pos_k + self.kd_phys * error_vel_k
        
        tau_phys = tau_spring_comp + tau_pd_comp
        return tau_phys.unsqueeze(1) # Shape [batch_size, 1]

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """
        General step function for training, validation, and testing.
        """
        x, y_measured_torque_raw = batch # x raw, y_measured_torque_raw is in physical units
        y_measured_torque_raw = y_measured_torque_raw.squeeze(-1) # Shape [batch]

        # 1. Normalize inputs (x)
        # Ensure input_mean and input_std are correctly shaped for broadcasting over (batch, seq_len, features)
        # Current stats are (features,) or (1, features). Reshape to (1, 1, features) if necessary.
        mean_x = self.input_mean.view(1, 1, -1) if self.input_mean.ndim == 1 else self.input_mean
        std_x = self.input_std.view(1, 1, -1) if self.input_std.ndim == 1 else self.input_std
        x_normalized = (x - mean_x) / (std_x + 1e-7) # Epsilon for stability

        # 2. Model makes prediction using normalized inputs. Output is in normalized scale.
        y_hat_model_output_normalized = self(x_normalized).squeeze(-1) # Output shape: [batch]

        # 3. Normalize the raw target torque for loss calculation
        # target_mean/std are likely (1,) or scalar. Ensure they broadcast with y_measured_torque_raw [batch]
        mean_y = self.target_mean.squeeze()
        std_y = self.target_std.squeeze()
        y_target_for_loss_normalized = (y_measured_torque_raw - mean_y) / (std_y + 1e-7)

        # Prepare target for loss, and final prediction for metrics (initially in normalized scale)
        final_target_for_loss_normalized = y_target_for_loss_normalized
        y_pred_final_normalized_for_metrics = y_hat_model_output_normalized
        
        tau_phys_raw_for_logging = None # For logging original scale tau_phys

        if self.use_residual:
            # Calculate tau_phys using RAW (unnormalized) x, as physics model expects physical units
            tau_phys_raw = self._calculate_tau_phys(x).squeeze(-1) # Shape [batch], physical units
            tau_phys_raw_for_logging = tau_phys_raw # Save for logging
            
            # Normalize tau_phys_raw to the same scale as the target
            tau_phys_normalized = (tau_phys_raw - mean_y) / (std_y + 1e-7)
            
            # Model learns to predict the residual in the normalized space
            final_target_for_loss_normalized = y_target_for_loss_normalized - tau_phys_normalized
            
            # For metrics, add the normalized model output to the normalized physics component
            y_pred_final_normalized_for_metrics = y_hat_model_output_normalized + tau_phys_normalized
        
        # 4. Calculate loss using normalized prediction and normalized target
        loss = F.mse_loss(y_hat_model_output_normalized, final_target_for_loss_normalized)
        
        # First-difference loss for smoothing (applied to the model's direct output in normalized space)
        if self.loss_diff_weight > 0 and len(y_hat_model_output_normalized) > 1:
            diff_y_hat_model_output_normalized = torch.diff(y_hat_model_output_normalized)
            # Smooth the target the model is trying to learn (which is also normalized)
            diff_final_target_for_loss_normalized = torch.diff(final_target_for_loss_normalized) 
            loss_diff = F.mse_loss(diff_y_hat_model_output_normalized, diff_final_target_for_loss_normalized)
            loss = loss + self.loss_diff_weight * loss_diff
        
        # 5. Denormalize the final prediction back to physical scale for metrics
        y_pred_final_denormalized_for_metrics = y_pred_final_normalized_for_metrics * (std_y + 1e-7) + mean_y
        
        # 6. Update metrics using denormalized predictions and raw measured torque
        metrics = getattr(self, f"{stage}_metrics")
        for name, metric in metrics.items():
            metric.update(y_pred_final_denormalized_for_metrics, y_measured_torque_raw) 
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'), sync_dist=True)
        if self.use_residual and tau_phys_raw_for_logging is not None:
            # Log the mean absolute value of the ORIGINAL, physical-scale tau_phys
            self.log(f"{stage}_tau_phys_mean_abs", torch.mean(torch.abs(tau_phys_raw_for_logging)), on_epoch=True, on_step=False, sync_dist=True)

        return {"loss": loss, "preds": y_pred_final_denormalized_for_metrics, "targets": y_measured_torque_raw}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")["loss"]

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "val")
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_output = self.step(batch, batch_idx, "test")
        # Store predictions and targets if a callback needs them (e.g., TestPredictionPlotter)
        self.test_step_outputs.append({'preds': step_output['preds'].detach(), 'targets': step_output['targets'].detach()})
        return step_output
    
    def on_train_epoch_end(self):
        self._log_epoch_metrics(self.train_metrics, "train")
    
    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self.val_metrics, "val")
                
    def on_test_epoch_end(self):
        self._log_epoch_metrics(self.test_metrics, "test")
        if hasattr(self, 'test_step_outputs') and self.test_step_outputs: # Check existence
            # For TestPredictionPlotter or similar callbacks
            # Ensure these are attributes expected by such callbacks
            try:
                self.all_test_preds = torch.cat([out['preds'] for out in self.test_step_outputs])
                self.all_test_targets = torch.cat([out['targets'] for out in self.test_step_outputs])
            except RuntimeError as e:
                print(f"Warning: Could not concatenate test outputs (possibly empty): {e}")
                self.all_test_preds = torch.empty(0) # Ensure attributes exist even if empty
                self.all_test_targets = torch.empty(0)
            self.test_step_outputs.clear() # Clear for next test run (if any)
                
    def _log_epoch_metrics(self, current_metrics_dict: Dict[str, Any], stage: str):
        for name, metric_instance in current_metrics_dict.items(): 
            try:
                # metric_instance = metric_instance.to(self.device) # Already moved in on_fit_start/on_test_start
                computed_metric_val = metric_instance.compute()
                if computed_metric_val is not None:
                    self.log(f"{stage}_{name}_epoch", computed_metric_val, prog_bar=True, sync_dist=True, logger=True, on_step=False, on_epoch=True)
                metric_instance.reset() 
            except Exception as e:
                 print(f"Error computing/logging epoch metric {stage}_{name}_epoch: {e}")
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # LR Scheduler: Linear warmup then cosine decay
        def lr_lambda_fn(current_step: int):
            # Ensure trainer and its attributes are available
            # Access num_training_batches via self.trainer.num_training_batches (PL >=1.7)
            # or self.trainer.estimated_stepping_batches / self.trainer.max_epochs (older versions or if accumulate_grad_batches > 1)
            # For simplicity, assume num_training_batches is available on trainer. Guard against it. 
            if not hasattr(self.trainer, 'num_training_batches') or self.trainer.num_training_batches == 0 or \
               not hasattr(self.trainer, 'max_epochs') or self.trainer.max_epochs == 0:
                print("Warning: Trainer attributes for LR scheduler (num_training_batches/max_epochs) not fully available or zero. Using LR=1.0 for this step. This might happen during sanity checks.")
                return 1.0 
                
            num_warmup_steps = self.hparams.warmup_epochs * self.trainer.num_training_batches
            if num_warmup_steps == 0: 
                num_warmup_steps = 1 
            
            num_training_steps = self.trainer.max_epochs * self.trainer.num_training_batches
            if num_training_steps <= num_warmup_steps:
                print(f"Warning: Total training steps ({num_training_steps}) <= warmup steps ({num_warmup_steps}). Cosine decay might not behave as expected. Effective LR may be constant or only warmup.")
                # In this case, just maintain the warmup phase or a constant factor if no warmup
                return float(current_step) / float(num_warmup_steps) if current_step < num_warmup_steps else 1.0

            if current_step < num_warmup_steps:
                return float(current_step) / float(num_warmup_steps)
            
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