from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from src.models.mlp import MLP


class ActuatorModel(pl.LightningModule):
    """
    Lightning module for actuator torque prediction using MLP.
    
    This model predicts torque based on current angle, desired angle, and their derivatives.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 128, 64],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Initialize ActuatorModel.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            batch_size: Batch size for training
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        
        # Build model
        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,  # Convert to list of integers
            output_dim=1,  # Torque is a scalar
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        
        # Initialize metrics for different stages
        metrics = {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "r2": R2Score()
        }
        
        self.train_metrics = metrics.copy()
        self.val_metrics = metrics.copy()
        self.test_metrics = metrics.copy()

        # Initialize lists to store test outputs
        self.test_step_outputs = []
    
    def on_fit_start(self):
        """Move metrics to the same device as the model."""
        device = next(self.parameters()).device
        for metric_dict in [self.train_metrics, self.val_metrics, self.test_metrics]:
            for name, metric in metric_dict.items():
                metric_dict[name] = metric.to(device)
    
    def on_test_start(self):
        """Move metrics to the same device as the model."""
        device = next(self.parameters()).device
        for name, metric in self.test_metrics.items():
            self.test_metrics[name] = metric.to(device)
    
    def forward(self, x):
        """Forward pass through the model."""
        # Input shape: [batch_size, input_dim]
        # Output shape: [batch_size, 1]
        return self.model(x)
    
    def step(self, batch, batch_idx, stage):
        """
        General step function for training, validation, and testing.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            stage: 'train', 'val', or 'test'
        
        Returns:
            Dictionary containing loss and predictions
        """
        x, y = batch
        y_hat = self(x).squeeze()  # Remove the last dimension to match y
        
        # Get metrics for the current stage
        metrics = getattr(self, f"{stage}_metrics")
        
        # Compute loss
        loss = F.mse_loss(y_hat, y)
        
        # Update metrics
        for name, metric in metrics.items():
            metric(y_hat, y)
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == 'train'), sync_dist=True)
        
        return {"loss": loss, "preds": y_hat, "targets": y}
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        return self.step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Get predictions and targets from the general step function
        step_output = self.step(batch, batch_idx, "test")
        # Keep track of predictions and targets for the epoch end plot
        self.test_step_outputs.append({'preds': step_output['preds'], 'targets': step_output['targets']})
        return step_output # Return the full output for metric calculation
    
    def on_train_epoch_end(self):
        """Log training metrics at the end of training."""
        self._log_metrics(self.train_metrics, "train")
    
    def on_validation_epoch_end(self):
        """Log validation metrics at the end of validation."""
        self._log_metrics(self.val_metrics, "val")
                
    def on_test_epoch_end(self):
        """Log test metrics and aggregate predictions/targets."""
        # Log aggregated metrics (MSE, MAE, R2)
        self._log_metrics(self.test_metrics, "test")

        # Aggregate predictions and targets from all test steps
        if self.test_step_outputs:
            all_preds = torch.cat([out['preds'] for out in self.test_step_outputs]).cpu().numpy()
            all_targets = torch.cat([out['targets'] for out in self.test_step_outputs]).cpu().numpy()
            # Store aggregated results for the callback to use
            self.all_test_preds = all_preds
            self.all_test_targets = all_targets
        
        # Clear the stored step outputs to free memory
        self.test_step_outputs.clear()
                
    def _log_metrics(self, metrics, stage):
        """Log metrics."""
        for name, metric in metrics.items():
            try:
                # Ensure metric is on the correct device before compute
                metric = metric.to(self.device)
                computed_metric = metric.compute()
                if computed_metric is not None:
                    self.log(f"{stage}_{name}", computed_metric, prog_bar=True, sync_dist=False, logger=True)
                    # metric.reset() # Removed: Let PyTorch Lightning handle metric resets
            except Exception as e:
                 # It's good practice to log the error if compute fails
                 print(f"Error computing/logging metric {stage}_{name}: {e}")
                
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        } 