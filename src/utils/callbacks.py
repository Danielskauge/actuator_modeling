import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
import wandb


class ModelExporter(Callback):
    """
    Callback for exporting model to TorchScript and ONNX formats.
    
    Exports the model at the end of training.
    """
    
    def __init__(
        self,
        output_dir: str = "models/exported",
        export_torchscript: bool = True,
        export_onnx: bool = True
    ):
        """
        Initialize the exporter.
        
        Args:
            output_dir: Directory to save exported models
            export_torchscript: Whether to export to TorchScript
            export_onnx: Whether to export to ONNX
        """
        super().__init__()
        self.output_dir = output_dir
        self.export_torchscript = export_torchscript
        self.export_onnx = export_onnx
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_train_end(self, trainer, pl_module):
        """Export model at the end of training."""
        # Get a sample input
        device = pl_module.device
        sample_input = torch.randn(1, pl_module.hparams.input_dim, device=device)
        
        # Export to TorchScript
        if self.export_torchscript:
            try:
                script_path = os.path.join(self.output_dir, "model.pt")
                script = pl_module.to_torchscript()
                torch.jit.save(script, script_path)
                print(f"TorchScript model saved to {script_path}")
            except Exception as e:
                print(f"TorchScript export failed: {e}")
                print("Saving the model state dict instead...")
                state_dict_path = os.path.join(self.output_dir, "model_state_dict.pt")
                torch.save(pl_module.state_dict(), state_dict_path)
                print(f"Model state dict saved to {state_dict_path}")
        
        # Export to ONNX
        if self.export_onnx:
            try:
                import onnx
                
                onnx_path = os.path.join(self.output_dir, "model.onnx")
                torch.onnx.export(
                    pl_module,
                    sample_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"}
                    }
                )
                
                # Verify the ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"ONNX model saved to {onnx_path}")
                
            except ImportError:
                print("ONNX export failed: onnx package not found.")
            except Exception as e:
                print(f"ONNX export failed: {e}")


class EarlySummary(Callback):
    """
    Callback to print model summary early in training.
    
    This helps verify the model architecture before full training.
    """
    
    def on_fit_start(self, trainer, pl_module):
        """Print model summary at start of training."""
        # Create a string representation of the model
        print("\nActuator Model Summary:")
        print(f"Input dimension: {pl_module.hparams.input_dim}")
        print(f"Hidden dimensions: {pl_module.hparams.hidden_dims}")
        print(f"Total parameters: {sum(p.numel() for p in pl_module.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in pl_module.parameters() if p.requires_grad)}")
        print("\n") 


class TestPredictionPlotter(Callback):
    """
    Callback to visualize model predictions vs. ground truth on the test set
    and log plots to WandB at the end of testing. Generates:
    1. Scatter plot of Predicted vs. Actual Torque.
    2. Time series plot of Predicted and Actual Torque over sample index.
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log prediction plots after testing finishes."""
        print("\nGenerating test set prediction plots...")

        # Check if aggregated predictions and targets are available
        if not hasattr(pl_module, 'all_test_preds') or not hasattr(pl_module, 'all_test_targets'):
            print("Aggregated test predictions/targets not found in the model. Skipping plots.")
            return

        preds = pl_module.all_test_preds
        targets = pl_module.all_test_targets

        # Ensure we have data
        if preds is None or targets is None or len(preds) == 0:
            print("No test predictions or targets available to plot.")
            return

        # --- 1. Scatter Plot ---
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.scatter(targets, preds, alpha=0.5, label='Predictions')
        lims = [
            np.min([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
            np.max([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
        ]
        ax_scatter.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
        ax_scatter.set_xlabel("Actual Torque")
        ax_scatter.set_ylabel("Predicted Torque")
        ax_scatter.set_title("Test Set: Predicted vs. Actual Torque")
        ax_scatter.legend()
        ax_scatter.grid(True)
        ax_scatter.set_aspect('equal', adjustable='box')

        # --- 2. Time Series Plot ---
        fig_ts, ax_ts = plt.subplots(figsize=(15, 5))
        sample_indices = np.arange(len(targets))
        ax_ts.plot(sample_indices, targets, label='Actual Torque', linewidth=1.5)
        ax_ts.plot(sample_indices, preds, label='Predicted Torque', alpha=0.8, linewidth=1.5)
        ax_ts.set_xlabel("Sample Index")
        ax_ts.set_ylabel("Torque")
        ax_ts.set_title("Test Set: Torque Prediction Over Time/Index")
        ax_ts.legend()
        ax_ts.grid(True)

        # Log plots to WandB
        if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
            print("Logging plots to WandB...")
            try:
                trainer.logger.experiment.log({
                    "Test Set Scatter Plot": wandb.Image(fig_scatter),
                    "Test Set Time Series Plot": wandb.Image(fig_ts)
                })
                print("Plots logged.")
            except Exception as e:
                print(f"Error logging plots to WandB: {e}")
        else:
            print("WandB logger not available. Cannot log plots.")

        plt.close(fig_scatter) # Close the scatter figure
        plt.close(fig_ts)      # Close the time series figure

        # Clean up stored tensors in module
        if hasattr(pl_module, 'all_test_preds'):
             delattr(pl_module, 'all_test_preds')
        if hasattr(pl_module, 'all_test_targets'):
             delattr(pl_module, 'all_test_targets')

