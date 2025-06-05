import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
import wandb
import pandas as pd


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
    3. Residuals (error) vs. time plot.
    4. Histogram of residuals (prediction errors).
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("\n[Callback] Generating test set prediction plots...")

        # Check if aggregated predictions, targets, and timestamps are available
        if not hasattr(pl_module, 'all_test_preds') or \
           not hasattr(pl_module, 'all_test_targets') or \
           not hasattr(pl_module, 'all_test_timestamps'):
            print("[Callback] Aggregated test predictions/targets/timestamps not found in the model. Skipping plots.")
            return

        preds_tensor = pl_module.all_test_preds
        targets_tensor = pl_module.all_test_targets
        timestamps_tensor = pl_module.all_test_timestamps

        # Log tensor shapes and stats for debugging
        print(f"[Callback] preds_tensor shape: {getattr(preds_tensor, 'shape', None)}")
        print(f"[Callback] targets_tensor shape: {getattr(targets_tensor, 'shape', None)}")
        print(f"[Callback] timestamps_tensor shape: {getattr(timestamps_tensor, 'shape', None)}")

        # Move to CPU and convert to NumPy
        try:
            # Flatten sequence outputs into 1D
            preds_np = preds_tensor.cpu().numpy().reshape(-1)
            targets_np = targets_tensor.cpu().numpy().reshape(-1)
            timestamps_np = timestamps_tensor.cpu().numpy().reshape(-1)
        except Exception as e:
            print(f"[Callback] Error converting tensors to NumPy or flattening: {e}. Skipping plots.")
            return

        # Log basic stats
        def log_stats(arr, name):
            print(f"[Callback] {name}: shape={arr.shape}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, NaNs={np.isnan(arr).sum()}")
        log_stats(preds_np, 'preds_np')
        log_stats(targets_np, 'targets_np')
        log_stats(timestamps_np, 'timestamps_np')

        if not (preds_np.ndim == 1 and targets_np.ndim == 1 and timestamps_np.ndim == 1 and \
                len(preds_np) == len(targets_np) == len(timestamps_np)):
            print(f"[Callback] Dimension mismatch after squeeze or not 1D: preds {preds_np.shape}, targets {targets_np.shape}, timestamps {timestamps_np.shape}. Skipping plots.")
            return

        # --- Create DataFrame and sort by timestamp for time series plot ---
        plot_df = pd.DataFrame({
            'timestamp': timestamps_np,
            'actual_torque': targets_np,
            'predicted_torque': preds_np
        })
        plot_df_sorted = plot_df.sort_values(by='timestamp').reset_index(drop=True)

        # --- 1. Scatter Plot (Predicted vs. Actual Torque) ---
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
        ax_scatter.scatter(targets_np, preds_np, alpha=0.5, label='Predictions')
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

        # --- 2. Time Series Plot (Predicted and Actual Torque over Time) ---
        fig_ts, ax_ts = plt.subplots(figsize=(15, 5))
        ax_ts.plot(plot_df_sorted['timestamp'], plot_df_sorted['actual_torque'], label='Actual Torque', linewidth=1.5)
        ax_ts.plot(plot_df_sorted['timestamp'], plot_df_sorted['predicted_torque'], label='Predicted Torque', alpha=0.8, linewidth=1.5)
        ax_ts.set_xlabel("Time (seconds since epoch)")
        ax_ts.set_ylabel("Torque")
        ax_ts.set_title("Test Set: Torque Prediction Over Time (Sorted by Timestamp)")
        ax_ts.legend()
        ax_ts.grid(True)

        # --- 3. Residuals vs. Time Plot ---
        residuals = plot_df_sorted['actual_torque'] - plot_df_sorted['predicted_torque']
        fig_resid, ax_resid = plt.subplots(figsize=(15, 5))
        ax_resid.plot(plot_df_sorted['timestamp'], residuals, label='Residual (Actual - Predicted)', color='purple', linewidth=1.2)
        ax_resid.set_xlabel("Time (seconds since epoch)")
        ax_resid.set_ylabel("Residual (Torque Error)")
        ax_resid.set_title("Test Set: Residuals (Actual - Predicted) Over Time")
        ax_resid.legend()
        ax_resid.grid(True)

        # --- 4. Histogram of Residuals ---
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        ax_hist.hist(residuals, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel("Residual (Actual - Predicted Torque)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Histogram of Prediction Errors (Residuals)")
        ax_hist.grid(True)

        # Log plots to WandB
        if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
            print("[Callback] Logging plots to WandB...")
            try:
                trainer.logger.experiment.log({
                    "Test Set Scatter Plot": wandb.Image(fig_scatter),
                    "Test Set Time Series Plot": wandb.Image(fig_ts),
                    "Test Set Residuals vs Time": wandb.Image(fig_resid),
                    "Test Set Residuals Histogram": wandb.Image(fig_hist)
                })
                print("[Callback] Plots logged.")
            except Exception as e:
                print(f"[Callback] Error logging plots to WandB: {e}")
        else:
            print("[Callback] WandB logger not available. Cannot log plots.")

        plt.close(fig_scatter)
        plt.close(fig_ts)
        plt.close(fig_resid)
        plt.close(fig_hist)

        # Clean up stored tensors in module
        if hasattr(pl_module, 'all_test_preds'):
             delattr(pl_module, 'all_test_preds')
        if hasattr(pl_module, 'all_test_targets'):
             delattr(pl_module, 'all_test_targets')
        if hasattr(pl_module, 'all_test_timestamps'):
             delattr(pl_module, 'all_test_timestamps')


class DatasetVisualizationCallback(Callback):
    """
    Callback to visualize and log dataset-wide information at the start of training.
    Logs time series plots of actual torques and Y-axis acceleration from the full dataset.
    Also logs histograms of these quantities for further debugging.
    """

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("\n[Callback] Generating dataset-wide visualization plots...")

        datamodule = trainer.datamodule
        if not datamodule:
            print("[Callback] DataModule not found in Trainer. Skipping dataset visualization.")
            return

        if not hasattr(datamodule, 'datasets_by_group') or not datamodule.datasets_by_group:
            try:
                datamodule.prepare_data()
                if not datamodule.datasets_by_group:
                    print("[Callback] No datasets found in DataModule after calling prepare_data. Skipping dataset visualization.")
                    return
            except Exception as e:
                print(f"[Callback] Error during datamodule.prepare_data(): {e}. Skipping dataset visualization.")
                return

        target_col_name = "torque"
        accel_col_name = datamodule.hparams.main_accel_axis

        all_torques_list = []
        all_accel_y_list = []

        for group_id, datasets_in_group in datamodule.datasets_by_group.items():
            for dataset_instance in datasets_in_group:
                if hasattr(dataset_instance, 'data_df') and isinstance(dataset_instance.data_df, pd.DataFrame):
                    df = dataset_instance.data_df
                    if target_col_name in df.columns:
                        all_torques_list.append(df[target_col_name])
                    else:
                        print(f"[Callback] Warning: Target column '{target_col_name}' not in data_df for a dataset in group '{group_id}'.")
                    if accel_col_name in df.columns:
                        all_accel_y_list.append(df[accel_col_name])
                    else:
                        print(f"[Callback] Warning: Acceleration column '{accel_col_name}' not in data_df for a dataset in group '{group_id}'.")
                else:
                    print(f"[Callback] Warning: data_df not found or not a DataFrame in a dataset in group '{group_id}'.")

        if not all_torques_list and not all_accel_y_list:
            print("[Callback] No data extracted for torque or acceleration plots. Skipping.")
            return

        # --- Plotting Actual Torques (Time Series) ---
        if all_torques_list:
            try:
                concatenated_torques = pd.concat(all_torques_list, ignore_index=True)
                fig_torque, ax_torque = plt.subplots(figsize=(15, 5))
                ax_torque.plot(concatenated_torques.index, concatenated_torques.values, label=f'Actual Torque ({target_col_name})', linewidth=1)
                ax_torque.set_xlabel("Global Sample Index (Concatenated)")
                ax_torque.set_ylabel("Torque")
                ax_torque.set_title("Full Dataset: Actual Torques Time Series")
                ax_torque.legend()
                ax_torque.grid(True)
                if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                    trainer.logger.experiment.log({"Dataset/Actual Torques": wandb.Image(fig_torque)})
                plt.close(fig_torque)
                print("[Callback] Actual torques plot generated and logged.")
            except Exception as e:
                print(f"[Callback] Error generating or logging actual torques plot: {e}")
        else:
            print("[Callback] No torque data to plot.")

        # --- Plotting Y-axis Acceleration (Time Series) ---
        if all_accel_y_list:
            try:
                concatenated_accel_y = pd.concat(all_accel_y_list, ignore_index=True)
                fig_accel, ax_accel = plt.subplots(figsize=(15, 5))
                ax_accel.plot(concatenated_accel_y.index, concatenated_accel_y.values, label=f'Y-axis Acceleration ({accel_col_name})', linewidth=1, color='green')
                ax_accel.set_xlabel("Global Sample Index (Concatenated)")
                ax_accel.set_ylabel("Acceleration (m/s^2)")
                ax_accel.set_title(f"Full Dataset: {accel_col_name} Time Series")
                ax_accel.legend()
                ax_accel.grid(True)
                if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                    trainer.logger.experiment.log({"Dataset/Y-axis Acceleration": wandb.Image(fig_accel)})
                plt.close(fig_accel)
                print("[Callback] Y-axis acceleration plot generated and logged.")
            except Exception as e:
                print(f"[Callback] Error generating or logging Y-axis acceleration plot: {e}")
        else:
            print("[Callback] No Y-axis acceleration data to plot.")

        # --- Histogram of Actual Torques ---
        if all_torques_list:
            try:
                concatenated_torques = pd.concat(all_torques_list, ignore_index=True)
                fig_torque_hist, ax_torque_hist = plt.subplots(figsize=(8, 5))
                ax_torque_hist.hist(concatenated_torques.values, bins=50, color='blue', alpha=0.7, edgecolor='black')
                ax_torque_hist.set_xlabel("Actual Torque")
                ax_torque_hist.set_ylabel("Count")
                ax_torque_hist.set_title("Histogram of Actual Torques (Full Dataset)")
                ax_torque_hist.grid(True)
                if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                    trainer.logger.experiment.log({"Dataset/Actual Torques Histogram": wandb.Image(fig_torque_hist)})
                plt.close(fig_torque_hist)
                print("[Callback] Actual torques histogram generated and logged.")
            except Exception as e:
                print(f"[Callback] Error generating or logging actual torques histogram: {e}")

        # --- Histogram of Y-axis Acceleration ---
        if all_accel_y_list:
            try:
                concatenated_accel_y = pd.concat(all_accel_y_list, ignore_index=True)
                fig_accel_hist, ax_accel_hist = plt.subplots(figsize=(8, 5))
                ax_accel_hist.hist(concatenated_accel_y.values, bins=50, color='green', alpha=0.7, edgecolor='black')
                ax_accel_hist.set_xlabel(f"{accel_col_name} (Y-axis Acceleration)")
                ax_accel_hist.set_ylabel("Count")
                ax_accel_hist.set_title(f"Histogram of {accel_col_name} (Full Dataset)")
                ax_accel_hist.grid(True)
                if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                    trainer.logger.experiment.log({f"Dataset/{accel_col_name} Histogram": wandb.Image(fig_accel_hist)})
                plt.close(fig_accel_hist)
                print(f"[Callback] {accel_col_name} histogram generated and logged.")
            except Exception as e:
                print(f"[Callback] Error generating or logging {accel_col_name} histogram: {e}")

        print("[Callback] Dataset visualization plots finished.")


class DebugTestTimeSeriesPlotter(Callback):
    """
    A new callback to independently plot predicted vs. actual torques from the test set.
    This is intended for debugging or cross-verifying other plotting callbacks.
    Generates a time series plot of Predicted and Actual Torque.
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log prediction plots after testing finishes."""
        print("\nGenerating DEBUG test set time series comparison plot...")

        if not hasattr(pl_module, 'all_test_preds') or \
           not hasattr(pl_module, 'all_test_targets') or \
           not hasattr(pl_module, 'all_test_timestamps'): # Check for timestamps
            print("DEBUG Plotter: Aggregated test predictions/targets/timestamps not found. Skipping plot.")
            return

        preds_tensor = pl_module.all_test_preds
        targets_tensor = pl_module.all_test_targets
        timestamps_tensor = pl_module.all_test_timestamps # Get timestamps

        if preds_tensor is None or targets_tensor is None or timestamps_tensor is None or \
           preds_tensor.numel() == 0 or targets_tensor.numel() == 0 or timestamps_tensor.numel() == 0:
            print("DEBUG Plotter: No test predictions, targets, or timestamps available. Skipping plot.")
            return
        
        # Move to CPU and convert to NumPy
        try:
            # Flatten sequence outputs into 1D
            preds_np = preds_tensor.cpu().numpy().reshape(-1)
            targets_np = targets_tensor.cpu().numpy().reshape(-1)
            timestamps_np = timestamps_tensor.cpu().numpy().reshape(-1)
        except Exception as e:
            print(f"DEBUG Plotter: Error converting tensors to NumPy or flattening: {e}. Skipping plot.")
            return

        if not (preds_np.ndim == 1 and targets_np.ndim == 1 and timestamps_np.ndim == 1 and \
                len(preds_np) == len(targets_np) == len(timestamps_np)):
            print(f"DEBUG Plotter: Dimension mismatch or not 1D. Skipping plots.")
            return

        # --- Create DataFrame and sort by timestamp ---
        debug_plot_df = pd.DataFrame({
            'timestamp': timestamps_np,
            'actual_torque': targets_np,
            'predicted_torque': preds_np
        })
        debug_plot_df_sorted = debug_plot_df.sort_values(by='timestamp').reset_index(drop=True)

        # --- Time Series Plot ---
        fig_debug_ts, ax_debug_ts = plt.subplots(figsize=(17, 6)) # Slightly different size
        ax_debug_ts.plot(debug_plot_df_sorted['timestamp'], debug_plot_df_sorted['actual_torque'], label='Actual Torque (Debug)', linewidth=1.7, linestyle='--') # Different style
        ax_debug_ts.plot(debug_plot_df_sorted['timestamp'], debug_plot_df_sorted['predicted_torque'], label='Predicted Torque (Debug)', alpha=0.7, linewidth=1.7, linestyle=':') # Different style
        ax_debug_ts.set_xlabel("Time (seconds since epoch)")
        ax_debug_ts.set_ylabel("Torque")
        ax_debug_ts.set_title("DEBUG Test Set: Torque Prediction Over Time (Sorted by Timestamp - New Plotter)")
        ax_debug_ts.legend()
        ax_debug_ts.grid(True)

        if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
            print("DEBUG Plotter: Logging new time series plot to WandB...")
            try:
                trainer.logger.experiment.log({
                    "Test Set Time Series Plot (DEBUG)": wandb.Image(fig_debug_ts)
                })
                print("DEBUG Plotter: Plot logged to WandB.")
            except Exception as e:
                print(f"DEBUG Plotter: Error logging plot to WandB: {e}")
        else:
            print("DEBUG Plotter: WandB logger not available. Cannot log plot.")

        plt.close(fig_debug_ts)
        
        # This debug plotter does not clean up pl_module.all_test_preds, pl_module.all_test_targets, or pl_module.all_test_timestamps
        # as other callbacks might still need them. The main TestPredictionPlotter handles cleanup.

