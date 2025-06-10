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
    1. Scatter plot of Predicted vs. Actual Torque (using ALL predictions).
    2. Time series plot of Predicted and Actual Torque over time (deduplicated).
    3. Residuals (error) vs. time plot (deduplicated).
    4. Histogram of residuals (using ALL predictions for better coverage).
    
    For residual mode, additionally generates:
    5. Analytical torque vs. actual torque scatter plot (using ALL predictions)
    6. Analytical torque vs. actual torque time series (deduplicated)
    7. Residual component analysis plots (mixed: time series deduplicated, statistics use all)
    
    Data Usage Strategy:
    - Time series plots use deduplicated data (one prediction per timestep) for visual clarity
    - Scatter plots and statistical analysis use ALL predictions for better coverage and robustness
    - For many-to-many models, deduplication keeps the prediction with highest position index
      (most historical context) for each unique timestep
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("\n[Callback] Generating test set prediction plots...")

        # Check if aggregated predictions, targets, timestamps, and positions are available
        if not hasattr(pl_module, 'all_test_preds') or \
           not hasattr(pl_module, 'all_test_targets') or \
           not hasattr(pl_module, 'all_test_timestamps') or \
           not hasattr(pl_module, 'all_test_positions'):
            print("[Callback] Aggregated test predictions/targets/timestamps/positions not found in the model. Skipping plots.")
            return

        preds_tensor = pl_module.all_test_preds
        targets_tensor = pl_module.all_test_targets
        timestamps_tensor = pl_module.all_test_timestamps
        positions_tensor = pl_module.all_test_positions

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
            positions_np = positions_tensor.cpu().numpy().reshape(-1)
        except Exception as e:
            print(f"[Callback] Error converting tensors to NumPy or flattening: {e}. Skipping plots.")
            return

        # Log basic stats
        def log_stats(arr, name):
            print(f"[Callback] {name}: shape={arr.shape}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, NaNs={np.isnan(arr).sum()}")
        log_stats(preds_np, 'preds_np')
        log_stats(targets_np, 'targets_np')
        log_stats(timestamps_np, 'timestamps_np')
        log_stats(positions_np, 'positions_np')

        if not (preds_np.ndim == 1 and targets_np.ndim == 1 and timestamps_np.ndim == 1 and positions_np.ndim == 1 and \
                len(preds_np) == len(targets_np) == len(timestamps_np) == len(positions_np)):
            print(f"[Callback] Dimension mismatch after squeeze or not 1D: preds {preds_np.shape}, targets {targets_np.shape}, timestamps {timestamps_np.shape}, positions {positions_np.shape}. Skipping plots.")
            return

        # --- Create DataFrame with all predictions and a deduplicated version ---
        plot_df_all = pd.DataFrame({
            'timestamp': timestamps_np,
            'actual_torque': targets_np,
            'predicted_torque': preds_np,
            'position': positions_np  # Position within sequence (0 to seq_len-1)
        })
        
        # Sort by timestamp first
        plot_df_all_sorted = plot_df_all.sort_values(by='timestamp').reset_index(drop=True)
        
        # Create deduplicated version for time series plots (one prediction per timestep)
        # Keep prediction with highest position (most context) for each timestamp
        plot_df_unique = plot_df_all_sorted.loc[plot_df_all_sorted.groupby('timestamp')['position'].idxmax()].reset_index(drop=True)
        
        print(f"[Callback] Total predictions: {len(plot_df_all_sorted)}")
        print(f"[Callback] Unique timesteps: {len(plot_df_unique)}")
        print(f"[Callback] Average predictions per timestep: {len(plot_df_all_sorted) / len(plot_df_unique):.1f}")
        print(f"[Callback] Position stats (deduplicated) - Min: {plot_df_unique['position'].min()}, Max: {plot_df_unique['position'].max()}, Mean: {plot_df_unique['position'].mean():.1f}")
        
        # Adaptive plot settings based on data size
        total_points = len(plot_df_all_sorted)
        is_large_dataset = total_points > 1_000_000
        
        if is_large_dataset:
            print(f"[Callback] Large dataset detected ({total_points:,} points). Using optimized plot settings.")
            # Smaller plots and lower DPI for large datasets
            scatter_figsize = (10, 10)
            timeseries_figsize = (16, 6)
            slice_figsize = (14, 5)
            component_figsize = (18, 8)
            hist_figsize = (10, 6)
            dpi = 100
            # Sample data for scatter plots to improve performance
            max_scatter_points = 50_000
            if total_points > max_scatter_points:
                scatter_sample_ratio = max_scatter_points / total_points
                print(f"[Callback] Sampling {max_scatter_points:,} points ({scatter_sample_ratio:.1%}) for scatter plots to improve performance.")
            else:
                scatter_sample_ratio = 1.0
        else:
            # Original large, high-res settings for smaller datasets
            scatter_figsize = (12, 12)
            timeseries_figsize = (20, 8)
            slice_figsize = (18, 6)
            component_figsize = (24, 10)
            hist_figsize = (12, 8)
            dpi = 150
            scatter_sample_ratio = 1.0

        # --- 1. Scatter Plot (Predicted vs. Actual Torque) - Use ALL predictions for better coverage ---
        # Sample data for scatter plot if dataset is very large
        if scatter_sample_ratio < 1.0:
            scatter_df = plot_df_all_sorted.sample(frac=scatter_sample_ratio, random_state=42).reset_index(drop=True)
        else:
            scatter_df = plot_df_all_sorted
            
        fig_scatter, ax_scatter = plt.subplots(figsize=scatter_figsize, dpi=dpi)
        ax_scatter.scatter(scatter_df['actual_torque'], scatter_df['predicted_torque'], alpha=0.5, s=1 if is_large_dataset else 20, label=f'Predictions (n={len(scatter_df):,})')
        lims = [
            np.min([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
            np.max([ax_scatter.get_xlim(), ax_scatter.get_ylim()]),
        ]
        ax_scatter.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
        ax_scatter.set_xlabel("Actual Torque", fontsize=14)
        ax_scatter.set_ylabel("Predicted Torque", fontsize=14)
        ax_scatter.set_title("Test Set: Predicted vs. Actual Torque", fontsize=16)
        ax_scatter.legend(fontsize=12, loc='upper left')
        ax_scatter.grid(True)
        ax_scatter.set_aspect('equal', adjustable='box')

        # --- 2. Time Series Plot (Predicted and Actual Torque over Time) - Use deduplicated for clarity ---
        fig_ts, ax_ts = plt.subplots(figsize=timeseries_figsize, dpi=dpi)
        linewidth = 0.8 if is_large_dataset else 1.2
        ax_ts.plot(plot_df_unique['timestamp'], plot_df_unique['actual_torque'], label='Actual Torque', linewidth=linewidth)
        ax_ts.plot(plot_df_unique['timestamp'], plot_df_unique['predicted_torque'], label='Predicted Torque', alpha=0.8, linewidth=linewidth)
        ax_ts.set_xlabel("Time (seconds since epoch)", fontsize=14)
        ax_ts.set_ylabel("Torque", fontsize=14)
        ax_ts.set_title("Test Set: Torque Prediction Over Time (Sorted by Timestamp)", fontsize=16)
        ax_ts.legend(fontsize=12, loc='upper right')
        ax_ts.grid(True)

        # --- 2b. Time Series Slices (3 different portions for better visualization) - Use deduplicated ---
        total_samples = len(plot_df_unique)
        slice_size = total_samples // 3
        
        slice_figs = []
        slice_names = []
        
        for i, (slice_name, start_idx, end_idx) in enumerate([
            ("Early", 0, slice_size),
            ("Middle", slice_size, 2 * slice_size),
            ("Late", 2 * slice_size, total_samples)
        ]):
            if start_idx < total_samples:
                slice_df = plot_df_unique.iloc[start_idx:end_idx]
                
                fig_slice, ax_slice = plt.subplots(figsize=slice_figsize, dpi=dpi)
                ax_slice.plot(slice_df['timestamp'], slice_df['actual_torque'], 
                             label='Actual Torque', linewidth=linewidth, color='blue')
                ax_slice.plot(slice_df['timestamp'], slice_df['predicted_torque'], 
                             label='Predicted Torque', alpha=0.8, linewidth=linewidth, color='orange')
                ax_slice.set_xlabel("Time (seconds since epoch)", fontsize=14)
                ax_slice.set_ylabel("Torque", fontsize=14)
                ax_slice.set_title(f"Test Set: {slice_name} Time Slice ({start_idx}-{end_idx-1}) - Torque Prediction", fontsize=16)
                ax_slice.legend(fontsize=12, loc='upper right')
                ax_slice.grid(True)
                
                slice_figs.append(fig_slice)
                slice_names.append(f"Test Set Time Series Slice ({slice_name})")

        # --- 2c. Short 5-Second Time Slice Plot - Use deduplicated ---
        short_window_duration = 5.0  # seconds
        start_ts = plot_df_unique['timestamp'].min()
        end_ts = start_ts + short_window_duration
        
        # Filter the deduplicated dataframe for the 5-second window
        short_mask = (plot_df_unique['timestamp'] >= start_ts) & (plot_df_unique['timestamp'] <= end_ts)
        short_df = plot_df_unique[short_mask]
        
        short_figs = []
        short_names = []
        
        if len(short_df) > 0:
            fig_short, ax_short = plt.subplots(figsize=slice_figsize, dpi=dpi)
            ax_short.plot(short_df['timestamp'], short_df['actual_torque'], 
                         label='Actual Torque', linewidth=linewidth, color='blue')
            ax_short.plot(short_df['timestamp'], short_df['predicted_torque'], 
                         label='Predicted Torque', alpha=0.8, linewidth=linewidth, color='orange')
            ax_short.set_xlabel("Time (seconds since epoch)", fontsize=14)
            ax_short.set_ylabel("Torque", fontsize=14)
            ax_short.set_title(f"Test Set: First {short_window_duration:.0f}s - Torque Prediction", fontsize=16)
            ax_short.legend(fontsize=12, loc='upper right')
            ax_short.grid(True)
            short_figs.append(fig_short)
            short_names.append(f"Test Set First {short_window_duration:.0f}s")

        # --- 3. Residuals vs. Time Plot - Use deduplicated for time series clarity ---
        residuals_unique = plot_df_unique['actual_torque'] - plot_df_unique['predicted_torque']
        fig_resid, ax_resid = plt.subplots(figsize=timeseries_figsize, dpi=dpi)
        ax_resid.plot(plot_df_unique['timestamp'], residuals_unique, label='Residual (Actual - Predicted)', color='purple', linewidth=linewidth)
        ax_resid.set_xlabel("Time (seconds since epoch)", fontsize=14)
        ax_resid.set_ylabel("Residual (Torque Error)", fontsize=14)
        ax_resid.set_title("Test Set: Residuals (Actual - Predicted) Over Time", fontsize=16)
        ax_resid.legend(fontsize=12, loc='upper right')
        ax_resid.grid(True)

        # --- 4. Histogram of Residuals - Use ALL data for better statistical coverage ---
        residuals_all = plot_df_all_sorted['actual_torque'] - plot_df_all_sorted['predicted_torque']
        fig_hist, ax_hist = plt.subplots(figsize=hist_figsize, dpi=dpi)
        bins = 30 if is_large_dataset else 50  # Fewer bins for large datasets
        ax_hist.hist(residuals_all, bins=bins, color='orange', alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel("Residual (Actual - Predicted Torque)", fontsize=14)
        ax_hist.set_ylabel("Count", fontsize=14)
        ax_hist.set_title(f"Histogram of Prediction Errors (n={len(residuals_all):,} predictions)", fontsize=16)
        ax_hist.grid(True)

        # Store figures to log
        log_dict = {
            "Test Set Scatter Plot": wandb.Image(fig_scatter),
            "Test Set Time Series Plot": wandb.Image(fig_ts),
            "Test Set Residuals vs Time": wandb.Image(fig_resid),
            "Test Set Residuals Histogram": wandb.Image(fig_hist)
        }

        # Add slice plots
        for slice_fig, slice_name in zip(slice_figs, slice_names):
            log_dict[slice_name] = wandb.Image(slice_fig)

        # Add short 5s plot
        for short_fig, short_name in zip(short_figs, short_names):
            log_dict[short_name] = wandb.Image(short_fig)

        # --- RESIDUAL MODE ANALYSIS ---
        if hasattr(pl_module, 'use_residual') and pl_module.use_residual:
            print("[Callback] Residual mode detected. Using stored analytical and residual components...")
            
            # Check if analytical and residual components are available from the model
            if hasattr(pl_module, 'all_test_analytical_torque') and hasattr(pl_module, 'all_test_residual_component'):
                analytical_tensor = pl_module.all_test_analytical_torque
                residual_tensor = pl_module.all_test_residual_component
                
                # Check if individual spring and PD components are available
                spring_tensor = getattr(pl_module, 'all_test_spring_component', None)
                pd_tensor = getattr(pl_module, 'all_test_pd_component', None)
                
                # Convert to numpy and flatten
                try:
                    analytical_np = analytical_tensor.cpu().numpy().reshape(-1)
                    residual_np = residual_tensor.cpu().numpy().reshape(-1)
                    
                    spring_np = None
                    pd_np = None
                    if spring_tensor is not None and pd_tensor is not None:
                        spring_np = spring_tensor.cpu().numpy().reshape(-1)
                        pd_np = pd_tensor.cpu().numpy().reshape(-1)
                    
                    # Ensure same length as other arrays (they should be, but just in case)
                    min_len = min(len(analytical_np), len(residual_np), len(targets_np), len(timestamps_np), len(positions_np))
                    analytical_np = analytical_np[:min_len]
                    residual_np = residual_np[:min_len]
                    targets_subset = targets_np[:min_len]
                    timestamps_subset = timestamps_np[:min_len]
                    positions_subset = positions_np[:min_len]
                    
                    if spring_np is not None and pd_np is not None:
                        spring_np = spring_np[:min_len]
                        pd_np = pd_np[:min_len]
                except Exception as e:
                    print(f"[Callback] Error processing stored analytical/residual components: {e}. Skipping residual mode analysis.")
                    return
            else:
                print("[Callback] Analytical and residual components not found in model. Skipping residual mode analysis.")
                return
            
            log_stats(analytical_np, 'analytical_torque')
            log_stats(residual_np, 'residual_component')
            if spring_np is not None:
                log_stats(spring_np, 'spring_component')
            if pd_np is not None:
                log_stats(pd_np, 'pd_component')
            
            # Create DataFrame for residual mode analysis - both all and deduplicated versions
            residual_df_all = pd.DataFrame({
                'timestamp': timestamps_subset,
                'actual_torque': targets_subset,
                'analytical_torque': analytical_np,
                'residual_component': residual_np,
                'combined_prediction': analytical_np + residual_np,
                'position': positions_subset
            })
            
            # Add spring and PD components if available
            if spring_np is not None and pd_np is not None:
                residual_df_all['spring_component'] = spring_np
                residual_df_all['pd_component'] = pd_np
            
            # Sort and create deduplicated version for time series plots
            residual_df_all_sorted = residual_df_all.sort_values(by='timestamp').reset_index(drop=True)
            residual_df_unique = residual_df_all_sorted.loc[residual_df_all_sorted.groupby('timestamp')['position'].idxmax()].reset_index(drop=True)
            
            print(f"[Callback] Residual mode - Total: {len(residual_df_all_sorted)}, Unique timesteps: {len(residual_df_unique)}")
            
            # --- 5. Analytical Torque vs. Actual Torque Scatter Plot - Use ALL data ---
            # Sample data for scatter plot if dataset is very large
            if scatter_sample_ratio < 1.0:
                anal_scatter_df = residual_df_all_sorted.sample(frac=scatter_sample_ratio, random_state=42).reset_index(drop=True)
            else:
                anal_scatter_df = residual_df_all_sorted
                
            fig_anal_scatter, ax_anal_scatter = plt.subplots(figsize=scatter_figsize, dpi=dpi)
            ax_anal_scatter.scatter(anal_scatter_df['actual_torque'], anal_scatter_df['analytical_torque'], 
                                  alpha=0.5, s=1 if is_large_dataset else 20, label=f'Analytical (n={len(anal_scatter_df):,})', color='green')
            lims_anal = [
                np.min([ax_anal_scatter.get_xlim(), ax_anal_scatter.get_ylim()]),
                np.max([ax_anal_scatter.get_xlim(), ax_anal_scatter.get_ylim()]),
            ]
            ax_anal_scatter.plot(lims_anal, lims_anal, 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
            ax_anal_scatter.set_xlabel("Actual Torque", fontsize=14)
            ax_anal_scatter.set_ylabel("Analytical Torque", fontsize=14)
            ax_anal_scatter.set_title("Test Set: Analytical Torque vs. Actual Torque", fontsize=16)
            ax_anal_scatter.legend(fontsize=12, loc='upper left')
            ax_anal_scatter.grid(True)
            ax_anal_scatter.set_aspect('equal', adjustable='box')
            
            # --- 6. Component Analysis Time Series - Use deduplicated for clarity ---
            fig_components, ax_components = plt.subplots(figsize=component_figsize, dpi=dpi)
            comp_linewidth = linewidth * 1.2 if not is_large_dataset else linewidth
            comp_thin_linewidth = linewidth * 0.8 if not is_large_dataset else linewidth * 0.7
            
            ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['actual_torque'], 
                             label='Actual Torque', linewidth=comp_linewidth, color='blue')
            ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['analytical_torque'], 
                             label='Analytical Total', linewidth=linewidth, color='green', alpha=0.9)
            
            # Plot individual spring and PD components if available
            if 'spring_component' in residual_df_unique.columns and 'pd_component' in residual_df_unique.columns:
                ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['spring_component'], 
                                 label='Spring Component', linewidth=comp_thin_linewidth, color='darkgreen', alpha=0.8, linestyle=':')
                ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['pd_component'], 
                                 label='PD Component', linewidth=comp_thin_linewidth, color='darkblue', alpha=0.8, linestyle=':')
            
            ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['residual_component'], 
                             label='Residual Component', linewidth=linewidth, color='red', alpha=0.9)
            ax_components.plot(residual_df_unique['timestamp'], residual_df_unique['combined_prediction'], 
                             label='Combined Prediction', linewidth=linewidth, color='orange', alpha=0.9, linestyle='--')
            ax_components.set_xlabel("Time (seconds since epoch)", fontsize=14)
            ax_components.set_ylabel("Torque", fontsize=14)
            ax_components.set_title("Test Set: Torque Component Analysis (Residual Mode)", fontsize=16)
            ncol = 1 if is_large_dataset else 2  # Single column legend for large datasets
            ax_components.legend(fontsize=12, ncol=ncol, loc='upper right')
            ax_components.grid(True)
            
            # --- 7. Short Component Analysis (first 5 seconds) - Use deduplicated ---
            short_mask_resid = (residual_df_unique['timestamp'] >= start_ts) & (residual_df_unique['timestamp'] <= end_ts)
            short_resid_df = residual_df_unique[short_mask_resid]
            
            fig_components_short = None
            if len(short_resid_df) > 0:
                fig_components_short, ax_components_short = plt.subplots(figsize=component_figsize, dpi=dpi)
                ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['actual_torque'], 
                                        label='Actual Torque', linewidth=comp_linewidth, color='blue')
                ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['analytical_torque'], 
                                        label='Analytical Total', linewidth=linewidth, color='green', alpha=0.9)
                
                # Plot individual spring and PD components if available
                if 'spring_component' in short_resid_df.columns and 'pd_component' in short_resid_df.columns:
                    ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['spring_component'], 
                                           label='Spring Component', linewidth=comp_thin_linewidth, color='darkgreen', alpha=0.8, linestyle=':')
                    ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['pd_component'], 
                                           label='PD Component', linewidth=comp_thin_linewidth, color='darkblue', alpha=0.8, linestyle=':')
                
                ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['residual_component'], 
                                        label='Residual Component', linewidth=linewidth, color='red', alpha=0.9)
                ax_components_short.plot(short_resid_df['timestamp'], short_resid_df['combined_prediction'], 
                                        label='Combined Prediction', linewidth=linewidth, color='orange', alpha=0.9, linestyle='--')
                ax_components_short.set_xlabel("Time (seconds since epoch)", fontsize=14)
                ax_components_short.set_ylabel("Torque", fontsize=14)
                ax_components_short.set_title(f"Test Set: First {short_window_duration:.0f}s - Torque Component Analysis", fontsize=16)
                ax_components_short.legend(fontsize=12, ncol=ncol, loc='upper right')
                ax_components_short.grid(True)
            
            # --- 8. Contribution Analysis - Use ALL data for better statistical coverage ---
            analytical_error = residual_df_all_sorted['actual_torque'] - residual_df_all_sorted['analytical_torque']
            residual_error = residual_df_all_sorted['actual_torque'] - residual_df_all_sorted['combined_prediction']
            
            contrib_figsize = (14, 10) if is_large_dataset else (16, 12)
            fig_contrib, (ax_contrib1, ax_contrib2) = plt.subplots(2, 1, figsize=contrib_figsize, dpi=dpi)
            
            # Error comparison
            ax_contrib1.hist(analytical_error, bins=bins, alpha=0.7, label='Analytical Error', color='green')
            ax_contrib1.hist(residual_error, bins=bins, alpha=0.7, label='Combined Error', color='orange')
            ax_contrib1.set_xlabel("Error (Actual - Predicted)", fontsize=14)
            ax_contrib1.set_ylabel("Count", fontsize=14)
            ax_contrib1.set_title("Error Distribution: Analytical vs Combined", fontsize=16)
            ax_contrib1.legend(fontsize=12, loc='upper right')
            ax_contrib1.grid(True)
            
            # RMS contribution analysis - using all data for better statistics
            analytical_rms = np.sqrt(np.mean(analytical_error**2))
            combined_rms = np.sqrt(np.mean(residual_error**2))
            residual_magnitude_rms = np.sqrt(np.mean(residual_df_all_sorted['residual_component']**2))
            
            contributions = ['Analytical RMS Error', 'Combined RMS Error', 'Residual RMS Magnitude']
            values = [analytical_rms, combined_rms, residual_magnitude_rms]
            colors = ['green', 'orange', 'red']
            
            ax_contrib2.bar(contributions, values, color=colors, alpha=0.7)
            ax_contrib2.set_ylabel("RMS Value", fontsize=14)
            ax_contrib2.set_title("RMS Analysis: Analytical vs Residual Contributions", fontsize=16)
            ax_contrib2.grid(True, axis='y')
            ax_contrib2.tick_params(axis='both', which='major', labelsize=12)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax_contrib2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=12)
            
            plt.tight_layout()
            
            # --- 9. Individual Component Scatter Plots if available ---
            spring_pd_scatter_figs = []
            if 'spring_component' in residual_df_all_sorted.columns and 'pd_component' in residual_df_all_sorted.columns:
                # Use same sampled data for component scatter plots
                comp_scatter_df = anal_scatter_df if scatter_sample_ratio < 1.0 else residual_df_all_sorted
                
                # Spring component scatter plot
                fig_spring_scatter, ax_spring_scatter = plt.subplots(figsize=scatter_figsize, dpi=dpi)
                ax_spring_scatter.scatter(comp_scatter_df['actual_torque'], comp_scatter_df['spring_component'], 
                                        alpha=0.5, s=1 if is_large_dataset else 20, label=f'Spring (n={len(comp_scatter_df):,})', color='darkgreen')
                lims_spring = [
                    np.min([ax_spring_scatter.get_xlim(), ax_spring_scatter.get_ylim()]),
                    np.max([ax_spring_scatter.get_xlim(), ax_spring_scatter.get_ylim()]),
                ]
                ax_spring_scatter.plot(lims_spring, lims_spring, 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
                ax_spring_scatter.set_xlabel("Actual Torque", fontsize=14)
                ax_spring_scatter.set_ylabel("Spring Component", fontsize=14)
                ax_spring_scatter.set_title("Test Set: Spring Component vs. Actual Torque", fontsize=16)
                ax_spring_scatter.legend(fontsize=12, loc='upper left')
                ax_spring_scatter.grid(True)
                ax_spring_scatter.set_aspect('equal', adjustable='box')
                spring_pd_scatter_figs.append(("Spring Component vs Actual Scatter", fig_spring_scatter))
                
                # PD component scatter plot
                fig_pd_scatter, ax_pd_scatter = plt.subplots(figsize=scatter_figsize, dpi=dpi)
                ax_pd_scatter.scatter(comp_scatter_df['actual_torque'], comp_scatter_df['pd_component'], 
                                    alpha=0.5, s=1 if is_large_dataset else 20, label=f'PD (n={len(comp_scatter_df):,})', color='darkblue')
                lims_pd = [
                    np.min([ax_pd_scatter.get_xlim(), ax_pd_scatter.get_ylim()]),
                    np.max([ax_pd_scatter.get_xlim(), ax_pd_scatter.get_ylim()]),
                ]
                ax_pd_scatter.plot(lims_pd, lims_pd, 'r--', alpha=0.75, zorder=0, label='Ideal (y=x)')
                ax_pd_scatter.set_xlabel("Actual Torque", fontsize=14)
                ax_pd_scatter.set_ylabel("PD Component", fontsize=14)
                ax_pd_scatter.set_title("Test Set: PD Component vs. Actual Torque", fontsize=16)
                ax_pd_scatter.legend(fontsize=12, loc='upper left')
                ax_pd_scatter.grid(True)
                ax_pd_scatter.set_aspect('equal', adjustable='box')
                spring_pd_scatter_figs.append(("PD Component vs Actual Scatter", fig_pd_scatter))

            # Add residual mode plots to log dictionary
            log_dict.update({
                "Analytical vs Actual Scatter": wandb.Image(fig_anal_scatter),
                "Component Analysis Time Series": wandb.Image(fig_components),
                "Contribution Analysis": wandb.Image(fig_contrib)
            })
            
            # Add individual component scatter plots if available
            for plot_name, fig in spring_pd_scatter_figs:
                log_dict[plot_name] = wandb.Image(fig)
            
            if fig_components_short:
                log_dict[f"Component Analysis First {short_window_duration:.0f}s"] = wandb.Image(fig_components_short)
            
            # Close residual mode figures
            plt.close(fig_anal_scatter)
            plt.close(fig_components)
            plt.close(fig_contrib)
            if fig_components_short:
                plt.close(fig_components_short)
            
            # Close individual component scatter plots
            for _, fig in spring_pd_scatter_figs:
                plt.close(fig)
                
            print(f"[Callback] Residual mode analysis complete. Added {len([k for k in log_dict.keys() if 'Analytical' in k or 'Component' in k or 'Contribution' in k or 'Spring' in k or 'PD' in k])} additional plots.")

        # Log plots to WandB
        if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
            print("[Callback] Logging plots to WandB...")
            try:
                trainer.logger.experiment.log(log_dict)
                print(f"[Callback] {len(log_dict)} plots logged successfully.")
            except Exception as e:
                print(f"[Callback] Error logging plots to WandB: {e}")
        else:
            print("[Callback] WandB logger not available. Cannot log plots.")

        plt.close(fig_scatter)
        plt.close(fig_ts)
        plt.close(fig_resid)
        plt.close(fig_hist)
        
        # Close slice plots
        for slice_fig in slice_figs:
            plt.close(slice_fig)
        # Close short window plots
        for short_fig in short_figs:
            plt.close(short_fig)

        # Clean up stored tensors in module
        if hasattr(pl_module, 'all_test_preds'):
             delattr(pl_module, 'all_test_preds')
        if hasattr(pl_module, 'all_test_targets'):
             delattr(pl_module, 'all_test_targets')
        if hasattr(pl_module, 'all_test_timestamps'):
             delattr(pl_module, 'all_test_timestamps')
        if hasattr(pl_module, 'all_test_positions'):
             delattr(pl_module, 'all_test_positions')
        # Clean up residual mode attributes
        if hasattr(pl_module, 'all_test_analytical_torque'):
             delattr(pl_module, 'all_test_analytical_torque')
        if hasattr(pl_module, 'all_test_residual_component'):
             delattr(pl_module, 'all_test_residual_component')
        if hasattr(pl_module, 'all_test_spring_component'):
             delattr(pl_module, 'all_test_spring_component')
        if hasattr(pl_module, 'all_test_pd_component'):
             delattr(pl_module, 'all_test_pd_component')


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

        # --- Detailed Torque Raw vs Filtered Visualizations (Time Series, Histogram, PSD) ---
        raw_col = 'torque_raw'
        filt_col = 'torque_filtered'
        all_raw = []
        all_filt = []
        for group_id, datasets_in_group in datamodule.datasets_by_group.items():
            for ds in datasets_in_group:
                df = getattr(ds, 'data_df', None)
                if isinstance(df, pd.DataFrame):
                    if raw_col in df.columns:
                        all_raw.append(df[raw_col])
                    if filt_col in df.columns:
                        all_filt.append(df[filt_col])
        if all_raw and all_filt:
            raw_concat = pd.concat(all_raw, ignore_index=True)
            filt_concat = pd.concat(all_filt, ignore_index=True)
            # Time Series
            fig_tf, ax_tf = plt.subplots(figsize=(15, 5))
            ax_tf.plot(raw_concat.index, raw_concat.values, label='Raw Torque', alpha=0.7)
            ax_tf.plot(filt_concat.index, filt_concat.values, label='Filtered Torque', alpha=0.7)
            ax_tf.set_xlabel("Sample Index")
            ax_tf.set_ylabel("Torque")
            ax_tf.set_title("Full Dataset: Torque Before and After Filtering")
            ax_tf.legend()
            ax_tf.grid(True)
            if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                trainer.logger.experiment.log({"Dataset/Torque Raw vs Filtered Time Series": wandb.Image(fig_tf)})
            plt.close(fig_tf)
            # Histogram
            fig_h, ax_h = plt.subplots(figsize=(8, 5))
            ax_h.hist(raw_concat.values, bins=50, alpha=0.5, label='Raw Torque')
            ax_h.hist(filt_concat.values, bins=50, alpha=0.5, label='Filtered Torque')
            ax_h.set_xlabel("Torque")
            ax_h.set_ylabel("Count")
            ax_h.set_title("Histogram of Torque Before and After Filtering")
            ax_h.legend()
            if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                trainer.logger.experiment.log({"Dataset/Torque Raw vs Filtered Histogram": wandb.Image(fig_h)})
            plt.close(fig_h)
            # PSD (Welch)
            try:
                from scipy.signal import welch
                fs = getattr(datamodule, 'sampling_frequency', None) or datamodule.get_sampling_frequency()
                if fs:
                    freqs_r, psd_r = welch(raw_concat.values, fs=fs)
                    freqs_f, psd_f = welch(filt_concat.values, fs=fs)
                    fig_psd, ax_psd = plt.subplots(figsize=(8, 5))
                    ax_psd.semilogy(freqs_r, psd_r, label='Raw Torque')
                    ax_psd.semilogy(freqs_f, psd_f, label='Filtered Torque')
                    ax_psd.set_xlabel("Frequency (Hz)")
                    ax_psd.set_ylabel("Power Spectral Density")
                    ax_psd.set_title("Torque PSD (Welch) Before and After Filtering")
                    ax_psd.legend()
                    if trainer.logger and hasattr(trainer.logger.experiment, 'log'):
                        trainer.logger.experiment.log({"Dataset/Torque PSD (Welch)": wandb.Image(fig_psd)})
                    plt.close(fig_psd)
                else:
                    print("[Callback] Sampling frequency not available. Skipping PSD plot for torque.")
            except ImportError:
                print("[Callback] scipy not available. Skipping PSD plot for torque.")
        else:
            print("[Callback] Raw or filtered torque data not found. Skipping detailed torque visualizations.")

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

        # Adaptive sizing for debug plot
        total_debug_points = len(debug_plot_df_sorted)
        is_large_debug = total_debug_points > 1_000_000
        
        if is_large_debug:
            debug_figsize = (16, 6)
            debug_dpi = 100
            debug_linewidth = 0.8
        else:
            debug_figsize = (20, 8)
            debug_dpi = 150
            debug_linewidth = 1.2
        
        # --- Time Series Plot ---
        fig_debug_ts, ax_debug_ts = plt.subplots(figsize=debug_figsize, dpi=debug_dpi)
        ax_debug_ts.plot(debug_plot_df_sorted['timestamp'], debug_plot_df_sorted['actual_torque'], label='Actual Torque (Debug)', linewidth=debug_linewidth, linestyle='--')
        ax_debug_ts.plot(debug_plot_df_sorted['timestamp'], debug_plot_df_sorted['predicted_torque'], label='Predicted Torque (Debug)', alpha=0.7, linewidth=debug_linewidth, linestyle=':')
        ax_debug_ts.set_xlabel("Time (seconds since epoch)", fontsize=14)
        ax_debug_ts.set_ylabel("Torque", fontsize=14)
        ax_debug_ts.set_title("DEBUG Test Set: Torque Prediction Over Time (Sorted by Timestamp - New Plotter)", fontsize=16)
        ax_debug_ts.legend(fontsize=12, loc='upper right')
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

