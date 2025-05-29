#!/usr/bin/env python
"""
Analyze the effects of the Butterworth filter on accelerometer data.

This script loads the same CSV files both with and without filtering,
then compares the power spectral density (PSD) and timeseries of the
accelerometer data to visualize the filter's impact.

Usage:
    python scripts/analyze_filter_effects.py [config options]
    python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=30.0
    python scripts/analyze_filter_effects.py --config-path=configs --config-name=config data=default
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from typing import List, Tuple, Optional
import seaborn as sns

from src.data.datasets import ActuatorDataset


def load_dataset_pair(
    csv_file: str, 
    inertia: float, 
    radius_accel: float,
    gyro_axis: str,
    accel_axis: str,
    filter_cutoff_freq_hz: Optional[float],
    filter_order: int
) -> Tuple[ActuatorDataset, ActuatorDataset]:
    """
    Load the same CSV file twice: once with filtering and once without.
    
    Returns:
        Tuple of (unfiltered_dataset, filtered_dataset)
    """
    # Unfiltered dataset
    dataset_unfiltered = ActuatorDataset(
        csv_file_path=csv_file,
        inertia=inertia,
        radius_accel=radius_accel,
        gyro_axis_for_ang_vel=gyro_axis,
        accel_axis_for_torque=accel_axis,
        filter_cutoff_freq_hz=None,  # No filter
        filter_order=filter_order
    )
    
    # Filtered dataset (only if filter_cutoff_freq_hz is provided)
    if filter_cutoff_freq_hz is not None:
        dataset_filtered = ActuatorDataset(
            csv_file_path=csv_file,
            inertia=inertia,
            radius_accel=radius_accel,
            gyro_axis_for_ang_vel=gyro_axis,
            accel_axis_for_torque=accel_axis,
            filter_cutoff_freq_hz=filter_cutoff_freq_hz,
            filter_order=filter_order
        )
    else:
        # If no filter is configured, just return the same dataset twice
        dataset_filtered = dataset_unfiltered
        
    return dataset_unfiltered, dataset_filtered


def analyze_accelerometer_signal(
    dataset_unfiltered: ActuatorDataset,
    dataset_filtered: ActuatorDataset,
    accel_axis: str,
    output_dir: Path,
    file_label: str
) -> None:
    """
    Analyze and plot the effects of filtering on the accelerometer signal.
    
    Args:
        dataset_unfiltered: Dataset without filtering
        dataset_filtered: Dataset with filtering applied
        accel_axis: The accelerometer axis to analyze
        output_dir: Directory to save plots
        file_label: Label for the file being analyzed
    """
    # Get the raw accelerometer data from both datasets
    unfiltered_df = dataset_unfiltered.data_df.reset_index()
    filtered_df = dataset_filtered.data_df.reset_index()
    
    # Extract time and acceleration data
    time_s = pd.to_numeric(unfiltered_df.index) if 'time_s' not in unfiltered_df.columns else unfiltered_df['time_s']
    
    # Get raw accelerometer data (need to load CSV again to get unprocessed accel data)
    csv_path = dataset_unfiltered.csv_file_path
    raw_df = pd.read_csv(csv_path)
    raw_df['time_s'] = raw_df['Time_ms'] / 1000.0
    
    accel_raw = raw_df[accel_axis].values
    time_raw = raw_df['time_s'].values
    
    # Calculate angular acceleration from tangential acceleration for both cases
    radius = dataset_unfiltered.radius_accel
    angular_accel_unfiltered = accel_raw / radius
    
    # For filtered data, we need to apply the same filter that was used in the dataset
    if dataset_filtered.filter_cutoff_freq_hz is not None:
        fs = dataset_unfiltered.get_sampling_frequency()
        nyquist_freq = 0.5 * fs
        if dataset_filtered.filter_cutoff_freq_hz < nyquist_freq:
            Wn = dataset_filtered.filter_cutoff_freq_hz / nyquist_freq
            b, a = signal.butter(dataset_filtered.filter_order, Wn, btype='low', analog=False)
            if len(angular_accel_unfiltered) > dataset_filtered.filter_order * 3:
                angular_accel_filtered = signal.filtfilt(b, a, angular_accel_unfiltered)
            else:
                angular_accel_filtered = angular_accel_unfiltered
        else:
            angular_accel_filtered = angular_accel_unfiltered
    else:
        angular_accel_filtered = angular_accel_unfiltered
    
    # Calculate derived torque for both cases
    inertia = dataset_unfiltered.inertia
    torque_unfiltered = inertia * angular_accel_unfiltered
    torque_filtered = inertia * angular_accel_filtered
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Filter Effects Analysis: {file_label}\n'
                f'Cutoff: {dataset_filtered.filter_cutoff_freq_hz} Hz, '
                f'Order: {dataset_filtered.filter_order}', fontsize=14)
    
    # Sampling frequency for PSD
    fs = dataset_unfiltered.get_sampling_frequency()
    
    # Row 1: Accelerometer data
    # Timeseries comparison
    axes[0, 0].plot(time_raw, accel_raw, 'b-', alpha=0.7, label='Unfiltered', linewidth=1)
    if dataset_filtered.filter_cutoff_freq_hz is not None:
        # Apply filter to the raw accelerometer data for visualization
        if dataset_filtered.filter_cutoff_freq_hz < 0.5 * fs:
            Wn = dataset_filtered.filter_cutoff_freq_hz / (0.5 * fs)
            b, a = signal.butter(dataset_filtered.filter_order, Wn, btype='low')
            accel_filtered_viz = signal.filtfilt(b, a, accel_raw)
            axes[0, 0].plot(time_raw, accel_filtered_viz, 'r-', alpha=0.8, label='Filtered', linewidth=1.5)
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel(f'{accel_axis} (m/s²)')
    axes[0, 0].set_title('Accelerometer Data Timeseries')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSD comparison for accelerometer
    if len(accel_raw) > 256:
        f_unfiltered, psd_unfiltered = signal.welch(accel_raw, fs=fs, nperseg=min(256, len(accel_raw)//4))
        axes[0, 1].semilogy(f_unfiltered, psd_unfiltered, 'b-', alpha=0.7, label='Unfiltered', linewidth=1)
        
        if dataset_filtered.filter_cutoff_freq_hz is not None:
            f_filtered, psd_filtered = signal.welch(accel_filtered_viz, fs=fs, nperseg=min(256, len(accel_raw)//4))
            axes[0, 1].semilogy(f_filtered, psd_filtered, 'r-', alpha=0.8, label='Filtered', linewidth=1.5)
            # Add vertical line at cutoff frequency
            axes[0, 1].axvline(dataset_filtered.filter_cutoff_freq_hz, color='red', linestyle='--', alpha=0.7, label=f'Cutoff: {dataset_filtered.filter_cutoff_freq_hz} Hz')
    
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD (m²/s⁴/Hz)')
    axes[0, 1].set_title(f'{accel_axis} Power Spectral Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference plot
    if dataset_filtered.filter_cutoff_freq_hz is not None:
        accel_diff = accel_raw - accel_filtered_viz
        axes[0, 2].plot(time_raw, accel_diff, 'g-', alpha=0.7, linewidth=1)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel(f'Difference (m/s²)')
        axes[0, 2].set_title('Filtered - Unfiltered Difference')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add RMS difference as text
        rms_diff = np.sqrt(np.mean(accel_diff**2))
        axes[0, 2].text(0.05, 0.95, f'RMS Diff: {rms_diff:.3f} m/s²', 
                       transform=axes[0, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Row 2: Derived torque data
    # Torque timeseries comparison
    axes[1, 0].plot(time_raw, torque_unfiltered, 'b-', alpha=0.7, label='Unfiltered', linewidth=1)
    if dataset_filtered.filter_cutoff_freq_hz is not None:
        axes[1, 0].plot(time_raw, torque_filtered, 'r-', alpha=0.8, label='Filtered', linewidth=1.5)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].set_title('Derived Torque Timeseries')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # PSD comparison for torque
    if len(torque_unfiltered) > 256:
        f_torque_unfiltered, psd_torque_unfiltered = signal.welch(torque_unfiltered, fs=fs, nperseg=min(256, len(torque_unfiltered)//4))
        axes[1, 1].semilogy(f_torque_unfiltered, psd_torque_unfiltered, 'b-', alpha=0.7, label='Unfiltered', linewidth=1)
        
        if dataset_filtered.filter_cutoff_freq_hz is not None:
            f_torque_filtered, psd_torque_filtered = signal.welch(torque_filtered, fs=fs, nperseg=min(256, len(torque_filtered)//4))
            axes[1, 1].semilogy(f_torque_filtered, psd_torque_filtered, 'r-', alpha=0.8, label='Filtered', linewidth=1.5)
            axes[1, 1].axvline(dataset_filtered.filter_cutoff_freq_hz, color='red', linestyle='--', alpha=0.7, label=f'Cutoff: {dataset_filtered.filter_cutoff_freq_hz} Hz')
    
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD (Nm²/Hz)')
    axes[1, 1].set_title('Derived Torque Power Spectral Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Torque difference plot
    if dataset_filtered.filter_cutoff_freq_hz is not None:
        torque_diff = torque_unfiltered - torque_filtered
        axes[1, 2].plot(time_raw, torque_diff, 'g-', alpha=0.7, linewidth=1)
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Torque Difference (Nm)')
        axes[1, 2].set_title('Torque: Unfiltered - Filtered')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add RMS difference as text
        rms_torque_diff = np.sqrt(np.mean(torque_diff**2))
        axes[1, 2].text(0.05, 0.95, f'RMS Diff: {rms_torque_diff:.4f} Nm', 
                       transform=axes[1, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{file_label}_filter_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved filter analysis plot: {output_file}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to analyze filter effects using Hydra configuration."""
    
    print("=== Butterworth Filter Effects Analysis ===")
    print(f"Configuration:")
    print(f"  Data directory: {cfg.data.data_base_dir}")
    print(f"  Filter cutoff frequency: {cfg.data.filter_cutoff_freq_hz} Hz")
    print(f"  Filter order: {cfg.data.filter_order}")
    print(f"  Accelerometer axis: {cfg.data.accel_axis_for_torque}")
    
    # Create output directory
    output_dir = Path("filter_analysis_plots")
    output_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir.resolve()}")
    
    # Clear old plots
    for old_plot in output_dir.glob("*.png"):
        old_plot.unlink()
    
    # Check if filtering is enabled
    if cfg.data.filter_cutoff_freq_hz is None:
        print("\nWarning: No filter cutoff frequency specified. This analysis will show identical data.")
        print("To enable filtering, set data.filter_cutoff_freq_hz to a positive value.")
    
    # Process each inertia group
    total_files_processed = 0
    
    for group_config in cfg.data.inertia_groups:
        group_id = group_config.id
        group_folder = group_config.folder
        group_inertia = group_config.inertia
        
        print(f"\nProcessing inertia group: {group_id}")
        print(f"  Folder: {group_folder}")
        print(f"  Inertia: {group_inertia} kg⋅m²")
        
        # Build path to group's subfolder
        group_path = Path(cfg.data.data_base_dir) / group_folder
        
        if not group_path.exists():
            print(f"  Warning: Group folder does not exist: {group_path}")
            continue
        
        # Find CSV files in the group
        csv_files = list(group_path.glob("*.csv"))
        if not csv_files:
            print(f"  Warning: No CSV files found in {group_path}")
            continue
        
        # Sort for consistent processing
        csv_files.sort()
        
        # Process each CSV file (limit to first few for manageable output)
        max_files_per_group = 3
        for i, csv_file in enumerate(csv_files[:max_files_per_group]):
            file_label = f"{group_id}_{csv_file.stem}"
            print(f"  Processing file: {csv_file.name}")
            
            try:
                # Load datasets with and without filtering
                dataset_unfiltered, dataset_filtered = load_dataset_pair(
                    csv_file=str(csv_file),
                    inertia=group_inertia,
                    radius_accel=cfg.data.radius_accel,
                    gyro_axis=cfg.data.gyro_axis_for_ang_vel,
                    accel_axis=cfg.data.accel_axis_for_torque,
                    filter_cutoff_freq_hz=cfg.data.filter_cutoff_freq_hz,
                    filter_order=cfg.data.filter_order
                )
                
                # Perform analysis
                analyze_accelerometer_signal(
                    dataset_unfiltered=dataset_unfiltered,
                    dataset_filtered=dataset_filtered,
                    accel_axis=cfg.data.accel_axis_for_torque,
                    output_dir=output_dir,
                    file_label=file_label
                )
                
                total_files_processed += 1
                
            except Exception as e:
                print(f"    Error processing {csv_file.name}: {e}")
                continue
    
    # Create summary information
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Butterworth Filter Effects Analysis Summary ===\n")
        f.write(f"Configuration used:\n")
        f.write(f"  Data directory: {cfg.data.data_base_dir}\n")
        f.write(f"  Filter cutoff frequency: {cfg.data.filter_cutoff_freq_hz} Hz\n")
        f.write(f"  Filter order: {cfg.data.filter_order}\n")
        f.write(f"  Accelerometer axis: {cfg.data.accel_axis_for_torque}\n")
        f.write(f"  Radius: {cfg.data.radius_accel} m\n")
        f.write(f"\nFiles processed: {total_files_processed}\n")
        f.write(f"Generated {len(list(output_dir.glob('*_filter_analysis.png')))} analysis plots\n")
        
        if cfg.data.filter_cutoff_freq_hz is not None:
            f.write(f"\nFilter characteristics:\n")
            f.write(f"  Type: Low-pass Butterworth\n")
            f.write(f"  Cutoff frequency: {cfg.data.filter_cutoff_freq_hz} Hz\n")
            f.write(f"  Order: {cfg.data.filter_order}\n")
            f.write(f"  Implementation: scipy.signal.filtfilt (zero-phase)\n")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Processed {total_files_processed} files")
    print(f"Generated plots in: {output_dir.resolve()}")
    print(f"Summary saved to: {summary_file}")
    
    if cfg.data.filter_cutoff_freq_hz is not None:
        print(f"\nFilter settings:")
        print(f"  Cutoff frequency: {cfg.data.filter_cutoff_freq_hz} Hz")
        print(f"  Order: {cfg.data.filter_order}")
        print(f"  Type: Low-pass Butterworth (zero-phase)")


if __name__ == "__main__":
    main()