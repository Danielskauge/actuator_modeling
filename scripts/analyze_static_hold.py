#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_static_hold(csv_file_path: str, output_dir: str = None, apply_bias_correction: bool = True):
    """
    Analyze static hold data from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Directory to save plots (default: same directory as script)
        apply_bias_correction: Whether to apply sensor bias correction
    """
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        print(f"Error: File '{csv_file_path}' not found.")
        return
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "static_hold_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from: {csv_path}")
    print(f"Saving plots to: {output_dir}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Apply sensor bias corrections (estimated from this same static data)
    sensor_biases = {
        'acc_y_bias': 0.039204,
        'gyro_z_bias': -0.002273
    }
    
    if apply_bias_correction:
        print("\n--- Applying Sensor Bias Corrections ---")
        if 'Acc_Y' in df.columns:
            original_mean = df['Acc_Y'].mean()
            df['Acc_Y'] = df['Acc_Y'] - sensor_biases['acc_y_bias']
            corrected_mean = df['Acc_Y'].mean()
            print(f"Acc_Y: {original_mean:.6f} -> {corrected_mean:.6f} (bias: {sensor_biases['acc_y_bias']:.6f})")
        
        if 'Gyro_Z' in df.columns:
            original_mean = df['Gyro_Z'].mean()
            df['Gyro_Z'] = df['Gyro_Z'] - sensor_biases['gyro_z_bias']
            corrected_mean = df['Gyro_Z'].mean()
            print(f"Gyro_Z: {original_mean:.6f} -> {corrected_mean:.6f} (bias: {sensor_biases['gyro_z_bias']:.6f})")
    
    # Convert time to seconds
    df['Time_s'] = df['Time_ms'] / 1000.0
    time_duration = df['Time_s'].max() - df['Time_s'].min()
    print(f"Time duration: {time_duration:.2f} seconds")
    
    # Calculate sampling frequency
    median_dt = df['Time_s'].diff().median()
    fs = 1.0 / median_dt if median_dt > 0 else None
    print(f"Sampling frequency: {fs:.2f} Hz" if fs else "Could not determine sampling frequency")
    
    # Basic statistics for position hold
    print("\n--- Position Hold Analysis ---")
    commanded_mean = df['Commanded_Angle'].mean()
    commanded_std = df['Commanded_Angle'].std()
    encoder_mean = df['Encoder_Angle'].mean()
    encoder_std = df['Encoder_Angle'].std()
    position_error = df['Encoder_Angle'] - df['Commanded_Angle']
    error_mean = position_error.mean()
    error_std = position_error.std()
    error_rms = np.sqrt(np.mean(position_error**2))
    
    print(f"Commanded angle: {commanded_mean:.3f} ± {commanded_std:.3f} degrees")
    print(f"Encoder angle: {encoder_mean:.3f} ± {encoder_std:.3f} degrees")
    print(f"Position error: {error_mean:.3f} ± {error_std:.3f} degrees (RMS: {error_rms:.3f})")
    
    # IMU statistics
    print("\n--- IMU Analysis ---")
    for axis in ['X', 'Y', 'Z']:
        acc_col = f'Acc_{axis}'
        gyro_col = f'Gyro_{axis}'
        if acc_col in df.columns:
            acc_rms = np.sqrt(np.mean(df[acc_col]**2))
            print(f"Acc_{axis} RMS: {acc_rms:.4f}")
        if gyro_col in df.columns:
            gyro_rms = np.sqrt(np.mean(df[gyro_col]**2))
            print(f"Gyro_{axis} RMS: {gyro_rms:.6f}")
    
    # Generate plots
    generate_timeseries_plots(df, output_dir)
    if fs and fs > 0:
        generate_psd_plots(df, fs, output_dir)
    
    print(f"\nAnalysis complete. Plots saved to: {output_dir}")

def generate_timeseries_plots(df, output_dir):
    """Generate time series plots for static hold analysis."""
    
    # 1. Position tracking plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_s'], df['Commanded_Angle'], label='Commanded', linewidth=1.5, alpha=0.8)
    plt.plot(df['Time_s'], df['Encoder_Angle'], label='Encoder', linewidth=1.0, alpha=0.9)
    plt.title('Position Tracking During Static Hold')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "position_tracking.png", dpi=150)
    plt.close()
    
    # 2. Position error plot
    position_error = df['Encoder_Angle'] - df['Commanded_Angle']
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_s'], position_error, linewidth=0.8, color='red')
    plt.title('Position Error During Static Hold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (degrees)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    error_rms = np.sqrt(np.mean(position_error**2))
    error_std = position_error.std()
    plt.text(0.02, 0.98, f'RMS Error: {error_rms:.4f}°\nStd Dev: {error_std:.4f}°', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "position_error.png", dpi=150)
    plt.close()
    
    # 3. Accelerometer data
    accel_cols = [col for col in ['Acc_X', 'Acc_Y', 'Acc_Z'] if col in df.columns]
    if accel_cols:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(accel_cols):
            plt.subplot(len(accel_cols), 1, i+1)
            plt.plot(df['Time_s'], df[col], linewidth=0.6)
            plt.title(f'{col} During Static Hold')
            plt.ylabel('Acceleration')
            plt.grid(True, alpha=0.3)
            if i == len(accel_cols) - 1:
                plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(output_dir / "accelerometer_timeseries.png", dpi=150)
        plt.close()
    
    # 4. Gyroscope data
    gyro_cols = [col for col in ['Gyro_X', 'Gyro_Y', 'Gyro_Z'] if col in df.columns]
    if gyro_cols:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(gyro_cols):
            plt.subplot(len(gyro_cols), 1, i+1)
            plt.plot(df['Time_s'], df[col], linewidth=0.6)
            plt.title(f'{col} During Static Hold')
            plt.ylabel('Angular Velocity')
            plt.grid(True, alpha=0.3)
            if i == len(gyro_cols) - 1:
                plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(output_dir / "gyroscope_timeseries.png", dpi=150)
        plt.close()

def generate_psd_plots(df, fs, output_dir):
    """Generate PSD plots for frequency analysis of static hold data."""
    
    # Columns to analyze
    analysis_cols = []
    position_error = df['Encoder_Angle'] - df['Commanded_Angle']
    df_analysis = df.copy()
    df_analysis['Position_Error'] = position_error
    
    # Add relevant columns
    for col in ['Position_Error', 'Encoder_Angle', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']:
        if col in df_analysis.columns and not df_analysis[col].isnull().all():
            analysis_cols.append(col)
    
    if not analysis_cols:
        print("No valid columns for PSD analysis")
        return
    
    # Create PSD plots
    num_plots = len(analysis_cols)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Limit data for PSD computation
    max_points = 8192
    if len(df_analysis) > max_points:
        step = len(df_analysis) // max_points
        df_psd = df_analysis.iloc[::step].copy()
    else:
        df_psd = df_analysis.copy()
    
    nfft = min(512, len(df_psd) // 4)
    
    for i, col in enumerate(analysis_cols):
        try:
            data = df_psd[col].dropna()
            if len(data) > 10:  # Minimum data points needed
                axes[i].psd(data, NFFT=nfft, Fs=fs)
                axes[i].set_title(f'PSD - {col}')
                axes[i].set_xlabel('Frequency (Hz)')
                axes[i].set_ylabel('Power/Frequency')
        except Exception as e:
            print(f"Error generating PSD for {col}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {col}', ha='center', va='center', transform=axes[i].transAxes)
    
    # Hide unused subplots
    for j in range(len(analysis_cols), len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle('Power Spectral Density - Static Hold Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "psd_analysis.png", dpi=150)
    plt.close()

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze static hold data from CSV file")
    parser.add_argument("csv_file", nargs='?', default="data/static/static_segment_plateau2.csv",
                        help="Path to CSV file (default: data/static/static_segment_plateau2.csv)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots (default: static_hold_analysis/)")
    parser.add_argument("--no_bias_correction", action="store_true",
                        help="Skip sensor bias correction")
    
    args = parser.parse_args()
    
    # Handle relative paths from script location
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / args.csv_file
    
    analyze_static_hold(str(csv_path), args.output_dir, apply_bias_correction=not args.no_bias_correction)

if __name__ == "__main__":
    main() 