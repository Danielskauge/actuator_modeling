#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.signal import butter, filtfilt

# Key columns expected in the CSV files, based on README.md
KEY_COLUMNS = [
    'Time_ms',
    'Encoder_Angle',
    'Commanded_Angle',
    'Acc_X',
    'Acc_Y',
    'Acc_Z',
    'Gyro_X',
    'Gyro_Y',
    'Gyro_Z'
]

# Columns to plot as time series and histograms
# Exclude Time_ms from direct histogram/time series plotting as it's the index
PLOT_COLUMNS = [col for col in KEY_COLUMNS if col != 'Time_ms']

# Columns for specific plot types
ACCEL_COLS = ['Acc_X', 'Acc_Y', 'Acc_Z']
GYRO_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
PSD_TARGET_COLS = ACCEL_COLS + ['Commanded_Angle', 'Gyro_Z', 'Encoder_Angle']
INDIVIDUAL_TIMESERIES_COLS = ['Encoder_Angle', 'Commanded_Angle']

# Torque calculation parameters (from default config)
DEFAULT_INERTIA = 0.013  # kg*m^2
DEFAULT_RADIUS_ACCEL = 0.03  # meters
DEFAULT_FILTER_CUTOFF = 15.0  # Hz
DEFAULT_FILTER_ORDER = 4
DEFAULT_RESAMPLING_FREQ = 240.0  # Hz

def downsample_for_plotting(df, max_points=10000):
    """Downsample data for faster plotting while preserving overall shape."""
    if len(df) <= max_points:
        return df
    
    # Take every nth point to reduce to approximately max_points
    step = len(df) // max_points
    return df.iloc[::step].copy()

def calculate_torque_processing_stages(df, inertia=DEFAULT_INERTIA, radius_accel=DEFAULT_RADIUS_ACCEL, 
                                     resampling_freq=DEFAULT_RESAMPLING_FREQ, 
                                     filter_cutoff=DEFAULT_FILTER_CUTOFF, 
                                     filter_order=DEFAULT_FILTER_ORDER):
    """
    Calculate torque at different processing stages to show the effect of resampling and filtering.
    Returns a DataFrame with time and torque columns for each stage.
    """
    if 'Acc_Y' not in df.columns or 'Time_ms' not in df.columns:
        return None
    
    # Convert time to seconds and normalize to start from 0
    df_work = df.copy()
    df_work['time_s'] = df_work['Time_ms'] / 1000.0
    df_work['time_s'] = df_work['time_s'] - df_work['time_s'].min()
    df_work = df_work.set_index('time_s')
    df_work = df_work.sort_index()
    
    # Remove first 5 seconds like ActuatorDataset does
    df_work = df_work[df_work.index >= 5.0]
    
    if len(df_work) < 10:  # Not enough data
        return None
    
    # Stage 1: Original torque (raw accelerometer data)
    torque_original = inertia * df_work['Acc_Y'] / radius_accel
    
    # Calculate original sampling frequency
    times = df_work.index.values
    if len(times) > 1:
        original_sampling_frequency = 1.0 / np.median(np.diff(times))
    else:
        return None
    
    # Stage 2: Resampled torque
    if resampling_freq and resampling_freq != original_sampling_frequency:
        start_time = df_work.index.min()
        end_time = df_work.index.max()
        dt = 1.0 / resampling_freq
        new_times = np.arange(start_time, end_time + dt, dt)
        
        # Interpolate Acc_Y data
        f = interpolate.interp1d(df_work.index.values, df_work['Acc_Y'].values, 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        acc_y_resampled = f(new_times)
        torque_resampled = inertia * acc_y_resampled / radius_accel
        
        # Create resampled dataframe
        df_resampled = pd.DataFrame({
            'time_s': new_times,
            'torque_resampled': torque_resampled
        }).set_index('time_s')
        
        sampling_freq_for_filter = resampling_freq
    else:
        # No resampling
        df_resampled = pd.DataFrame({
            'torque_resampled': torque_original
        }, index=df_work.index)
        sampling_freq_for_filter = original_sampling_frequency
    
    # Stage 3: Filtered torque
    if filter_cutoff and filter_cutoff > 0:
        nyquist_freq = 0.5 * sampling_freq_for_filter
        if filter_cutoff < nyquist_freq:
            Wn = filter_cutoff / nyquist_freq
            b, a = butter(filter_order, Wn, btype='low', analog=False)
            torque_filtered = filtfilt(b, a, df_resampled['torque_resampled'].values)
        else:
            torque_filtered = df_resampled['torque_resampled'].values
    else:
        torque_filtered = df_resampled['torque_resampled'].values
    
    # Combine results into a single DataFrame
    result_df = df_resampled.copy()
    result_df['torque_filtered'] = torque_filtered
    
    # Add original torque by interpolating to the resampled timeline
    f_orig = interpolate.interp1d(df_work.index.values, torque_original.values, 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
    result_df['torque_original'] = f_orig(result_df.index.values)
    
    return result_df

def generate_torque_comparison_plot(df, output_dir, file_stem, max_window_s=10):
    """Generate a plot comparing original, resampled, and filtered torque over a max 10s window."""
    torque_df = calculate_torque_processing_stages(df)
    
    if torque_df is None or len(torque_df) == 0:
        print(f"        Cannot generate torque plot - insufficient data or missing columns")
        return
    
    # Limit to max window
    if max_window_s:
        start_time = torque_df.index.min()
        end_time = min(start_time + max_window_s, torque_df.index.max())
        torque_df_plot = torque_df[(torque_df.index >= start_time) & (torque_df.index <= end_time)]
    else:
        torque_df_plot = torque_df
    
    if len(torque_df_plot) == 0:
        return
    
    plt.figure(figsize=(15, 8))
    
    # Plot all three torque signals
    plt.plot(torque_df_plot.index, torque_df_plot['torque_original'], 
             label='Original (Raw)', linewidth=0.8, alpha=0.7)
    plt.plot(torque_df_plot.index, torque_df_plot['torque_resampled'], 
             label='Resampled', linewidth=0.8, alpha=0.8)
    plt.plot(torque_df_plot.index, torque_df_plot['torque_filtered'], 
             label='Filtered', linewidth=1.0)
    
    # Add horizontal lines for reference
    plt.axhline(y=3.6, color='red', linestyle='--', alpha=0.7, label='Max Torque Limit (3.6 Nm)')
    plt.axhline(y=-3.6, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title(f'Torque Comparison: Processing Stages - {file_stem}')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = []
    for col, label in [('torque_original', 'Original'), 
                      ('torque_resampled', 'Resampled'), 
                      ('torque_filtered', 'Filtered')]:
        data = torque_df_plot[col]
        stats_text.append(f'{label}: max={data.max():.2f}, min={data.min():.2f}, std={data.std():.2f} Nm')
    
    plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plot_filename = output_dir / f"torque_comparison.png"
    plt.savefig(plot_filename, dpi=100)
    plt.close()
    
    # Print some statistics
    max_orig = abs(torque_df_plot['torque_original']).max()
    max_filt = abs(torque_df_plot['torque_filtered']).max()
    print(f"        Torque stats - Max original: {max_orig:.2f} Nm, Max filtered: {max_filt:.2f} Nm")

def process_single_file(csv_file: Path, file_output_dir: Path, plot_options: dict):
    """Process a single CSV file and generate plots."""
    print(f"    Processing file: {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"      Error loading CSV '{csv_file.name}': {e}")
        return None

    if df.empty:
        print(f"      File '{csv_file.name}' is empty. Skipping.")
        return None

    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"      Shape: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"      Missing values: {missing_values[missing_values > 0].to_dict()}")

    # Ensure key columns are present
    actual_key_cols = [col for col in KEY_COLUMNS if col in df.columns]
    actual_plot_cols = [col for col in PLOT_COLUMNS if col in df.columns]

    if not actual_key_cols:
        print(f"      No key columns found. Skipping detailed analysis.")
        return None

    # Downsample for plotting speed
    df_plot = downsample_for_plotting(df, plot_options['max_plot_points'])
    if len(df_plot) < len(df):
        print(f"      Downsampled from {len(df)} to {len(df_plot)} points for plotting")

    # Prepare time column
    time_col_for_plot = None
    fs = None
    if 'Time_ms' in df_plot.columns:
        df_plot['Time_s'] = df_plot['Time_ms'] / 1000.0
        time_col_for_plot = 'Time_s'
        if df_plot[time_col_for_plot].nunique() > 1:
            median_diff = df_plot[time_col_for_plot].diff().median()
            if median_diff > 0:
                fs = 1.0 / median_diff
                print(f"      Estimated sampling frequency: {fs:.2f} Hz")

    # Generate plots based on options
    if plot_options['histograms'] and actual_plot_cols:
        generate_histograms(df_plot, actual_plot_cols, file_output_dir, csv_file.stem)

    if plot_options['timeseries'] and time_col_for_plot:
        generate_timeseries(df_plot, time_col_for_plot, file_output_dir, csv_file.stem)
    
    # Generate torque comparison plot (using original df, not downsampled)
    if plot_options['torque']:
        generate_torque_comparison_plot(df, file_output_dir, csv_file.stem, max_window_s=plot_options['torque_window_s'])

    if plot_options['psd'] and fs and fs > 0:
        generate_psd_plots(df_plot, fs, file_output_dir, csv_file.stem, plot_options['max_psd_points'])

    return {'rows': len(df), 'file_name': csv_file.name}

def generate_histograms(df, plot_cols, output_dir, file_stem):
    """Generate histogram plots."""
    num_cols = len(plot_cols)
    ncols = min(3, num_cols)
    nrows = (num_cols + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(plot_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'{col}', fontsize=9)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Histograms - {file_stem}', fontsize=12)
    plt.tight_layout()
    
    plot_filename = output_dir / f"histograms.png"
    plt.savefig(plot_filename, dpi=100)  # Reduced DPI for speed
    plt.close()

def generate_timeseries(df, time_col, output_dir, file_stem):
    """Generate time series plots."""
    # Accelerometer plot
    accel_cols_present = [col for col in ACCEL_COLS if col in df.columns]
    if accel_cols_present:
        plt.figure(figsize=(12, 6))
        for col in accel_cols_present:
            plt.plot(df[time_col], df[col], label=col, linewidth=0.8)
        plt.title(f'Accelerometer Data - {file_stem}')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_filename = output_dir / f"accelerometer_timeseries.png"
        plt.savefig(plot_filename, dpi=100)
        plt.close()

    # Gyroscope plot
    gyro_cols_present = [col for col in GYRO_COLS if col in df.columns]
    if gyro_cols_present:
        plt.figure(figsize=(12, 6))
        for col in gyro_cols_present:
            plt.plot(df[time_col], df[col], label=col, linewidth=0.8)
        plt.title(f'Gyroscope Data - {file_stem}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_filename = output_dir / f"gyroscope_timeseries.png"
        plt.savefig(plot_filename, dpi=100)
        plt.close()

    # Angle comparison plot
    if 'Commanded_Angle' in df.columns and 'Encoder_Angle' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df[time_col], df['Commanded_Angle'], label='Commanded_Angle', linewidth=0.8)
        plt.plot(df[time_col], df['Encoder_Angle'], label='Encoder_Angle', linewidth=0.8)
        plt.title(f'Angle Comparison - {file_stem}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_filename = output_dir / f"angle_comparison_timeseries.png"
        plt.savefig(plot_filename, dpi=100)
        plt.close()

def generate_psd_plots(df, fs, output_dir, file_stem, max_psd_points):
    """Generate separate PSD plots for original method and Welch's method."""
    psd_cols = [col for col in PSD_TARGET_COLS if col in df.columns]
    valid_psd_cols = []
    
    for col in psd_cols:
        if df[col].isnull().all() or df[col].nunique() < 2:
            continue
        valid_psd_cols.append(col)

    if not valid_psd_cols:
        return

    # Limit data points for PSD to speed up computation
    df_psd = df.copy()
    if len(df_psd) > max_psd_points:
        step = len(df_psd) // max_psd_points
        df_psd = df_psd.iloc[::step]

    num_plots = len(valid_psd_cols)
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    
    # Original method plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    nfft = min(256, len(df_psd) // 4)
    
    for i, col in enumerate(valid_psd_cols):
        try:
            axes[i].psd(df_psd[col].dropna(), NFFT=nfft, Fs=fs)
            axes[i].set_title(f'PSD - {col}', fontsize=9)
        except Exception as e:
            print(f"        Error generating original PSD for {col}: {e}")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Power Spectral Density (Original Method) - {file_stem}', fontsize=12)
    plt.tight_layout()
    
    plot_filename = output_dir / f"psd_plots_original.png"
    plt.savefig(plot_filename, dpi=100)
    plt.close()

    # Welch method plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    nperseg = min(len(df_psd) // 8, 1024)
    nperseg = max(nperseg, 64)
    noverlap = nperseg // 2
    
    for i, col in enumerate(valid_psd_cols):
        try:
            data = df_psd[col].dropna().values
            if len(data) < nperseg:
                print(f"        Insufficient data for Welch PSD {col}")
                continue
            
            frequencies, psd = signal.welch(
                data, 
                fs=fs, 
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                detrend='linear',
                scaling='density'
            )
            
            axes[i].semilogy(frequencies, psd)
            axes[i].set_title(f'PSD - {col}', fontsize=9)
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('PSD (units²/Hz)')
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"        Error generating Welch PSD for {col}: {e}")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Power Spectral Density (Welch Method) - {file_stem}', fontsize=12)
    plt.tight_layout()
    
    plot_filename = output_dir / f"psd_plots_welch.png"
    plt.savefig(plot_filename, dpi=100)
    plt.close()

def explore_data(data_dir: Path, output_dir: Path, plot_options: dict):
    """
    Explores data in CSV files within subdirectories of data_dir,
    processing each file individually and saving plots per file.
    """
    if not data_dir.is_dir():
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving exploration plots to: {output_dir.resolve()}")
    
    # Clear old plots
    if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    print("Cleared old plots from output directory.")

    inertia_groups = [d for d in data_dir.iterdir() if d.is_dir()]
    if not inertia_groups:
        print(f"No subdirectories (inertia groups) found in '{data_dir}'.")
        return

    group_summaries = {}

    for group_dir in inertia_groups:
        group_name = group_dir.name
        print(f"\nProcessing inertia group: {group_name}")
        
        group_output_dir = output_dir / group_name
        csv_files = list(group_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"  No CSV files found in '{group_dir}'.")
            continue

        # Sort files for consistent ordering
        csv_files.sort()
        
        # Limit to first file only if requested
        if plot_options.get('first_only', False):
            csv_files = csv_files[:1]
            print(f"  Processing only first file: {csv_files[0].name}")
        else:
            print(f"  Found {len(csv_files)} CSV files")

        file_summaries = []
        total_rows = 0

        for csv_file in csv_files:
            file_output_dir = group_output_dir / csv_file.stem
            summary = process_single_file(csv_file, file_output_dir, plot_options)
            
            if summary:
                file_summaries.append(summary)
                total_rows += summary['rows']

        group_summaries[group_name] = {
            'total_rows': total_rows,
            'file_count': len(file_summaries),
            'files': file_summaries
        }

        print(f"  Completed {group_name}: {len(file_summaries)} files, {total_rows} total rows")

    # Generate summary
    print("\n--- Processing Summary ---")
    for group_name, summary in group_summaries.items():
        print(f"  {group_name}: {summary['file_count']} files, {summary['total_rows']} rows")
        for file_info in summary['files']:
            print(f"    {file_info['file_name']}: {file_info['rows']} rows")

def main():
    parser = argparse.ArgumentParser(
        description="Explore actuator data CSV files, processing each file individually."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["real", "synthetic"],
        default="real",
        help="Type of data to explore: 'real' or 'synthetic'. Default: real",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing inertia group subfolders with CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "exploration_plots",
        help="Directory to save plots. Default: ../exploration_plots",
    )
    parser.add_argument(
        "--max_plot_points",
        type=int,
        default=10000,
        help="Maximum data points to use for plotting (for speed). Default: 10000",
    )
    parser.add_argument(
        "--max_psd_points", 
        type=int,
        default=5000,
        help="Maximum data points for PSD computation. Default: 5000",
    )
    parser.add_argument(
        "--skip_histograms",
        action="store_true",
        help="Skip histogram generation",
    )
    parser.add_argument(
        "--skip_timeseries",
        action="store_true",
        help="Skip time series plots",
    )
    parser.add_argument(
        "--skip_torque",
        action="store_true",
        help="Skip torque comparison plots",
    )
    parser.add_argument(
        "--skip_psd",
        action="store_true",
        help="Skip PSD plots",
    )
    parser.add_argument(
        "--first_only",
        action="store_true",
        help="Process only the first CSV file in each group (for faster execution)",
    )
    parser.add_argument(
        "--torque_window_s",
        type=float,
        default=5.0,
        help="Maximum time window (in seconds) for torque comparison plot. Default: 10.0",
    )
    
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent.parent / "data" / args.data_type

    plot_options = {
        'histograms': not args.skip_histograms,
        'timeseries': not args.skip_timeseries,
        'torque': not args.skip_torque,
        'psd': not args.skip_psd,
        'max_plot_points': args.max_plot_points,
        'max_psd_points': args.max_psd_points,
        'first_only': args.first_only,
        'torque_window_s': args.torque_window_s,
    }

    explore_data(args.data_dir, args.output_dir, plot_options)

if __name__ == "__main__":
    main()