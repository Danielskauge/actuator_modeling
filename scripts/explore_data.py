#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

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
# Commanded_Angle is also in PLOT_COLUMNS for individual histogram/timeseries if not handled by PSD explicitly
PSD_TARGET_COLS = ACCEL_COLS + ['Commanded_Angle', 'Gyro_Z', 'Encoder_Angle']
INDIVIDUAL_TIMESERIES_COLS = ['Encoder_Angle', 'Commanded_Angle']
    

def explore_data(data_dir: Path, output_dir: Path):
    """
    Explores data in CSV files within subdirectories of data_dir,
    prints summaries, and saves plots to output_dir.
    """
    if not data_dir.is_dir():
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving exploration plots and summaries to: {output_dir.resolve()}")
    
    # Delete old plots
    for item in output_dir.iterdir():
        if item.is_file() and item.suffix == '.png':
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    print("Cleared old plots from output directory.")

    inertia_groups = [d for d in data_dir.iterdir() if d.is_dir()]
    if not inertia_groups:
        print(f"No subdirectories (inertia groups) found in '{data_dir}'.")
        return

    group_data_counts = {}  # To store total rows per inertia group

    for group_dir in inertia_groups:
        group_name = group_dir.name
        print(f"\nProcessing inertia group: {group_name}")
        group_data_counts[group_name] = {'total_rows': 0, 'file_count': 0}
        
        group_output_dir = output_dir / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = list(group_dir.glob('*.csv'))
        if not csv_files:
            print(f"  No CSV files found in '{group_dir}'.")
            continue

        # Combine all CSV files in the group
        combined_df = pd.DataFrame()
        for csv_file in csv_files:
            print(f"\n  Loading file: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                group_data_counts[group_name]['total_rows'] += df.shape[0]
                group_data_counts[group_name]['file_count'] += 1
            except Exception as e:
                print(f"    Error loading CSV '{csv_file.name}': {e}")

        if combined_df.empty:
            print(f"  No valid data found in '{group_dir}'.")
            continue

        print("\n    Basic Information:")
        combined_df.info()
        print(f"      Shape: {combined_df.shape}")
        
        print("\n    Missing Values (per column):")
        missing_values = combined_df.isnull().sum()
        if missing_values.sum() == 0:
            print("      No missing values found.")
        else:
            print(missing_values[missing_values > 0])

        # Ensure all key columns are present, add if missing for describe/plot
        for col in KEY_COLUMNS:
            if col not in combined_df.columns:
                print(f"      Warning: Key column '{col}' not found in combined data. Skipping for stats/plots.")
        
        actual_key_cols_in_df = [col for col in KEY_COLUMNS if col in combined_df.columns]
        actual_plot_cols_for_hist_in_df = [col for col in PLOT_COLUMNS if col in combined_df.columns] # For histograms

        if not actual_key_cols_in_df:
            print(f"      No key columns found in combined data. Skipping detailed analysis.")
            continue
            
        print("\n    Descriptive Statistics (for key columns):")
        print(combined_df[actual_key_cols_in_df].describe())

        time_col_for_plot = None
        fs = None
        if 'Time_ms' in combined_df.columns:
            combined_df['Time_s'] = combined_df['Time_ms'] / 1000.0
            time_col_for_plot = 'Time_s'
            if combined_df[time_col_for_plot].nunique() > 1:
                median_diff = combined_df[time_col_for_plot].diff().median()
                if median_diff > 0:
                    fs = 1.0 / median_diff
                    print(f"      Estimated sampling frequency: {fs:.2f} Hz")
                else:
                    print("      Warning: Could not estimate sampling frequency (median time diff <= 0). PSD plots may be unreliable or skipped.")
            else:
                print("      Warning: Not enough unique time points to estimate sampling frequency. PSD plots will be skipped.")
        else:
            print("      Warning: 'Time_ms' column not found. Cannot create time series plots or reliable PSD plots.")

        # --- Plotting ---
        # Histograms
        if actual_plot_cols_for_hist_in_df:
            print(f"\n    Generating combined histograms for plottable columns...")
            num_cols = len(actual_plot_cols_for_hist_in_df)
            # Adjust subplot layout dynamically, aim for ~3 columns
            ncols_subplot = min(3, num_cols)
            nrows_subplot = (num_cols + ncols_subplot - 1) // ncols_subplot # Ceiling division
            
            fig_hist, axes_hist = plt.subplots(nrows_subplot, ncols_subplot, figsize=(5 * ncols_subplot, 4 * nrows_subplot))
            axes_hist = axes_hist.flatten() # Flatten in case of multi-dim array

            for i, col in enumerate(actual_plot_cols_for_hist_in_df):
                sns.histplot(combined_df[col].dropna(), kde=True, ax=axes_hist[i])
                axes_hist[i].set_title(f'Histogram of {col}', fontsize=9)
                axes_hist[i].set_xlabel(col)
                axes_hist[i].set_ylabel('Frequency')
            
            # Hide any unused subplots
            for j in range(i + 1, len(axes_hist)):
                fig_hist.delaxes(axes_hist[j])

            fig_hist.suptitle(f'Histograms of Key Features\n({group_name})', fontsize=12, y=1.0) # Adjust y to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
            plot_filename = group_output_dir / f"ALL_Features_histograms.png"
            plt.savefig(plot_filename)
            plt.close(fig_hist)
            print(f"      Combined histograms saved to {plot_filename}")

        # Combined Time Series Plots
        accel_cols_present = [col for col in ACCEL_COLS if col in combined_df.columns]
        gyro_cols_present = [col for col in GYRO_COLS if col in combined_df.columns]

        if time_col_for_plot:
            if accel_cols_present:
                print(f"\n    Generating combined time series plot for Accelerometer data...")
                plt.figure(figsize=(12, 6))
                for col in accel_cols_present:
                    sns.lineplot(x=combined_df[time_col_for_plot], y=combined_df[col], label=col)
                plt.title(f'Accelerometer Data vs. Time\n({group_name})', fontsize=10)
                plt.xlabel('Time (s)')
                plt.ylabel('Acceleration') # General unit
                plt.legend()
                plot_filename = group_output_dir / f"ACC_XYZ_timeseries.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"      Accelerometer combined time series plot saved to {plot_filename}")

            if gyro_cols_present:
                print(f"\n    Generating combined time series plot for Gyroscope data...")
                plt.figure(figsize=(12, 6))
                for col in gyro_cols_present:
                    sns.lineplot(x=combined_df[time_col_for_plot], y=combined_df[col], label=col)
                plt.title(f'Gyroscope Data vs. Time\n({group_name})', fontsize=10)
                plt.xlabel('Time (s)')
                plt.ylabel('Angular Velocity') # General unit
                plt.legend()
                plot_filename = group_output_dir / f"GYRO_XYZ_timeseries.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"      Gyroscope combined time series plot saved to {plot_filename}")
        
        # Combined Time Series Plot for Commanded and Encoder Angle
        if time_col_for_plot and 'Commanded_Angle' in combined_df.columns and 'Encoder_Angle' in combined_df.columns:
            print(f"\n    Generating combined time series plot for Commanded and Encoder Angle...")
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=combined_df[time_col_for_plot], y=combined_df['Commanded_Angle'], label='Commanded_Angle')
            sns.lineplot(x=combined_df[time_col_for_plot], y=combined_df['Encoder_Angle'], label='Encoder_Angle')
            plt.title(f'Commanded and Encoder Angle vs. Time\n({group_name})', fontsize=10)
            plt.xlabel('Time (s)')
            plt.ylabel('Angle')
            plt.legend()
            plot_filename = group_output_dir / f"Commanded_Encoder_Angle_timeseries.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"      Combined time series plot for Commanded and Encoder Angle saved to {plot_filename}")
        
        # PSD Plots
        psd_cols_to_plot = [col for col in PSD_TARGET_COLS if col in combined_df.columns]

        if fs and psd_cols_to_plot:
            print(f"\n    Generating combined PSD plots (Fs={fs:.2f} Hz)...")
            # Only keep columns that will actually be plotted
            valid_psd_cols = []
            for col in psd_cols_to_plot:
                if combined_df[col].isnull().all():
                    print(f"      Skipping PSD for {col} as it contains all NaN values.")
                    continue
                if combined_df[col].nunique() < 2:
                    print(f"      Skipping PSD for {col} as it has less than 2 unique values or is constant.")
                    continue
                nfft_val = min(256, len(combined_df[col].dropna()) - 1 if len(combined_df[col].dropna()) > 1 else 1)
                if nfft_val <= 0:
                    print(f"      Skipping PSD for {col} due to insufficient data points after dropna for NFFT.")
                    continue
                valid_psd_cols.append((col, nfft_val))

            num_psd_plots = len(valid_psd_cols)
            if num_psd_plots == 0:
                print(f"      No PSD plots were generated for group {group_name}.")
            else:
                ncols_psd_subplot = min(3, num_psd_plots)
                nrows_psd_subplot = (num_psd_plots + ncols_psd_subplot - 1) // ncols_psd_subplot
                fig_psd, axes_psd = plt.subplots(nrows_psd_subplot, ncols_psd_subplot, figsize=(6 * ncols_psd_subplot, 4 * nrows_psd_subplot))
                if num_psd_plots == 1:
                    axes_psd = [axes_psd]
                else:
                    axes_psd = axes_psd.flatten()
                for i, (col, nfft_val) in enumerate(valid_psd_cols):
                    try:
                        axes_psd[i].psd(combined_df[col].dropna(), NFFT=nfft_val, Fs=fs)
                        axes_psd[i].set_title(f'PSD of {col}', fontsize=9)
                    except Exception as e_psd:
                        print(f"      Error generating PSD for {col}: {e_psd}")
                # Hide any unused subplots
                for j in range(i + 1, len(axes_psd)):
                    fig_psd.delaxes(axes_psd[j])
                fig_psd.suptitle(f'Power Spectral Density (PSD) Plots\n({group_name}, Fs={fs:.2f} Hz)', fontsize=12, y=1.0)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plot_filename = group_output_dir / f"ALL_Features_PSD.png"
                plt.savefig(plot_filename)
                print(f"      Combined PSD plots saved to {plot_filename}")
                plt.close(fig_psd)
            
        elif not fs and psd_cols_to_plot : # if there are cols to plot but fs is None
             print(f"\n    Skipping PSD plots because sampling frequency (Fs) could not be determined or is invalid.")
        
        print(f"\n  Finished analysis for {group_name}")

    # --- Summary of data amounts per group ---
    print("\n--- Inertia Group Data Summary ---")
    if not group_data_counts:
        print("No data processed to summarize.")
    else:
        for group_name, counts in group_data_counts.items():
            print(f"  Group: {group_name}, Total Rows: {counts['total_rows']}, Files: {counts['file_count']}")
        
        # Plotting the summary
        group_names = list(group_data_counts.keys())
        total_rows_list = [counts['total_rows'] for counts in group_data_counts.values()]

        if group_names and total_rows_list:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=group_names, y=total_rows_list)
            plt.title('Total Data Rows per Inertia Group')
            plt.xlabel('Inertia Group')
            plt.ylabel('Total Number of Rows')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            summary_plot_filename = output_dir / "inertia_group_data_summary.png"
            plt.savefig(summary_plot_filename)
            plt.close()
            print(f"\nSummary plot saved to: {summary_plot_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Explore actuator data CSV files, print summaries, and generate plots."
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
        help="Directory containing inertia group subfolders with CSV files. If not provided, defaults to ../data/<data_type>.",
    )
    parser.add_argument(
        
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "exploration_plots",
        help="Directory to save plots and summaries. Default: ../exploration_plots",
    )
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent.parent / "data" / args.data_type

    explore_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()