#!/usr/bin/env python
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

        for csv_file in csv_files:
            print(f"\n  Analyzing file: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"    Error loading CSV '{csv_file.name}': {e}")
                continue

            group_data_counts[group_name]['total_rows'] += df.shape[0]
            group_data_counts[group_name]['file_count'] += 1

            print("\n    Basic Information:")
            df.info(buf=open(os.devnull, 'w')) # Suppress default print, manage manually
            print(f"      Shape: {df.shape}")
            
            print("\n    Missing Values (per column):")
            missing_values = df.isnull().sum()
            if missing_values.sum() == 0:
                print("      No missing values found.")
            else:
                print(missing_values[missing_values > 0])

            # Ensure all key columns are present, add if missing for describe/plot
            for col in KEY_COLUMNS:
                if col not in df.columns:
                    print(f"      Warning: Key column '{col}' not found in {csv_file.name}. Skipping for stats/plots.")
            
            actual_key_cols_in_df = [col for col in KEY_COLUMNS if col in df.columns]
            actual_plot_cols_in_df = [col for col in PLOT_COLUMNS if col in df.columns]

            if not actual_key_cols_in_df:
                print(f"      No key columns found in {csv_file.name}. Skipping detailed analysis.")
                continue
                
            print("\n    Descriptive Statistics (for key columns):")
            print(df[actual_key_cols_in_df].describe())

            if 'Time_ms' in df.columns:
                df['Time_s'] = df['Time_ms'] / 1000.0
                time_col_for_plot = 'Time_s'
            else:
                print("      Warning: 'Time_ms' column not found. Cannot create time series plots.")
                time_col_for_plot = None

            # --- Plotting ---
            # Histograms
            if actual_plot_cols_in_df:
                print(f"\n    Generating histograms for key columns...")
                for col in actual_plot_cols_in_df:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Histogram of {col}\n({group_name} - {csv_file.name})', fontsize=10)
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plot_filename = group_output_dir / f"{csv_file.stem}_{col}_hist.png"
                    plt.savefig(plot_filename)
                    plt.close()
                print(f"      Histograms saved to {group_output_dir}")

            # Time Series Plots
            if time_col_for_plot and actual_plot_cols_in_df:
                print(f"\n    Generating time series plots...")
                for col in actual_plot_cols_in_df:
                    plt.figure(figsize=(12, 6))
                    sns.lineplot(x=df[time_col_for_plot], y=df[col])
                    plt.title(f'{col} vs. Time\n({group_name} - {csv_file.name})', fontsize=10)
                    plt.xlabel('Time (s)')
                    plt.ylabel(col)
                    plot_filename = group_output_dir / f"{csv_file.stem}_{col}_timeseries.png"
                    plt.savefig(plot_filename)
                    plt.close()
                print(f"      Time series plots saved to {group_output_dir}")
            
            print(f"\n  Finished analysis for {csv_file.name}")
        print(f"\nFinished processing inertia group: {group_name}")

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
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "real",
        help="Directory containing inertia group subfolders with CSV files. Default: ../data/real",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "exploration_plots",
        help="Directory to save plots and summaries. Default: ../exploration_plots",
    )
    args = parser.parse_args()

    explore_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()