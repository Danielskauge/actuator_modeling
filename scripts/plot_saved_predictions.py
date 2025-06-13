#!/usr/bin/env python3
"""Generate high-resolution torque-prediction plots from saved CSV data.

The CSV must contain at least the columns:
    timestamp, actual_torque, predicted_torque
(these are exactly what TestPredictionPlotter writes).

Example usage:
    python plot_saved_predictions.py --csv path/to/predictions_vs_targets_test.csv \
                                     --out path/to/plot.png
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def make_last_window_plot(df: pd.DataFrame, window_s: float = 3.0, dpi: int = 600, figsize=(20, 6)) -> plt.Figure:
    """Return a Matplotlib Figure showing actual vs predicted torque for the *last* `window_s` seconds."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    # Ensure the DataFrame is sorted by timestamp
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    end_ts = df_sorted["timestamp"].iloc[-1]
    start_ts = max(end_ts - window_s, df_sorted["timestamp"].iloc[0])
    mask = (df_sorted["timestamp"] >= start_ts) & (df_sorted["timestamp"] <= end_ts)
    slice_df = df_sorted.loc[mask]
    if slice_df.empty:
        raise ValueError("No data found inside the requested time window.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(slice_df["timestamp"], slice_df["actual_torque"], label="Actual Torque", color="blue")
    ax.plot(slice_df["timestamp"], slice_df["predicted_torque"], label="Predicted Torque", color="orange", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"Predicted vs Actual Torque – Last {window_s:.0f}s")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-1, 1)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs actual torques from saved CSV data.")
    parser.add_argument("--csv", required=True, help="Path to predictions_vs_targets_test.csv")
    parser.add_argument("--out", default=None, help="Output image file (PNG). If omitted, will be '<csv_dir>/last3s_plot.png'")
    parser.add_argument("--window", type=float, default=3.0, help="Time window in seconds (default: 3.0)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig = make_last_window_plot(df, window_s=args.window)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.csv), "last3s_plot.png")

    fig.savefig(out_path, dpi=fig.dpi, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main() 