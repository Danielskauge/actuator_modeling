import argparse
import glob
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_csv_files(base_dir: str) -> List[str]:
    """Recursively find all predictions_vs_targets_test.csv files under *base_dir*."""
    pattern = os.path.join(base_dir, "**", "plots", "predictions_vs_targets_test.csv")
    return glob.glob(pattern, recursive=True)


def extract_fold_info(path: str) -> str:
    """Return a short label like 'fold_3' or 'global' parsed from *path*."""
    m = re.search(r"fold_(\d+)", path)
    return f"fold_{m.group(1)}" if m else "global"


def scatter_plot(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(df["actual_torque"], df["predicted_torque"], s=5, alpha=0.5)
    lims = [
        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),
        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Torque")
    plt.ylabel("Predicted Torque")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def time_series_plot(df: pd.DataFrame, title: str, out_path: str) -> None:
    df_sorted = df.sort_values("timestamp")
    plt.figure(figsize=(10, 4))
    plt.plot(df_sorted["timestamp"], df_sorted["actual_torque"], label="Actual", linewidth=0.8)
    plt.plot(df_sorted["timestamp"], df_sorted["predicted_torque"], label="Predicted", linewidth=0.8, alpha=0.8)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Torque")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def residual_hist(df: pd.DataFrame, title: str, out_path: str) -> None:
    resid = df["actual_torque"] - df["predicted_torque"]
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40, color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_plots(csv_path: str, output_root: str) -> None:
    df = pd.read_csv(csv_path)
    label = extract_fold_info(csv_path)
    base_title = f"{label}: Torque Prediction"

    os.makedirs(output_root, exist_ok=True)

    scatter_out = os.path.join(output_root, f"scatter_{label}.png")

    scatter_plot(df, f"{base_title} – Predicted vs Actual", scatter_out)

    print(f"Saved scatter plot for {label} to {scatter_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate test-set plots from saved CSV files.")
    parser.add_argument("path", help="Either the CSV file itself or the run output directory that contains it in <run>/plots/")
    parser.add_argument("--out_dir", default=None, help="Directory to store regenerated plots. Defaults to <run_dir>/analysis_plots or <csv parent dir>")
    args = parser.parse_args()

    # Resolve CSV path
    if args.path.endswith(".csv"):
        csv_path = args.path
    else:
        # Treat as run directory; search for CSV inside plots/
        pattern = os.path.join(args.path, "**", "plots", "predictions_vs_targets_test.csv")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print("Could not find predictions_vs_targets_test.csv under", args.path)
            return
        if len(matches) > 1:
            print("Multiple CSV files found; please supply the exact CSV file path. Found:")
            for m in matches:
                print(" •", m)
            return
        csv_path = matches[0]

    run_dir = os.path.dirname(os.path.dirname(csv_path))  # up to run folder
    output_root = args.out_dir or os.path.join(run_dir, "analysis_plots")
    os.makedirs(output_root, exist_ok=True)

    generate_plots(csv_path, output_root)


if __name__ == "__main__":
    main() 