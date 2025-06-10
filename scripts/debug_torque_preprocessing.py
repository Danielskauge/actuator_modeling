#!/usr/bin/env python3
"""
Script to debug torque preprocessing steps of ActuatorDataset.
Generates plots of raw, resampled, and filtered torque for a given CSV segment.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def main():
    parser = argparse.ArgumentParser(description="Debug torque preprocessing pipeline")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--inertia", type=float, required=True, help="Inertia (kg*m^2)")
    parser.add_argument("--radius", type=float, required=True, help="Radius for accel->ang_acc (m)")
    parser.add_argument("--resample_freq", type=float, default=None, help="Resample frequency in Hz (optional)")
    parser.add_argument("--filter_cutoff", type=float, default=None, help="Butterworth cutoff frequency (Hz, optional)")
    parser.add_argument("--filter_order", type=int, default=4, help="Butterworth filter order")
    parser.add_argument("--accel_axis", type=str, default="Acc_Y", help="Column name for acceleration axis in CSV")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    args = parser.parse_args()

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV and compute time in seconds
    df = pd.read_csv(args.csv)
    df['time_s'] = df['Time_ms'] / 1000.0
    df['time_s'] -= df['time_s'].min()
    df = df.set_index('time_s').sort_index()

    # --- Stage 1: Trim first 5 seconds ---
    df_trim = df[df.index >= 5.0]
    time_trim = df_trim.index.values
    lin_acc_trim = df_trim[args.accel_axis].values
    ang_acc_trim = lin_acc_trim / args.radius
    torque_raw = args.inertia * ang_acc_trim

    plt.figure()
    plt.plot(time_trim, torque_raw, label='Raw Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque')
    plt.title('Stage 1: Raw Torque (after 5s trim)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'stage1_raw_torque.png'))

    # --- Stage 2: Resample (if requested) ---
    if args.resample_freq:
        dt = 1.0 / args.resample_freq
        new_times = np.arange(df_trim.index.min(), df_trim.index.max() + dt, dt)
        # Interpolate linear acceleration
        interp_fn = interp1d(time_trim, lin_acc_trim, kind='linear', bounds_error=False, fill_value='extrapolate')
        lin_acc_res = interp_fn(new_times)
        ang_acc_res = lin_acc_res / args.radius
        torque_res = args.inertia * ang_acc_res
        time_res = new_times

        plt.figure()
        plt.plot(time_res, torque_res, label=f'Resampled Torque @ {args.resample_freq}Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque')
        plt.title('Stage 2: Resampled Torque')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'stage2_resampled_torque.png'))
    else:
        torque_res = torque_raw
        time_res = time_trim

    # --- Stage 3: Filtering (if requested) ---
    if args.filter_cutoff:
        # Determine sampling freq
        fs = args.resample_freq if args.resample_freq else 1.0 / (time_trim[1] - time_trim[0])
        nyq = 0.5 * fs
        if args.filter_cutoff < nyq:
            Wn = args.filter_cutoff / nyq
            b, a = butter(args.filter_order, Wn, btype='low', analog=False)
            torque_filt = filtfilt(b, a, torque_res)

            plt.figure()
            plt.plot(time_res, torque_filt, label=f'Filtered Torque (cutoff={args.filter_cutoff}Hz)')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque')
            plt.title('Stage 3: Filtered Torque')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, 'stage3_filtered_torque.png'))
        else:
            print(f"Cutoff {args.filter_cutoff}Hz >= Nyquist {nyq:.2f}Hz; skipping filter")
            torque_filt = torque_res
    else:
        torque_filt = torque_res

    print(f"Plots saved to {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main() 