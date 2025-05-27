#!/usr/bin/env python3
"""Estimate PD gains (Kp, Kd) from actuator log.

CSV must contain at minimum:
    time            [s]  (monotonic, uniform or not)
    target_angle    [rad]
    angle           [rad]
    ang_vel         [rad/s]               (joint velocity)
    ang_accel       [rad/s^2]             (joint acceleration)

Optionally:
    target_ang_vel  [rad/s]               (pre‑computed derivative); if absent we derive it.

Usage
-----
python pd_fit.py log.csv 0.075  --plot
                   ^      ^ inertia [kg·m²]
                   |
                   path to CSV

The script:
  1. Computes tau_meas = I_nom * ang_accel
  2. Builds error e = target_angle - angle
     and error rate e_dot.
  3. Solves least‑squares [Kp, Kd] to minimise ||tau_meas - (Kp*e + Kd*e_dot)||²
  4. Reports Kp, Kd, R², RMSE, MAE
  5. Optionally plots actual vs model torque time‑series.
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def finite_diff(col, t):
    """Central finite difference with edge forward/backward"""
    d = np.zeros_like(col)
    d[1:-1] = (col[2:] - col[:-2]) / (t[2:] - t[:-2])
    d[0] = (col[1] - col[0]) / (t[1] - t[0])
    d[-1] = (col[-1] - col[-2]) / (t[-1] - t[-2])
    return d

def fit_pd(df: pd.DataFrame, I_nom: float) -> Tuple[float, float, pd.Series]:
    # 1. measured torque
    tau_meas = I_nom * df['ang_accel'].values

    # 2. errors
    e = df['target_angle'].values - df['angle'].values

    if 'target_ang_vel' in df.columns:
        e_dot = df['target_ang_vel'].values - df['ang_vel'].values
    else:
        # derive target vel numerically
        e_dot = finite_diff(df['target_angle'].values, df['time'].values) - df['ang_vel'].values

    # 3. least squares solve
    X = np.column_stack([e, e_dot])          # shape (N,2)
    coeffs, *_ = np.linalg.lstsq(X, tau_meas, rcond=None)
    Kp, Kd = coeffs

    tau_pred = X @ coeffs
    df_out = df.copy()
    df_out['tau_meas'] = tau_meas
    df_out['tau_pred'] = tau_pred
    df_out['tau_error'] = tau_meas - tau_pred
    return Kp, Kd, df_out

def metrics(df: pd.DataFrame) -> dict:
    y = df['tau_meas'].values
    y_hat = df['tau_pred'].values
    rmse = np.sqrt(np.mean((y_hat - y)**2))
    mae = np.mean(np.abs(y_hat - y))
    r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}

def make_plot(df: pd.DataFrame, path: pathlib.Path, Kp: float, Kd: float):
    plt.figure(figsize=(9,4))
    plt.plot(df['time'], df['tau_meas'], label='measured', linewidth=1)
    plt.plot(df['time'], df['tau_pred'], label='Kp*e + Kd*e_dot', linewidth=1, linestyle='--')
    plt.xlabel('time [s]')
    plt.ylabel('joint torque [Nm]')
    plt.title(f'PD fit  Kp={Kp:.3f}, Kd={Kd:.3f}')
    plt.legend()
    plt.tight_layout()
    out = path.with_suffix('.png')
    plt.savefig(out, dpi=150)
    print(f'Plot saved → {out}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=pathlib.Path, help='log file with required columns')
    parser.add_argument('inertia', type=float, help='nominal joint inertia in kg·m²')
    parser.add_argument('--plot', action='store_true', help='save PNG plot')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {'time', 'target_angle', 'angle', 'ang_vel', 'ang_accel'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'Missing columns: {missing}')

    Kp, Kd, df_out = fit_pd(df, args.inertia)
    print(f'Kp = {Kp:.4f}  [Nm/rad]')
    print(f'Kd = {Kd:.4f}  [Nm·s/rad]')

    m = metrics(df_out)
    print('Metrics:', ', '.join(f'{k} = {v:.4f}' for k,v in m.items()))

    if args.plot:
        make_plot(df_out, args.csv, Kp, Kd)

if __name__ == '__main__':
    main()
