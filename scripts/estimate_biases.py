#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path

def estimate_biases(static_csv_path: str):
    """Estimate sensor biases from static hold data."""
    
    print(f"Loading static hold data from: {static_csv_path}")
    df = pd.read_csv(static_csv_path)
    
    # Calculate biases (mean during static hold - should be zero ideally)
    acc_y_bias = df['Acc_Y'].mean()
    gyro_z_bias = df['Gyro_Z'].mean()
    
    print(f'\nEstimated biases from static hold data:')
    print(f'  Acc_Y bias: {acc_y_bias:.6f}')
    print(f'  Gyro_Z bias: {gyro_z_bias:.6f}')
    
    # Also show some statistics for context
    print(f'\nStatic hold statistics:')
    print(f'  Acc_Y: mean={df["Acc_Y"].mean():.6f}, std={df["Acc_Y"].std():.6f}')
    print(f'  Gyro_Z: mean={df["Gyro_Z"].mean():.6f}, std={df["Gyro_Z"].std():.6f}')
    print(f'  Data points: {len(df)}')
    print(f'  Duration: {(df["Time_ms"].max() - df["Time_ms"].min()) / 1000:.1f} seconds')
    
    return acc_y_bias, gyro_z_bias

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    static_csv_path = script_dir.parent / "data" / "static" / "static_segment_plateau2.csv"
    estimate_biases(str(static_csv_path)) 