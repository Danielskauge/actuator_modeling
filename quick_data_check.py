import pandas as pd
import numpy as np

print("Loading full CSV file...")
# Load the full file this time
df = pd.read_csv('data/real/new_pos_A_300g_data/new_pos_A__300g__data_1.csv')
print(f'Total samples: {len(df)}')

print('\nAcc_Y statistics (full file):')
print(f'  Mean: {df["Acc_Y"].mean():.4f}')
print(f'  Std: {df["Acc_Y"].std():.4f}')
print(f'  Min: {df["Acc_Y"].min():.4f}')
print(f'  Max: {df["Acc_Y"].max():.4f}')
print(f'  95th percentile: {df["Acc_Y"].quantile(0.95):.4f}')
print(f'  99th percentile: {df["Acc_Y"].quantile(0.99):.4f}')
print(f'  99.9th percentile: {df["Acc_Y"].quantile(0.999):.4f}')
print(f'  99.99th percentile: {df["Acc_Y"].quantile(0.9999):.4f}')

# Calculate raw torque (before filtering)
inertia = 0.013
radius = 0.03
torque_values_raw = inertia * df['Acc_Y'] / radius

print(f'\nRaw torques (before filtering, using inertia={inertia}, radius={radius}):')
print(f'  Mean: {torque_values_raw.mean():.4f} Nm')
print(f'  Std: {torque_values_raw.std():.4f} Nm')
print(f'  Min: {torque_values_raw.min():.4f} Nm')
print(f'  Max: {torque_values_raw.max():.4f} Nm')
print(f'  95th percentile: {torque_values_raw.quantile(0.95):.4f} Nm')
print(f'  99th percentile: {torque_values_raw.quantile(0.99):.4f} Nm')
print(f'  99.9th percentile: {torque_values_raw.quantile(0.999):.4f} Nm')
print(f'  99.99th percentile: {torque_values_raw.quantile(0.9999):.4f} Nm')

# Check for outliers
outlier_threshold = 3.6  # Your max torque limit
outliers = torque_values_raw[abs(torque_values_raw) > outlier_threshold]
print(f'\nNumber of samples exceeding {outlier_threshold} Nm: {len(outliers)} out of {len(torque_values_raw)} ({100*len(outliers)/len(torque_values_raw):.2f}%)')

if len(outliers) > 0:
    print(f'Outlier torques: {outliers.tolist()[:10]}...')  # Show first 10
    corresponding_acc = df['Acc_Y'][abs(torque_values_raw) > outlier_threshold]
    print(f'Corresponding Acc_Y values: {corresponding_acc.tolist()[:10]}...')
    
    # Look at extreme values
    max_torque_idx = abs(torque_values_raw).idxmax()
    print(f'\nMax torque at index {max_torque_idx}:')
    print(f'  Torque: {torque_values_raw.iloc[max_torque_idx]:.4f} Nm')
    print(f'  Acc_Y: {df["Acc_Y"].iloc[max_torque_idx]:.4f} m/s²')
    print(f'  Time: {df["Time_ms"].iloc[max_torque_idx]:.1f} ms')
    
    # Look at context around the spike
    start_idx = max(0, max_torque_idx - 5)
    end_idx = min(len(df), max_torque_idx + 6)
    print(f'\nContext around max torque (±5 samples):')
    for i in range(start_idx, end_idx):
        marker = ">>>" if i == max_torque_idx else "   "
        print(f'{marker} Index {i}: Acc_Y={df["Acc_Y"].iloc[i]:.4f}, Torque={torque_values_raw.iloc[i]:.4f} Nm')

# Check for sudden jumps that might indicate issues
acc_y_diff = df['Acc_Y'].diff().abs()
print(f'\nAcc_Y sudden changes:')
print(f'  Max diff: {acc_y_diff.max():.4f} m/s²')
print(f'  99.9th percentile diff: {acc_y_diff.quantile(0.999):.4f} m/s²')

large_jumps = acc_y_diff > 1.0  # More than 1 m/s² change between samples
if large_jumps.any():
    print(f'  Large jumps (>1 m/s²): {large_jumps.sum()} occurrences')
    jump_indices = df[large_jumps].index[:5]  # First 5 occurrences
    for idx in jump_indices:
        if idx > 0:
            print(f'    At index {idx}: {df["Acc_Y"].iloc[idx-1]:.4f} -> {df["Acc_Y"].iloc[idx]:.4f} m/s²') 