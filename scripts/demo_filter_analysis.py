#!/usr/bin/env python
"""
Demo script for the filter effects analysis tool.

This script generates synthetic actuator data and demonstrates
how to use the filter analysis functionality.

Usage:
    python scripts/demo_filter_analysis.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

def generate_demo_data():
    """Generate synthetic actuator data for demonstration."""
    print("Generating synthetic actuator data for demonstration...")
    
    # Create temporary data directory
    temp_dir = Path("demo_data")
    temp_dir.mkdir(exist_ok=True)
    
    # Clear any existing demo data
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Create inertia group directory
    group_dir = temp_dir / "0.010kgm2"
    group_dir.mkdir()
    
    # Generate synthetic data
    fs = 100  # 100 Hz sampling
    duration = 5  # 5 seconds
    t = np.linspace(0, duration, int(fs * duration))
    time_ms = t * 1000
    
    # Generate realistic actuator motion
    # Target angle: step response with some oscillation
    target_freq = 0.5  # Hz
    target_amplitude = 45  # degrees
    target_angle = target_amplitude * (0.5 + 0.5 * np.tanh(2 * (t - 1))) * np.sin(2 * np.pi * target_freq * t)
    
    # Encoder angle: follows target with some lag and noise
    encoder_angle = target_angle * 0.9 + 2 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.5, len(t))
    
    # Gyroscope data (angular velocity)
    gyro_z = np.gradient(encoder_angle * np.pi / 180, 1/fs) * 180 / np.pi  # Convert to deg/s
    gyro_z += np.random.normal(0, 1, len(t))  # Add noise
    
    # Accelerometer data (with high-frequency noise to show filter effects)
    # Base tangential acceleration from motion
    ang_accel = np.gradient(gyro_z * np.pi / 180, 1/fs)  # rad/s²
    radius = 0.2  # meters
    acc_y_base = ang_accel * radius  # tangential acceleration
    
    # Add high-frequency noise (this is what the filter should remove)
    noise_freq1 = 25  # Hz
    noise_freq2 = 35  # Hz
    noise_freq3 = 60  # Hz
    high_freq_noise = (
        0.5 * np.sin(2 * np.pi * noise_freq1 * t) +
        0.3 * np.sin(2 * np.pi * noise_freq2 * t) +
        0.2 * np.sin(2 * np.pi * noise_freq3 * t) +
        np.random.normal(0, 0.1, len(t))
    )
    
    acc_y = acc_y_base + high_freq_noise
    acc_x = np.random.normal(0, 0.5, len(t))
    acc_z = 9.81 + np.random.normal(0, 0.2, len(t))  # Gravity + noise
    
    gyro_x = np.random.normal(0, 2, len(t))
    gyro_y = np.random.normal(0, 2, len(t))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Time_ms': time_ms,
        'Encoder_Angle': encoder_angle,
        'Commanded_Angle': target_angle,
        'Acc_X': acc_x,
        'Acc_Y': acc_y,
        'Acc_Z': acc_z,
        'Gyro_X': gyro_x,
        'Gyro_Y': gyro_y,
        'Gyro_Z': gyro_z
    })
    
    # Save to CSV
    csv_file = group_dir / "demo_run_001.csv"
    data.to_csv(csv_file, index=False)
    
    print(f"  Generated {len(data)} data points")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Saved to: {csv_file}")
    print(f"  Added high-frequency noise at {noise_freq1}, {noise_freq2}, and {noise_freq3} Hz")
    
    return str(temp_dir)


def create_demo_config():
    """Create a temporary config file for the demo."""
    config_content = """
# Demo configuration for filter analysis
data:
  data_base_dir: "demo_data"
  inertia_groups:
    - {id: "demo_inertia", folder: "0.010kgm2", inertia: 0.0100}
  
  radius_accel: 0.2
  gyro_axis_for_ang_vel: 'Gyro_Z'
  accel_axis_for_torque: 'Acc_Y'
  
  # Filter parameters - these will be used for the analysis
  filter_cutoff_freq_hz: 20.0  # 20 Hz cutoff to remove the 25, 35, 60 Hz noise
  filter_order: 4
  
  batch_size: 32
  num_workers: 0  # Set to 0 for demo to avoid multiprocessing issues
  global_train_ratio: 0.7
  global_val_ratio: 0.15
  fallback_sampling_frequency: 100.0
"""
    
    config_file = Path("demo_config.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return str(config_file)


def run_demo():
    """Run the complete filter analysis demo."""
    print("=== Filter Effects Analysis Demo ===\n")
    
    try:
        # Step 1: Generate demo data
        data_dir = generate_demo_data()
        
        # Step 2: Create demo config
        config_file = create_demo_config()
        print(f"\nGenerated demo config: {config_file}")
        
        # Step 3: Run the filter analysis
        print(f"\nRunning filter analysis with 20 Hz cutoff frequency...")
        print("This will show the effect of filtering on accelerometer data with artificial high-frequency noise.\n")
        
        # Import and run the analysis
        from scripts.analyze_filter_effects import main as analyze_main
        import hydra
        from omegaconf import OmegaConf
        
        # Create configuration
        cfg = OmegaConf.create({
            'data': {
                'data_base_dir': data_dir,
                'inertia_groups': [
                    {'id': 'demo_inertia', 'folder': '0.010kgm2', 'inertia': 0.0100}
                ],
                'radius_accel': 0.2,
                'gyro_axis_for_ang_vel': 'Gyro_Z',
                'accel_axis_for_torque': 'Acc_Y',
                'filter_cutoff_freq_hz': 20.0,
                'filter_order': 4
            }
        })
        
        # Run analysis directly
        print("=== Butterworth Filter Effects Analysis ===")
        print(f"Configuration:")
        print(f"  Data directory: {cfg.data.data_base_dir}")
        print(f"  Filter cutoff frequency: {cfg.data.filter_cutoff_freq_hz} Hz")
        print(f"  Filter order: {cfg.data.filter_order}")
        print(f"  Accelerometer axis: {cfg.data.accel_axis_for_torque}")
        
        # Import analysis functions directly
        from scripts.analyze_filter_effects import load_dataset_pair, analyze_accelerometer_signal
        
        # Create output directory
        output_dir = Path("demo_filter_analysis_plots")
        output_dir.mkdir(exist_ok=True)
        print(f"  Output directory: {output_dir.resolve()}")
        
        # Clear old plots
        for old_plot in output_dir.glob("*.png"):
            old_plot.unlink()
        
        # Process the demo file
        csv_file = Path(data_dir) / "0.010kgm2" / "demo_run_001.csv"
        print(f"\nProcessing demo file: {csv_file.name}")
        
        # Load datasets with and without filtering
        dataset_unfiltered, dataset_filtered = load_dataset_pair(
            csv_file=str(csv_file),
            inertia=0.0100,
            radius_accel=0.2,
            gyro_axis='Gyro_Z',
            accel_axis='Acc_Y',
            filter_cutoff_freq_hz=20.0,
            filter_order=4
        )
        
        # Perform analysis
        analyze_accelerometer_signal(
            dataset_unfiltered=dataset_unfiltered,
            dataset_filtered=dataset_filtered,
            accel_axis='Acc_Y',
            output_dir=output_dir,
            file_label="demo_run_001"
        )
        
        print(f"\n=== Demo Complete ===")
        print(f"Generated plots in: {output_dir.resolve()}")
        print(f"\nThe plots show:")
        print(f"  - Raw accelerometer data with high-frequency noise (25, 35, 60 Hz)")
        print(f"  - Filtered data with 20 Hz cutoff (should remove most noise)")
        print(f"  - PSD plots showing frequency content before and after filtering")
        print(f"  - Difference plots showing what the filter removed")
        print(f"  - Same analysis for the derived torque signals")
        
        print(f"\nTo run the analysis tool on your own data:")
        print(f"  python scripts/analyze_filter_effects.py")
        print(f"  python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=30.0")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\nCleaning up demo files...")
        demo_files = ["demo_data", "demo_config.yaml"]
        for item in demo_files:
            if Path(item).exists():
                if Path(item).is_dir():
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"  Removed: {item}")


if __name__ == "__main__":
    run_demo()