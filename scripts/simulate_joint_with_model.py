#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.model import ActuatorModel # Ensure this path is correct for your project structure

def get_target_trajectory(t: float, amplitude: float, frequency: float) -> tuple[float, float]:
    """
    Calculates the target angle and angular velocity for a sinusoidal trajectory.
    """
    omega = 2 * np.pi * frequency
    target_angle = amplitude * np.sin(omega * t)
    target_angular_velocity = amplitude * omega * np.cos(omega * t)
    return target_angle, target_angular_velocity

def main(args):
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Model ---
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    # Load on CPU for this simulation script
    model = ActuatorModel.load_from_checkpoint(args.checkpoint_path, map_location='cpu')
    model.eval() # Set to evaluation mode
    model.cpu()  # Ensure model is on CPU
    print("Model loaded successfully.")

    # Extract necessary parameters from loaded model
    # Normalization stats are buffers on the model
    input_mean = model.input_mean.cpu()
    input_std = model.input_std.cpu()
    target_mean = model.target_mean.cpu()
    target_std = model.target_std.cpu()

    # Reshape normalization stats for broadcasting: (1, 1, features) or (1,)
    input_mean_reshaped = input_mean.view(1, 1, -1) if input_mean.ndim == 1 else input_mean
    input_std_reshaped = input_std.view(1, 1, -1) if input_std.ndim == 1 else input_std
    target_mean_squeezed = target_mean.squeeze()
    target_std_squeezed = target_std.squeeze()

    gru_num_layers = model.hparams.gru_num_layers
    gru_hidden_dim = model.hparams.gru_hidden_dim

    # --- 2. Simulation Setup ---
    print("Setting up simulation...")
    time_steps = np.arange(0, args.simulation_duration_s, args.dt_s)
    
    current_angle_rad = args.initial_angle_rad
    current_ang_vel_rad_s = args.initial_angular_velocity_rad_s
    
    # Initialize hidden state for GRU (batch_size=1 for single simulation instance)
    # Shape: (num_layers, batch_size, hidden_dim)
    hidden_state = torch.zeros(gru_num_layers, 1, gru_hidden_dim, device='cpu')

    log_data = []

    # --- 3. Simulation Loop ---
    print("Starting simulation loop...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for t in time_steps:
            # Get target state
            target_angle_rad, target_ang_vel_rad_s = get_target_trajectory(
                t, args.target_amplitude_rad, args.target_frequency_hz
            )

            # Prepare model input features: [current_angle, target_angle, current_ang_vel]
            # Shape: (batch_size=1, seq_len=1, num_features=3)
            raw_input_features = torch.tensor(
                [[[current_angle_rad, target_angle_rad, current_ang_vel_rad_s]]],
                dtype=torch.float32, device='cpu'
            )

            # Normalize input features
            normalized_input_features = (raw_input_features - input_mean_reshaped) / (input_std_reshaped + 1e-7)

            # Model prediction
            # model.forward returns (prediction, h_next)
            # prediction is normalized, shape (1,1) for output_dim=1
            model_output_normalized, hidden_state = model(normalized_input_features, hidden_state)
            model_output_normalized = model_output_normalized.squeeze() # Shape (1,) or scalar

            # Handle residual prediction and denormalization
            predicted_torque_final_normalized = model_output_normalized # Direct model output initially

            if model.use_residual:
                # _calculate_tau_phys expects raw (unnormalized) inputs
                # x_sequence shape: [batch_size, sequence_length, features_per_step]
                tau_phys_raw = model._calculate_tau_phys(raw_input_features).squeeze() # Shape (1,) or scalar
                
                # Normalize tau_phys_raw
                tau_phys_normalized = (tau_phys_raw - target_mean_squeezed) / (target_std_squeezed + 1e-7)
                
                predicted_torque_final_normalized = model_output_normalized + tau_phys_normalized

            # Denormalize final prediction to physical scale
            final_predicted_torque_physical = predicted_torque_final_normalized * (target_std_squeezed + 1e-7) + target_mean_squeezed
            final_predicted_torque_physical = final_predicted_torque_physical.item() # Convert to scalar

            # Simulate joint dynamics
            angular_acceleration = final_predicted_torque_physical / args.inertia
            current_ang_vel_rad_s += angular_acceleration * args.dt_s
            current_angle_rad += current_ang_vel_rad_s * args.dt_s
            
            # Log data
            log_data.append({
                'time_s': t,
                'target_angle_rad': target_angle_rad,
                'actual_angle_rad': current_angle_rad,
                'target_ang_vel_rad_s': target_ang_vel_rad_s,
                'actual_ang_vel_rad_s': current_ang_vel_rad_s,
                'model_torque_nm': final_predicted_torque_physical,
                'angular_acceleration_rad_s2': angular_acceleration
            })
    print("Simulation loop finished.")

    # --- 4. Post-processing and Plotting ---
    results_df = pd.DataFrame(log_data)
    
    print("Generating plots...")
    # Plot 1: Target Angle vs. Actual Angle
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['time_s'], results_df['target_angle_rad'], label='Target Angle (rad)', linestyle='--')
    plt.plot(results_df['time_s'], results_df['actual_angle_rad'], label='Actual Angle (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Joint Angle Tracking (Inertia: {args.inertia} kg*m^2)')
    plt.legend()
    plt.grid(True)
    angle_plot_path = output_dir / 'angle_tracking_plot.png'
    plt.savefig(angle_plot_path)
    plt.close()
    print(f"Angle tracking plot saved to {angle_plot_path}")

    # Plot 2: Model Torque Output
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['time_s'], results_df['model_torque_nm'], label='Model Torque (Nm)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title(f'GRU Model Torque Output (Inertia: {args.inertia} kg*m^2)')
    plt.legend()
    plt.grid(True)
    torque_plot_path = output_dir / 'model_torque_plot.png'
    plt.savefig(torque_plot_path)
    plt.close()
    print(f"Model torque plot saved to {torque_plot_path}")

    # Optional: Save results to CSV
    results_csv_path = output_dir / 'simulation_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"Simulation results saved to {results_csv_path}")

    print("Script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a joint controlled by a trained GRU ActuatorModel.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint (.ckpt file).")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia of the simulated joint (kg*m^2).")
    parser.add_argument("--simulation_duration_s", type=float, default=10.0, help="Duration of the simulation (seconds).")
    parser.add_argument("--dt_s", type=float, default=0.01, help="Simulation time step (seconds).")
    parser.add_argument("--target_amplitude_rad", type=float, default=np.pi/4, help="Amplitude of the sinusoidal target angle (radians).")
    parser.add_argument("--target_frequency_hz", type=float, default=0.5, help="Frequency of the sinusoidal target angle (Hz).")
    parser.add_argument("--initial_angle_rad", type=float, default=0.0, help="Initial angle of the joint (radians).")
    parser.add_argument("--initial_angular_velocity_rad_s", type=float, default=0.0, help="Initial angular velocity of the joint (rad/s).")
    parser.add_argument("--output_dir", type=str, default="simulation_output", help="Directory to save plots and results.")
    
    args = parser.parse_args()
    main(args) 