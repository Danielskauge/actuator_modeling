import numpy as np
import pandas as pd
import os
import math

# Constants from ActuatorDataset (for consistency if used elsewhere, though not strictly needed here for generation formula)
RAD_PER_DEG = np.pi / 180.0
G_ACCEL = 9.81  # m/s^2

# --- Simulation Parameters for Synthetic Data Generation ---
class SyntheticDataConfig:
    # File output
    output_dir: str = "data/synthetic_raw" # Subdirectory for generated CSVs
    num_files_to_generate: int = 3
    base_filename: str = "synthetic_actuator_data_mass"

    # System properties (can be varied per file)
    base_inertia: float = 0.01  # kg*m^2 (will be varied)
    radius_for_acc_y: float = 0.2 # meters (distance from pivot to Acc_Y sensor, for Acc_Y generation)
    
    # PD Controller for generating behavior (these are the *true* underlying parameters)
    kp_true: float = 6.0
    kd_true: float = 0.2

    # Parallel Spring for generating behavior (true underlying parameters)
    k_spring_true: float = 0.5 # Nm/rad
    theta0_spring_true: float = 0.0 # Equilibrium angle in radians

    # Simulation settings
    dt: float = 0.01  # Simulation time step [s] (100 Hz)
    t_sim_per_file: float = 20.0  # Total simulation time per file [s]
    noise_torque_std: float = 0.05 # Standard deviation of noise added to PD torque command [Nm]
    noise_gyro_std_rad_s: float = 0.01 # Noise on GyroZ readings (rad/s)
    noise_accel_std_m_s2: float = 0.1 # Noise on Acc_Y readings (m/s^2)
    noise_encoder_std_rad: float = 0.001 # Noise on Encoder_Angle readings (rad)

    # Target Trajectory parameters (can be varied per file)
    base_amp_target_rad: float = np.pi / 4  # Amplitude of target angle [rad]
    base_freq_target_hz: float = 0.5     # Frequency of target angle [Hz]

# Helper for target trajectory
def get_target_trajectory(t: float, amp: float, freq: float) -> tuple[float, float, float]:
    omega = 2 * np.pi * freq
    target_angle = amp * np.sin(omega * t)
    target_ang_vel = amp * omega * np.cos(omega * t)
    target_ang_accel = -amp * omega**2 * np.sin(omega * t)
    return target_angle, target_ang_vel, target_ang_accel

def generate_single_csv(config: SyntheticDataConfig, file_idx: int, inertia_val: float, amp_traj: float, freq_traj: float) -> str:
    """Generates one CSV file of synthetic actuator data."""
    print(f"Generating file {file_idx+1}: Inertia={inertia_val:.3f}, Amp={amp_traj:.2f}, Freq={freq_traj:.2f}")

    time_vec = np.arange(0, config.t_sim_per_file, config.dt)
    
    current_angle_rad_actual = 0.0
    current_ang_vel_rad_s_actual = 0.0
    log = []

    for t_current in time_vec:
        target_angle_rad, target_ang_vel_rad_s, _ = get_target_trajectory(t_current, amp_traj, freq_traj)

        error = target_angle_rad - current_angle_rad_actual
        error_dot = target_ang_vel_rad_s - current_ang_vel_rad_s_actual
        # PD part of the true command
        tau_pd_command_component = config.kp_true * error + config.kd_true * error_dot
        
        # True spring force acting on the system
        true_spring_force = config.k_spring_true * (current_angle_rad_actual - config.theta0_spring_true)
        
        # Ideal net torque from the controller side (PD command meant to counteract/overcome spring and drive motion)
        # If the goal is for the PD to command the *net* torque, then it must account for the spring.
        # Let's assume the PD calculates a desired torque, and the spring is an inherent part of the plant dynamics.
        # So, the torque the motor *tries* to apply (from PD) is tau_pd_command_component.
        # The actual net torque causing acceleration is this PD torque MINUS the spring force.
        
        # Add noise to the PD commanded torque (simulating noise in the motor command itself)
        torque_noise_sample = np.random.normal(0, config.noise_torque_std)
        noisy_pd_command_component = tau_pd_command_component + torque_noise_sample
        
        # Net torque causing acceleration of the inertia
        net_torque_for_acceleration = noisy_pd_command_component - true_spring_force

        current_ang_accel_rad_s2_actual = net_torque_for_acceleration / inertia_val

        # Simulate noisy sensor readings
        noisy_encoder_angle_rad = current_angle_rad_actual + np.random.normal(0, config.noise_encoder_std_rad)
        noisy_encoder_angle_deg = noisy_encoder_angle_rad / RAD_PER_DEG
        commanded_angle_deg = target_angle_rad / RAD_PER_DEG

        noisy_gyro_z_rad_s = current_ang_vel_rad_s_actual + np.random.normal(0, config.noise_gyro_std_rad_s)
        noisy_gyro_z_deg_s = noisy_gyro_z_rad_s / RAD_PER_DEG

        actual_tangential_accel_y = current_ang_accel_rad_s2_actual * config.radius_for_acc_y
        noisy_acc_y = actual_tangential_accel_y + np.random.normal(0, config.noise_accel_std_m_s2)
        
        noisy_acc_x = np.random.normal(0, config.noise_accel_std_m_s2 * 2) 
        noisy_acc_z = G_ACCEL + np.random.normal(0, config.noise_accel_std_m_s2 * 2)

        log.append({
            'Time_ms': t_current * 1000.0,
            'Encoder_Angle': noisy_encoder_angle_deg,
            'Commanded_Angle': commanded_angle_deg,
            'Acc_X': noisy_acc_x,
            'Acc_Y': noisy_acc_y,
            'Acc_Z': noisy_acc_z,
            'Gyro_X': np.random.normal(0, config.noise_gyro_std_rad_s * 10) / RAD_PER_DEG,
            'Gyro_Y': np.random.normal(0, config.noise_gyro_std_rad_s * 10) / RAD_PER_DEG,
            'Gyro_Z': noisy_gyro_z_deg_s,
        })
        
        current_angle_rad_actual += current_ang_vel_rad_s_actual * config.dt
        current_ang_vel_rad_s_actual += current_ang_accel_rad_s2_actual * config.dt
            
    df = pd.DataFrame(log)
    output_filename = f"{config.base_filename}_{file_idx}.csv"
    full_output_path = os.path.join(config.output_dir, output_filename)
    df.to_csv(full_output_path, index=False, float_format='%.6f')
    return full_output_path

def main():
    cfg = SyntheticDataConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Generating synthetic data. Output directory: {cfg.output_dir}")

    generated_files = []
    for i in range(cfg.num_files_to_generate):
        current_inertia = cfg.base_inertia * (1 + i * 0.5)
        current_amp = cfg.base_amp_target_rad * (1 - i * 0.1)
        current_freq = cfg.base_freq_target_hz * (1 + i * 0.2)

        file_path = generate_single_csv(cfg, i, current_inertia, current_amp, current_freq)
        generated_files.append({"csv_file_path": file_path, "inertia": current_inertia})
    
    print("\nSynthetic data generation complete.")
    print("Generated file configs (for data.dataset_configs in Hydra):")
    for item in generated_files:
        relative_path = os.path.relpath(item["csv_file_path"], start=os.getcwd())
        print(f"  - {{csv_file_path: \"{relative_path}\", inertia: {item['inertia']:.4f}}}")

if __name__ == "__main__":
    main() 