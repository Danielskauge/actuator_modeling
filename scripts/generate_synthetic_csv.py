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
    kp_true: float = 10.0
    kd_true: float = 0.2
    stall_torque_nm: float = 2.0

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
    print(f"Generating file {file_idx+1}: Inertia={inertia_val:.4f}, Amp={amp_traj:.2f}, Freq={freq_traj:.2f}")

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
        
        # Add noise to the PD commanded torque (simulating noise in the motor command itself)
        torque_noise_sample = np.random.normal(0, config.noise_torque_std)
        noisy_pd_command_component = tau_pd_command_component + torque_noise_sample

        # Apply stall torque saturation
        saturated_pd_command_component = np.clip(noisy_pd_command_component, -config.stall_torque_nm, config.stall_torque_nm)
        
        # Net torque causing acceleration of the inertia
        # The motor attempts to apply saturated_pd_command_component. The spring resists this.
        net_torque_for_acceleration = saturated_pd_command_component - true_spring_force

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

    # Create inertia-specific subdirectory
    # Format inertia value for folder name, e.g., 0.01 -> "0.010kgm2"
    inertia_folder_name = f"{inertia_val:.3f}kgm2" 
    # Sanitize folder name if needed, though f-string format should be fine for typical values.
    # Example: inertia_folder_name = inertia_folder_name.replace('.', '_') if issues arise.

    group_output_dir = os.path.join(config.output_dir, inertia_folder_name)
    os.makedirs(group_output_dir, exist_ok=True)

    output_filename = f"{config.base_filename}_{file_idx}.csv"
    full_output_path = os.path.join(group_output_dir, output_filename)
    df.to_csv(full_output_path, index=False, float_format='%.6f')
    return full_output_path

def main():
    cfg = SyntheticDataConfig()
    # Top-level output directory (e.g., data/synthetic_raw) is created here,
    # subdirs for each inertia will be created by generate_single_csv
    os.makedirs(cfg.output_dir, exist_ok=True) 
    print(f"Generating synthetic data. Output directory: {cfg.output_dir}")

    generated_files_metadata = [] # Store dicts with path, inertia, and folder name
    for i in range(cfg.num_files_to_generate):
        # Vary inertia for each file group.
        # The script currently generates distinct files each with potentially different inertias.
        # For the LOMO CV structure, each of these files will effectively become its own "group"
        # unless multiple files are generated *for the same inertia* into the same folder.
        # The current logic varies inertia per file, so each file gets its own inertia folder.
        current_inertia = cfg.base_inertia * (1 + i * 0.5) # 0.01, 0.015, 0.02 for num_files=3
        current_amp = cfg.base_amp_target_rad * (1 - i * 0.1)
        current_freq = cfg.base_freq_target_hz * (1 + i * 0.2)

        file_path = generate_single_csv(cfg, i, current_inertia, current_amp, current_freq)
        
        inertia_folder_name = f"{current_inertia:.3f}kgm2"
        generated_files_metadata.append({
            "csv_file_path": file_path, 
            "inertia": current_inertia,
            "folder": inertia_folder_name 
        })
    
    print("\nSynthetic data generation complete.")
    print("Generated file configs (for data.dataset_configs in Hydra):")
    # Create a unique list of group configs for Hydra
    # This assumes each generated file (with its unique inertia) forms a group
    hydra_group_configs = []
    seen_folders = set()
    for item in generated_files_metadata:
        if item["folder"] not in seen_folders:
            # The relative path for csv_file_path in Hydra config is usually just the folder.
            # The datamodule will then look for all CSVs in that folder.
            # However, the current setup is one file per folder.
            # For Hydra, we list the "groups" (folders)
            hydra_group_configs.append(
                f"  - {{id: \"inertia_{item['folder'].replace('kgm2','').replace('.','_')}\", folder: \"{item['folder']}\", inertia: {item['inertia']:.4f}}}"
            )
            seen_folders.add(item['folder'])
            
    for cfg_line in hydra_group_configs:
        print(cfg_line)

if __name__ == "__main__":
    main() 