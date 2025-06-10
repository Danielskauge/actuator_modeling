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
    output_dir: str = "data/synthetic" # Subdirectory for generated CSVs
    num_different_inertias: int = 1  # Just one inertia group
    base_filename: str = "synthetic_actuator_data_mass"

    # System properties (can be varied per file)
    base_inertia: float = 0.1  # kg*m^2 (will be varied)
    radius_for_acc_y: float = 0.2 # meters (distance from pivot to Acc_Y sensor, for Acc_Y generation)
    
    # PD Controller for generating behavior (these are the *true* underlying parameters)
    kp_true: float = 10.0
    kd_true: float = 1.0
    stall_torque_nm: float = 3.6  # Match real system limits

    # Parallel Spring for generating behavior (true underlying parameters)
    k_spring_true: float = 0.0 # Nm/rad
    theta0_spring_true: float = 0.0 # Equilibrium angle in radians
    # Advanced motor characteristics
    include_spring: bool = False  # Whether to include spring torque
    include_torque_speed_curve: bool = False  # Whether to include torque-speed curve
    no_load_speed_hz: float = 100.0  # No-load speed for torque-speed curve [Hz]

    # Simulation settings
    dt: float = 1/240.0  # Simulation time step [s] (240 Hz to match Isaac Sim)
    t_sim_per_file: float = 60.0  # Back to 60 seconds per file
    noise_torque_std: float = 0.01 # Standard deviation of noise added to PD torque command [Nm]
    noise_gyro_std_rad_s: float = 0.01 # Noise on GyroZ readings (rad/s)
    noise_accel_std_m_s2: float = 0.01 # Noise on Acc_Y readings (m/s^2)
    noise_encoder_std_rad: float = 0.005 # Noise o n Encoder_Angle readings (rad)

    # Target Trajectory parameters (chirp characteristics will be varied per file)
    base_amp_target_rad: float = np.pi / 6  # Base amplitude of target angle [rad]
    base_freq_target_hz: float = 1.0     # Base frequency of target angle [Hz]

# Helper for target trajectory
def get_target_trajectory(t: float, amp_base: float, freq_base: float, t_total: float) -> tuple[float, float, float]:
    """Generate chirp signal with continuously changing frequency and amplitude."""
    # Frequency chirp: linearly increase from freq_base/2 to freq_base*2 over time
    freq_min = freq_base * 0.3
    freq_max = freq_base * 2.0
    freq_rate = (freq_max - freq_min) / t_total
    instantaneous_freq = freq_min + freq_rate * t
    
    # Amplitude modulation: vary between 0.5*amp_base and 1.5*amp_base with slower cycle
    amp_mod_freq = 0.1  # Hz, slow amplitude modulation
    amp_variation = 0.5 * amp_base * np.sin(2 * np.pi * amp_mod_freq * t)
    instantaneous_amp = amp_base + amp_variation
    
    # Phase accumulation for chirp
    phase = 2 * np.pi * (freq_min * t + 0.5 * freq_rate * t**2)
    
    target_angle = instantaneous_amp * np.sin(phase)
    target_ang_vel = instantaneous_amp * instantaneous_freq * 2 * np.pi * np.cos(phase)
    target_ang_accel = -instantaneous_amp * (instantaneous_freq * 2 * np.pi)**2 * np.sin(phase)
    
    return target_angle, target_ang_vel, target_ang_accel

def generate_single_csv(config: SyntheticDataConfig, file_idx: int, inertia_val: float, amp_traj: float, freq_traj: float) -> str:
    """Generates one CSV file of synthetic actuator data."""
    print(f"Generating file {file_idx+1}: Inertia={inertia_val:.4f}, Amp={amp_traj:.2f}, Freq={freq_traj:.2f}")

    time_vec = np.arange(0, config.t_sim_per_file, config.dt)
    
    current_angle_rad_actual = 0.0
    current_ang_vel_rad_s_actual = 0.0
    log = []

    for t_current in time_vec:
        target_angle_rad, target_ang_vel_rad_s, _ = get_target_trajectory(t_current, amp_traj, freq_traj, config.t_sim_per_file)

        error = target_angle_rad - current_angle_rad_actual
        error_dot = target_ang_vel_rad_s - current_ang_vel_rad_s_actual
        # PD part of the true command
        tau_pd_command_component = config.kp_true * error + config.kd_true * error_dot
        
        # True spring force acting on the system
        if config.include_spring:
            true_spring_force = config.k_spring_true * (current_angle_rad_actual - config.theta0_spring_true)
        else:
            true_spring_force = 0.0
        
        # Add noise to the PD commanded torque (simulating noise in the motor command itself)
        torque_noise_sample = np.random.normal(0, config.noise_torque_std)
        noisy_pd_command_component = tau_pd_command_component + torque_noise_sample

        # Apply stall torque saturation
        if config.include_torque_speed_curve:
            no_load_speed_rad_s = config.no_load_speed_hz * 2 * math.pi
            max_torque = config.stall_torque_nm * max(0.0, 1 - abs(current_ang_vel_rad_s_actual) / no_load_speed_rad_s)
        else:
            max_torque = config.stall_torque_nm
        saturated_pd_command_component = np.clip(noisy_pd_command_component, -max_torque, max_torque)
        
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
    
    # Generate multiple files per inertia value for better training data variety
    files_per_inertia = 1  # Just one file per inertia
    base_inertia = cfg.base_inertia
    
    for i in range(cfg.num_different_inertias):
        current_inertia = base_inertia * (1 + i * 0.2)  # 0.1, 0.12, 0.14 kg*m^2
        inertia_folder_name = f"{current_inertia:.3f}kgm2"
        
        for file_idx in range(files_per_inertia):
            # Vary chirp characteristics for each file
            amp_variation = 1.0 + (file_idx - 2) * 0.2  # 0.6, 0.8, 1.0, 1.2, 1.4
            freq_variation = 1.0 + (file_idx - 2) * 0.3  # 0.4, 0.7, 1.0, 1.3, 1.6
            
            current_amp = cfg.base_amp_target_rad * amp_variation
            current_freq = cfg.base_freq_target_hz * freq_variation
            
            global_file_idx = i * files_per_inertia + file_idx
            
            file_path = generate_single_csv(cfg, global_file_idx, current_inertia, current_amp, current_freq)
            
            generated_files_metadata.append({
                "csv_file_path": file_path, 
                "inertia": current_inertia,
                "folder": inertia_folder_name 
            })
    
    print("\nSynthetic data generation complete.")
    print("Generated file configs (for data.inertia_groups in Hydra):")
    # Create a unique list of group configs for Hydra
    # This assumes each generated file (with its unique inertia) forms a group
    hydra_group_configs = []
    seen_folders = set()
    for item in generated_files_metadata:
        if item["folder"] not in seen_folders:
            # The relative path for csv_file_path in Hydra config is usually just the folder.
            # The datamodule will then look for all CSVs in that folder.
            hydra_group_configs.append(
                f"  - {{id: \"inertia_{item['folder'].replace('kgm2','').replace('.','_')}\", folder: \"{item['folder']}\"}}"
            )
            seen_folders.add(item['folder'])
            
    for cfg_line in hydra_group_configs:
        print(cfg_line)

if __name__ == "__main__":
    main() 