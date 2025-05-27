#!/usr/bin/env python3
"""
Generates synthetic data for pd_estimation.py and tests its parameter recovery.
Includes option for a torque-speed limited actuator model.
"""

import numpy as np
import pandas as pd
import subprocess
import pathlib
import re
import sys
import math # For math.isclose

# --- Simulation Parameters ---
KP_TRUE = 6.0  # Nm/rad
KD_TRUE = 0.2   # Nm·s/rad
INERTIA_TRUE = 0.1  # kg·m²
DT = 0.001  # Simulation time step [s]
T_SIM = 10.0  # Total simulation time [s]
NOISE_TORQUE_STD = 0.1 # Standard deviation of torque noise [Nm]

# Target trajectory parameters
AMP_TARGET = np.pi / 4  # Amplitude of target angle [rad]
FREQ_TARGET = 0.5  # Frequency of target angle [Hz]

# --- Actual Actuator Torque-Speed Limit Parameters (for data generation) ---
# To disable torque-speed limit during generation, set USE_TORQUE_SPEED_LIMIT_GENERATION to False
# or set NO_LOAD_SPEED_ACTUAL to 0 or STALL_TORQUE_ACTUAL to 0
USE_TORQUE_SPEED_LIMIT_GENERATION = True  # Set to True to enable TS limit in synthetic data
STALL_TORQUE_ACTUAL = 1.81  # Stall torque [Nm]
NO_LOAD_SPEED_ACTUAL = 35.0 # No-load speed [rad/s] (Ensure > 0 for limit to apply)

# --- Helper function for target trajectory ---
def get_target_trajectory(t: float, amp: float, freq: float) -> tuple[float, float]:
    """Calculates target angle and angular velocity for a given time."""
    omega = 2 * np.pi * freq
    target_angle = amp * np.sin(omega * t)
    target_ang_vel = amp * omega * np.cos(omega * t)
    return target_angle, target_ang_vel

# --- Main simulation function for generating initial data ---
def generate_synthetic_data(
    kp_gen: float, 
    kd_gen: float, 
    inertia_gen: float,
    amp_traj: float,
    freq_traj_initial: float, # Renamed to indicate it might be adjusted
    t_sim_gen: float, 
    dt_gen: float,
    noise_std_gen: float,
    use_ts_limit: bool,
    stall_torque_sys: float,
    no_load_speed_sys: float
) -> pd.DataFrame:
    """Generates synthetic actuator data."""
    
    freq_traj = freq_traj_initial
    if use_ts_limit and no_load_speed_sys > 0:
        max_target_ang_vel_allowed = 0.3 * no_load_speed_sys
        current_max_target_ang_vel = amp_traj * 2 * np.pi * freq_traj
        if current_max_target_ang_vel > max_target_ang_vel_allowed:
            original_freq = freq_traj
            freq_traj = max_target_ang_vel_allowed / (amp_traj * 2 * np.pi)
            if freq_traj <= 0: # Should not happen if no_load_speed_sys is reasonable
                print(f"Warning: Calculated target frequency {freq_traj:.2f} Hz is too low or zero. Using 0.1 Hz instead.")
                freq_traj = 0.1
            print(f"Warning: Target trajectory's max angular velocity ({current_max_target_ang_vel:.2f} rad/s) "
                  f"exceeds 30% of no-load speed ({max_target_ang_vel_allowed:.2f} rad/s). "
                  f"Reducing target frequency from {original_freq:.2f} Hz to {freq_traj:.2f} Hz.")
    elif use_ts_limit and no_load_speed_sys <= 0 and stall_torque_sys > 0:
        print("Warning: Torque-speed limit enabled but no_load_speed_sys is <= 0. Limit will not be effective.")


    time_vec = np.arange(0, t_sim_gen, dt_gen)
    
    angle = 0.0
    ang_vel = 0.0
    data_log = []

    for t_current in time_vec:
        target_angle_current, target_ang_vel_current = get_target_trajectory(t_current, amp_traj, freq_traj)

        error = target_angle_current - angle
        error_dot = target_ang_vel_current - ang_vel
        tau_pd = kp_gen * error + kd_gen * error_dot
        torque_noise = np.random.normal(0, noise_std_gen)
        tau_command = tau_pd + torque_noise # Torque before any system limits

        tau_applied = tau_command
        if use_ts_limit and no_load_speed_sys > 0 and stall_torque_sys > 0:
            # Asymmetric torque-speed curve
            if tau_command * ang_vel > 0: # Motor driving: torque and velocity in the same direction
                max_limit_val = stall_torque_sys * max(0.0, (1.0 - abs(ang_vel) / no_load_speed_sys))
            else: # Motor braking/holding: torque opposes velocity, or ang_vel is zero
                max_limit_val = stall_torque_sys
            
            if abs(tau_command) > max_limit_val:
                tau_applied = math.copysign(max_limit_val, tau_command)
            # else tau_applied remains tau_command
        
        current_ang_accel = tau_applied / inertia_gen
        
        data_log.append({
            'time': t_current,
            'target_angle': target_angle_current,
            'angle': angle,
            'ang_vel': ang_vel,
            'ang_accel': current_ang_accel, # This is a_applied = tau_applied / I
            'target_ang_vel': target_ang_vel_current,
            'tau_applied_actual': tau_applied # Log the actual torque applied
        })
        
        angle += ang_vel * dt_gen + 0.5 * current_ang_accel * dt_gen**2
        ang_vel += current_ang_accel * dt_gen
            
    return pd.DataFrame(data_log)

# --- Function to run estimation and parse results ---
def run_pd_estimation(csv_path: pathlib.Path, inertia_val: float) -> tuple[float, float, dict]:
    """Runs pd_estimation.py and parses its Kp, Kd output and metrics."""
    script_dir = pathlib.Path(__file__).parent.resolve()
    estimator_script_path = script_dir / "pd_estimation.py"
    
    cmd = [
        sys.executable,
        str(estimator_script_path),
        str(csv_path),
        str(inertia_val)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=script_dir)
    output = result.stdout
    
    print("\n--- pd_estimation.py output ---")
    print(output.strip())
    print("-----------------------------")

    kp_match = re.search(r"Kp\s*=\s*([-\d.]+)", output)
    kd_match = re.search(r"Kd\s*=\s*([-\d.]+)", output)
    
    metrics_parsed = {}
    r2_match = re.search(r"R2\s*=\s*([-\d.]+)", output)
    rmse_match = re.search(r"RMSE\s*=\s*([-\d.]+)", output)
    mae_match = re.search(r"MAE\s*=\s*([-\d.]+)", output)
    if r2_match: metrics_parsed['R2_estimation'] = float(r2_match.group(1))
    if rmse_match: metrics_parsed['RMSE_estimation_torque'] = float(rmse_match.group(1))
    if mae_match: metrics_parsed['MAE_estimation_torque'] = float(mae_match.group(1))

    if not kp_match or not kd_match:
        raise ValueError(f"Could not parse Kp/Kd from pd_estimation.py output:\n{output}")
        
    kp_est = float(kp_match.group(1))
    kd_est = float(kd_match.group(1))
    
    return kp_est, kd_est, metrics_parsed

# --- Function to evaluate performance of estimated PD parameters ---
def evaluate_pd_performance(
    original_synthetic_df: pd.DataFrame,
    kp_eval: float,
    kd_eval: float,
    inertia_eval: float,
    dt_eval: float,
    apply_ts_limit_to_controller_output: bool, # New flag for this eval
    stall_torque_eval_limit: float,  # Stall torque for limiting the evaluated controller
    no_load_speed_eval_limit: float # No-load speed for limiting the evaluated controller
) -> pd.DataFrame:
    """
    Simulates the system using the evaluated Kp, Kd and compares to original synthetic data.
    The original_synthetic_df contains the target trajectory and the 'true' noisy, saturated motion.
    This function does NOT add new random noise. It evaluates the deterministic PD controller
    (optionally with its own output saturation) against the already noisy/saturated ground truth.
    """
    
    time_vec = original_synthetic_df['time'].values
    angle_sim = original_synthetic_df['angle'].iloc[0]
    ang_vel_sim = original_synthetic_df['ang_vel'].iloc[0]
    
    sim_results_log = []

    for i, t_current in enumerate(time_vec):
        target_angle_current = original_synthetic_df['target_angle'].iloc[i]
        target_ang_vel_current = original_synthetic_df['target_ang_vel'].iloc[i]

        error_sim = target_angle_current - angle_sim
        error_dot_sim = target_ang_vel_current - ang_vel_sim
        tau_pd_controller_output = kp_eval * error_sim + kd_eval * error_dot_sim

        tau_applied_by_controller = tau_pd_controller_output
        if apply_ts_limit_to_controller_output and no_load_speed_eval_limit > 0 and stall_torque_eval_limit > 0:
            # Apply the same asymmetric torque-speed curve logic
            current_ang_vel_sim = ang_vel_sim # ang_vel of the controller's own simulation
            if tau_pd_controller_output * current_ang_vel_sim > 0: # Motor driving
                max_limit_val_controller = stall_torque_eval_limit * max(0.0, (1.0 - abs(current_ang_vel_sim) / no_load_speed_eval_limit))
            else: # Motor braking/holding
                max_limit_val_controller = stall_torque_eval_limit
            
            if abs(tau_pd_controller_output) > max_limit_val_controller:
                tau_applied_by_controller = math.copysign(max_limit_val_controller, tau_pd_controller_output)
            # else tau_applied_by_controller remains tau_pd_controller_output
        
        current_ang_accel_sim = tau_applied_by_controller / inertia_eval
        
        sim_results_log.append({
            'time': t_current,
            'angle_sim': angle_sim,
            'ang_vel_sim': ang_vel_sim,
            'tau_applied_sim': tau_applied_by_controller
        })
        
        # Use the same integration as in generate_synthetic_data
        angle_sim += ang_vel_sim * dt_eval + 0.5 * current_ang_accel_sim * dt_eval**2
        ang_vel_sim += current_ang_accel_sim * dt_eval
            
    return pd.DataFrame(sim_results_log)

def calculate_tracking_metrics(df_eval_results: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """Calculates RMSE and MAE for angle tracking."""
    # Ensure indices align if they are not default range indices
    df_eval_results = df_eval_results.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)
    
    # Align lengths if slightly different due to integration endpoint handling
    min_len = min(len(df_eval_results), len(original_df))
    
    angle_sim = df_eval_results['angle_sim'][:min_len].values
    angle_true_model = original_df['angle'][:min_len].values # From the initial rich simulation

    # tau_applied_sim = df_eval_results['tau_applied_sim'][:min_len].values
    # tau_applied_true_model = original_df['tau_applied_actual'][:min_len].values


    angle_rmse = np.sqrt(np.mean((angle_sim - angle_true_model)**2))
    angle_mae = np.mean(np.abs(angle_sim - angle_true_model))
    
    # Optional: Torque comparison (less direct for "performance" of estimated params)
    # tau_rmse = np.sqrt(np.mean((tau_applied_sim - tau_applied_true_model)**2))
    # tau_mae = np.mean(np.abs(tau_applied_sim - tau_applied_true_model))

    return {'angle_RMSE': angle_rmse, 'angle_MAE': angle_mae}


# --- Main execution ---
if __name__ == "__main__":
    # 1. Generate synthetic data
    print("Generating initial synthetic data...")
    print(f"  Using KP_TRUE={KP_TRUE}, KD_TRUE={KD_TRUE}, INERTIA_TRUE={INERTIA_TRUE}")
    if USE_TORQUE_SPEED_LIMIT_GENERATION:
        print(f"  Applying torque-speed limit during generation: STALL_TORQUE={STALL_TORQUE_ACTUAL} Nm, NO_LOAD_SPEED={NO_LOAD_SPEED_ACTUAL} rad/s")
    else:
        print("  NOT applying torque-speed limit during generation.")

    synthetic_df_actual_model = generate_synthetic_data(
        kp_gen=KP_TRUE,
        kd_gen=KD_TRUE,
        inertia_gen=INERTIA_TRUE,
        amp_traj=AMP_TARGET,
        freq_traj_initial=FREQ_TARGET,
        t_sim_gen=T_SIM,
        dt_gen=DT,
        noise_std_gen=NOISE_TORQUE_STD,
        use_ts_limit=USE_TORQUE_SPEED_LIMIT_GENERATION,
        stall_torque_sys=STALL_TORQUE_ACTUAL,
        no_load_speed_sys=NO_LOAD_SPEED_ACTUAL
    )
    
    script_location = pathlib.Path(__file__).parent.resolve()
    csv_filename = "synthetic_pd_log.csv"
    csv_full_path = script_location / csv_filename
    synthetic_df_actual_model.to_csv(csv_full_path, index=False, float_format='%.8f')
    print(f"Synthetic data saved to {csv_full_path}")

    # 2. Run pd_estimation.py on the generated data
    # The estimation script is unaware of any torque-speed limits used during generation.
    # It uses INERTIA_TRUE because that's the "assumed known" inertia for estimation.
    print(f"\nRunning PD estimation using INERTIA_TRUE = {INERTIA_TRUE} kg·m²...")
    try:
        kp_estimated, kd_estimated, estimation_metrics = run_pd_estimation(csv_full_path, INERTIA_TRUE)
        
        print("\n--- Comparison of True vs. Estimated PD Parameters (from pd_estimation.py) ---")
        print(f"True Kp: {KP_TRUE:>10.4f} [Nm/rad], Estimated Kp: {kp_estimated:>10.4f} [Nm/rad]")
        print(f"True Kd: {KD_TRUE:>10.4f} [Nm·s/rad], Estimated Kd: {kd_estimated:>10.4f} [Nm·s/rad]")

        error_kp = abs(KP_TRUE - kp_estimated)
        error_kd = abs(KD_TRUE - kd_estimated)
        # Handle division by zero if true params are zero
        rel_error_kp = (error_kp / abs(KP_TRUE)) if not math.isclose(KP_TRUE,0) else (0 if math.isclose(kp_estimated,0) else float('inf'))
        rel_error_kd = (error_kd / abs(KD_TRUE)) if not math.isclose(KD_TRUE,0) else (0 if math.isclose(kd_estimated,0) else float('inf'))

        print(f"Absolute Error Kp: {error_kp:.4f} (Relative: {rel_error_kp:.2%})")
        print(f"Absolute Error Kd: {error_kd:.4f} (Relative: {rel_error_kd:.2%})")
        if estimation_metrics:
            print(f"Estimation script's torque fit metrics: {estimation_metrics}")


        # 3. Evaluate performance of the *estimated* PD parameters
        print("\n--- Evaluating performance of ESTIMATED PD parameters against the original synthetic model ---")

        # Scenario 1: Estimated PD controller's output IS subject to the same TS limit as the true system
        print("\nScenario 1: Estimated PD controller's output IS SATURATED by actual system's TS curve.")
        eval_df_est_pd_with_ts_limit = evaluate_pd_performance(
            original_synthetic_df=synthetic_df_actual_model,
            kp_eval=kp_estimated,
            kd_eval=kd_estimated,
            inertia_eval=INERTIA_TRUE, # Use true inertia for this simulation
            dt_eval=DT,
            apply_ts_limit_to_controller_output=True,
            stall_torque_eval_limit=STALL_TORQUE_ACTUAL, # Use actual system's TS params
            no_load_speed_eval_limit=NO_LOAD_SPEED_ACTUAL
        )
        metrics_est_pd_with_ts_limit = calculate_tracking_metrics(eval_df_est_pd_with_ts_limit, synthetic_df_actual_model)
        print(f"  Tracking Performance (vs. actual synthetic model angle):")
        print(f"    Angle RMSE: {metrics_est_pd_with_ts_limit['angle_RMSE']:.6f} rad")
        print(f"    Angle MAE:  {metrics_est_pd_with_ts_limit['angle_MAE']:.6f} rad")

        # Scenario 2: Estimated PD controller's output is NOT subject to any TS limit
        print("\nScenario 2: Estimated PD controller's output is NOT SATURATED (ideal PD).")
        eval_df_est_pd_no_ts_limit = evaluate_pd_performance(
            original_synthetic_df=synthetic_df_actual_model,
            kp_eval=kp_estimated,
            kd_eval=kd_estimated,
            inertia_eval=INERTIA_TRUE,
            dt_eval=DT,
            apply_ts_limit_to_controller_output=False, # Key difference
            stall_torque_eval_limit=STALL_TORQUE_ACTUAL, # Not used if above is False
            no_load_speed_eval_limit=NO_LOAD_SPEED_ACTUAL # Not used if above is False
        )
        metrics_est_pd_no_ts_limit = calculate_tracking_metrics(eval_df_est_pd_no_ts_limit, synthetic_df_actual_model)
        print(f"  Tracking Performance (vs. actual synthetic model angle):")
        print(f"    Angle RMSE: {metrics_est_pd_no_ts_limit['angle_RMSE']:.6f} rad")
        print(f"    Angle MAE:  {metrics_est_pd_no_ts_limit['angle_MAE']:.6f} rad")
        
        # For reference: how well would the TRUE PD parameters track if their output was NOT TS limited?
        # (This helps understand the impact of the TS limit itself on the "best possible" PD)
        if USE_TORQUE_SPEED_LIMIT_GENERATION: # Only relevant if original data was limited
            print("\nFor Reference: How TRUE PD params would perform if their output were NOT TS limited (ideal PD on ideal plant)")
            eval_df_true_pd_no_ts_limit = evaluate_pd_performance(
                original_synthetic_df=synthetic_df_actual_model, # Target and initial state from here
                kp_eval=KP_TRUE,
                kd_eval=KD_TRUE,
                inertia_eval=INERTIA_TRUE,
                dt_eval=DT,
                apply_ts_limit_to_controller_output=False, 
                stall_torque_eval_limit=0, 
                no_load_speed_eval_limit=0
            )
            metrics_true_pd_no_ts_limit = calculate_tracking_metrics(eval_df_true_pd_no_ts_limit, synthetic_df_actual_model)
            print(f"  TRUE PD (No TS limit on its output) Tracking Performance (vs. actual limited synthetic model angle):")
            print(f"    Angle RMSE: {metrics_true_pd_no_ts_limit['angle_RMSE']:.6f} rad")
            print(f"    Angle MAE:  {metrics_true_pd_no_ts_limit['angle_MAE']:.6f} rad")


    except subprocess.CalledProcessError as e:
        print(f"Error running pd_estimation.py (return code {e.returncode}):")
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
    except ValueError as e:
        print(f"Error processing results: {e}")
    finally:
        # Clean up the generated CSV file
        if csv_full_path.exists():
            try:
                csv_full_path.unlink()
                print(f"\nCleaned up {csv_full_path}")
            except OSError as e_unlink:
                print(f"Error cleaning up {csv_full_path}: {e_unlink}")
        pass 