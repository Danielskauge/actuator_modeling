import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any

# Constants
RAD_PER_DEG = np.pi / 180.0
G_ACCEL = 9.81  # m/s^2

class ActuatorDataset(Dataset):
    """
    Dataset for actuator modeling.
    
    Processes raw timestamped sensor data from CSV files, performs feature engineering,
    and generates sequences suitable for recurrent neural networks.
    Input features: current_angle_rad, target_angle_rad, current_ang_vel_rad_s
    """
    
    FEATURE_NAMES = [
        'current_angle_rad',
        'target_angle_rad',
        'current_ang_vel_rad_s'
        # 'target_ang_vel_rad_s' # Removed as per user request
    ]
    INPUT_DIM = len(FEATURE_NAMES) # Should be 3 now
    SEQUENCE_LENGTH = 2

    def __init__(
        self,
        csv_file_path: str,
        inertia: float,
        radius_accel: float,
        gyro_axis_for_ang_vel: str = 'Gyro_Z',
        accel_axis_for_torque: str = 'Acc_Y',
        target_name: str = 'tau_measured'
    ):
        """
        Args:
            csv_file_path: Path to the CSV data file.
            inertia: Inertia of the system (kg*m^2).
            radius_accel: Radius to the point where tangential acceleration is measured by accel_axis_for_torque (meters).
            gyro_axis_for_ang_vel: Gyroscope axis ('Gyro_X', 'Gyro_Y', 'Gyro_Z') for signed angular velocity.
            accel_axis_for_torque: Accelerometer axis ('Acc_X', 'Acc_Y', 'Acc_Z') for tangential acceleration.
            target_name: Name of the target column to predict.
        """
        self.csv_file_path = csv_file_path
        self.inertia = inertia
        self.radius_accel = radius_accel
        self.gyro_axis_for_ang_vel = gyro_axis_for_ang_vel
        self.accel_axis_for_torque = accel_axis_for_torque
        self.target_name = target_name

        self.data_df = self._load_and_preprocess_data()
        self.X_sequences, self.y_sequences = self._create_sequences()

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Loads data from CSV and performs all preprocessing and feature engineering."""
        try:
            df = pd.read_csv(self.csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        # 1. Time handling: Convert Time_ms to seconds and set as index
        df['time_s'] = df['Time_ms'] / 1000.0
        df = df.set_index('time_s')
        df.index = pd.to_datetime(df.index, unit='s') # For proper time-based operations
        
        # Calculate dt (sampling period) - important for derivatives
        # Ensure time is sorted
        df = df.sort_index()
        dt_series = df.index.to_series().diff().median().total_seconds()
        if pd.isna(dt_series) or dt_series <= 0:
             # Fallback if diff().median() fails (e.g. too few data points)
            if len(df) > 1:
                dt_series = (df.index[-1] - df.index[0]).total_seconds() / (len(df) -1)
            else:
                dt_series = 0.01 # Default if only one point, though this case is problematic
            print(f"Warning: Could not reliably determine dt from time index for {self.csv_file_path}. Using {dt_series:.6f}s. Ensure data is uniformly sampled.")

        self.sampling_frequency = 1.0 / dt_series if dt_series > 0 else 100 # Default 100Hz if dt_series is 0
        
        # 2. Angle conversions: Degrees to Radians
        df['current_angle_rad'] = df['Encoder_Angle'] * RAD_PER_DEG
        df['target_angle_rad'] = df['Commanded_Angle'] * RAD_PER_DEG

        # 3. Angular Velocities
        # 3a. Current Angular Velocity (theta_dot) from Gyro
        gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
        for col in gyro_cols: # Convert gyro data from deg/s to rad/s
            df[col + '_rad_s'] = df[col] * RAD_PER_DEG
        
        if self.gyro_axis_for_ang_vel and self.gyro_axis_for_ang_vel + '_rad_s' in df.columns:
            df['current_ang_vel_rad_s'] = df[self.gyro_axis_for_ang_vel + '_rad_s']
            # print(f"Using '{self.gyro_axis_for_ang_vel}_rad_s' for current_ang_vel_rad_s.") # Less verbose
        else:
            # Fallback or if explicitly wanting magnitude (though sign is lost)
            # This is generally not recommended if a primary axis is known.
            df['current_ang_vel_rad_s_mag'] = np.sqrt(df['Gyro_X_rad_s']**2 + df['Gyro_Y_rad_s']**2 + df['Gyro_Z_rad_s']**2)
            df['current_ang_vel_rad_s'] = df['current_ang_vel_rad_s_mag'] # Default to magnitude if specific axis fails
            print(f"Warning: Using magnitude of gyro for current_ang_vel_rad_s. Sign information is lost. Gyro_axis specified: {self.gyro_axis_for_ang_vel}")


        # 3b. Target Angular Velocity (theta_d_dot) - REMOVED
        # df['target_ang_vel_rad_s'] = np.gradient(df['target_angle_rad'], dt_series, edge_order=2)

        # 4. Angular Acceleration (alpha) for tau_measured - numerical differentiation of current_ang_vel
        # This is used for calculating the target torque, not as a model input feature directly.
        df['current_ang_accel_rad_s2_for_target'] = np.gradient(df['current_ang_vel_rad_s'], dt_series, edge_order=2)
        
        # 5. Measured Torque (tau_measured) - Ground Truth
        # Calculation based on tangential accelerometer: tau = I * (a_tangential / r)
        if self.accel_axis_for_torque not in df.columns:
            raise ValueError(f"Specified accel_axis_for_torque '{self.accel_axis_for_torque}' not found. Columns: {df.columns.tolist()}")
        if self.radius_accel <= 0:
            raise ValueError("radius_accel must be positive.")
        
        # The acceleration used here is the one related to the chosen tangential axis, 
        # NOT the one derived from gyro (current_ang_accel_rad_s2_for_target)
        angular_accel_from_tangential_sensor = df[self.accel_axis_for_torque] / self.radius_accel
        df[self.target_name] = self.inertia * angular_accel_from_tangential_sensor


        # 6. Accelerometer Data Processing (Optional features, not part of core input currently)
        # Remove gravity from Z-axis. Acc_X, Acc_Y are kept as is.
        # This assumes Acc_Z is primarily aligned with gravity.
        # df['Acc_Z_corrected'] = df['Acc_Z'] - G_ACCEL 
        # df['Acc_X_raw'] = df['Acc_X'] 
        # df['Acc_Y_raw'] = df['Acc_Y']

        # Ensure all requested features are present
        missing_features = [f for f in self.FEATURE_NAMES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features after processing: {missing_features}. Available: {df.columns.tolist()}")

        # print(f"Successfully processed {self.csv_file_path}. Shape: {df.shape}. Input Dim: {self.INPUT_DIM}") # Less verbose
        return df[self.FEATURE_NAMES + [self.target_name]] # Select only needed columns

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates sequences of features and corresponding targets."""
        feature_data = self.data_df[self.FEATURE_NAMES].values
        target_data = self.data_df[self.target_name].values

        num_samples = len(feature_data) - self.SEQUENCE_LENGTH + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data points ({len(feature_data)}) to create sequences "
                f"of length {self.SEQUENCE_LENGTH} for file {self.csv_file_path}."
            )

        X = np.array([feature_data[i : i + self.SEQUENCE_LENGTH] for i in range(num_samples)])
        # The target corresponds to the state at the *end* of the sequence
        y = np.array([target_data[i + self.SEQUENCE_LENGTH - 1] for i in range(num_samples)])
        
        return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1) # Target shape: [num_samples, 1]

    def __len__(self) -> int:
        return len(self.X_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_sequences[idx], self.y_sequences[idx]

    def get_sampling_frequency(self) -> float:
        return self.sampling_frequency

    @staticmethod
    def get_input_dim() -> int:
        return ActuatorDataset.INPUT_DIM

    @staticmethod
    def get_sequence_length() -> int:
        return ActuatorDataset.SEQUENCE_LENGTH

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy CSV file for testing
    print("Creating dummy_actuator_data.csv for testing ActuatorDataset...")
    num_rows = 200
    fs_test = 100 # 100 Hz
    dt_test = 1.0 / fs_test
    time_ms_test = np.arange(0, num_rows * dt_test, dt_test) * 1000

    dummy_data = pd.DataFrame({
        'Time_ms': time_ms_test,
        'Encoder_Angle': 90 * np.sin(2 * np.pi * 0.5 * time_ms_test / 1000), # 0.5 Hz sine wave
        'Commanded_Angle': 90 * np.sin(2 * np.pi * 0.5 * time_ms_test / 1000 + np.pi/4), # Phase shifted
        'Acc_X': np.random.randn(num_rows) * 0.5,
        'Acc_Y': np.sin(2 * np.pi * 0.5 * time_ms_test / 1000) * 10, # Tangential accel mimic
        'Acc_Z': 9.81 + np.random.randn(num_rows) * 0.5, # Gravity + noise
        'Gyro_X': np.random.randn(num_rows) * 5,
        'Gyro_Y': np.random.randn(num_rows) * 5,
        'Gyro_Z': (90 * RAD_PER_DEG * 0.5 * 2 * np.pi) * np.cos(2 * np.pi * 0.5 * time_ms_test / 1000) / RAD_PER_DEG, # Gyro_Z in deg/s
        'ADC': np.random.randint(500, 600, num_rows)
    })
    dummy_csv_path = "dummy_actuator_data.csv"
    dummy_data.to_csv(dummy_csv_path, index=False)
    print(f"Dummy data saved to {dummy_csv_path}")

    # Define features required by the model, especially for tau_phys calculation
    # Order matters if _calculate_tau_phys in the LightningModule uses hardcoded indices.
    # current_angle_rad, target_angle_rad, current_ang_vel_rad_s, target_ang_vel_rad_s
    # then Acc_X_raw, Acc_Y_raw, Acc_Z_corrected
    model_input_features = [
        'current_angle_rad', 
        'target_angle_rad', 
        'current_ang_vel_rad_s', 
        'target_ang_vel_rad_s',
        'Acc_X_raw',
        'Acc_Y_raw',
        'Acc_Z_corrected'
    ]

    test_inertia = 0.05 # kg*m^2
    test_radius_accel = 0.2 # m
    test_gyro_axis = 'Gyro_Z'
    test_accel_axis = 'Acc_Y'
    # Sequence length should be fs * 1s. If fs=100Hz, then sequence_length = 100.
    # For this dummy data, dt is 0.01s, so fs = 100 Hz.
    test_sequence_length = int(fs_test * 1.0) # 1 second sequences

    try:
        dataset = ActuatorDataset(
            csv_file_path=dummy_csv_path,
            inertia=test_inertia,
            radius_accel=test_radius_accel,
            gyro_axis_for_ang_vel=test_gyro_axis,
            accel_axis_for_torque=test_accel_axis
        )
        print(f"Dataset created. Number of sequences: {len(dataset)}")
        X_sample, y_sample = dataset[0]
        print(f"Sample X shape: {X_sample.shape}") # Expected: (sequence_length, num_features)
        print(f"Sample y shape: {y_sample.shape}")   # Expected: (1,) or (1,1)
        print(f"Input Dim for model should be: {X_sample.shape[1]}")
        print(f"Sampling frequency from dataset: {dataset.get_sampling_frequency()} Hz")

        # Check feature order for tau_phys (first 4 are critical for indexing in model)
        print("\nFeature names in order from dataset (X_sample.shape[1]):")
        for i, name in enumerate(dataset.FEATURE_NAMES):
            print(f"  Index {i}: {name}")
        
        # Verify derived quantities
        print("\nSample of processed data (first 5 rows of internal df):")
        print(dataset.data_df.head())
        
        print("\nSample of target 'tau_measured':")
        print(dataset.data_df['tau_measured'].head())


    except Exception as e:
        print(f"Error during ActuatorDataset test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)
            print(f"Cleaned up {dummy_csv_path}")

"""
Features to be used by the model must be specified in `features_to_use`.
The order of the first four features in this list is critical if `use_residual=True` in `ActuatorModel`,
as `_calculate_tau_phys` assumes:
- Index 0: current_angle_rad
- Index 1: target_angle_rad
- Index 2: current_ang_vel_rad_s
- Index 3: target_ang_vel_rad_s

The remaining features can be in any order and will be appended.
The `input_dim` for the model will be `len(features_to_use)`.
The target is `tau_measured`.
"""