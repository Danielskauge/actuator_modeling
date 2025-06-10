import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional
from scipy.signal import butter, filtfilt
from scipy import interpolate
import os

# Constants
RAD_PER_DEG = np.pi / 180.0

class ActuatorDataset(Dataset):
    """
    Dataset for actuator modeling.
    
    Processes raw timestamped sensor data from CSV files, performs feature engineering,
    and generates sequences suitable for recurrent neural networks.
    Input features: current_angle_rad, target_angle_rad, current_ang_vel_rad_s
    """
    
  

    def __init__(
        self,
        csv_file_path: str,
        inertia: float,
        radius_accel: float,
        sequence_length_timesteps: int, # Number of timesteps per sequence
        resampling_frequency_hz: float, # Mandatory target sampling frequency
        main_gyro_axis: str = 'Gyro_Z',
        main_accel_axis: str = 'Acc_Y',
        filter_cutoff_freq_hz: Optional[float] = None,
        filter_cutoff_freqs: Optional[Dict[str, float]] = None,
        filter_order: int = 4,
        is_synthetic_data: bool = False, # Flag to indicate synthetic vs real data
        max_torque_nm: Optional[float] = None, # Max allowed torque threshold
        sensor_biases: Optional[Dict[str, float]] = None, # Sensor bias correction values
        max_commanded_vel: Optional[float] = None, # Max commanded angular velocity threshold (rad/s)
    ):
        """
        Args:
            csv_file_path: Path to the CSV data file.
            inertia: Inertia of the system (kg*m^2).
            radius_accel: Radius to the point where tangential acceleration is measured by accel_axis_for_torque (meters).
            sequence_length_timesteps: Number of timesteps per sequence (calculated by DataModule).
            resampling_frequency_hz: Data will be resampled to this frequency if not synthetic.
            main_gyro_axis: Gyroscope axis ('Gyro_X', 'Gyro_Y', 'Gyro_Z') for signed angular velocity.
            main_accel_axis: Accelerometer axis ('Acc_X', 'Acc_Y', 'Acc_Z') for tangential acceleration.
            filter_cutoff_freq_hz: Optional cutoff frequency in Hz for a low-pass Butterworth filter applied to the tangential acceleration signal used for deriving tau_measured. If None, no filter is applied.
            filter_cutoff_freqs: Optional dictionary of cutoff frequencies in Hz for filtering individual columns.
            filter_order: Order of the Butterworth filter if used.
            is_synthetic_data: Flag to indicate whether data is synthetic (True) or real (False). For real data, a 10.8° offset is subtracted from the commanded angle.
            max_torque_nm: Max allowed torque threshold
            sensor_biases: Dictionary containing sensor bias corrections (e.g., {'acc_y_bias': 0.039, 'gyro_z_bias': -0.002})
            max_commanded_vel: Max commanded angular velocity threshold (rad/s)
        """
        self.csv_file_path = csv_file_path
        self.inertia = inertia
        self.radius_accel = radius_accel
        self.main_gyro_axis = main_gyro_axis
        self.main_accel_axis = main_accel_axis
        self.filter_cutoff_freq_hz = filter_cutoff_freq_hz
        self.filter_cutoff_freqs = filter_cutoff_freqs
        self.filter_order = filter_order
        self.resampling_frequency_hz = resampling_frequency_hz # Store for internal use (esp. non-synthetic)
        self.sequence_length_timesteps = sequence_length_timesteps # Set directly from DataModule
        self.is_synthetic_data = is_synthetic_data # Store synthetic flag
        self.max_torque_nm = max_torque_nm # Store max torque threshold
        self.sensor_biases = sensor_biases or {} # Store sensor bias corrections
        self.max_commanded_vel = max_commanded_vel  # Store max commanded velocity threshold
        self.sampling_frequency: Optional[float] = None # Will be set during preprocessing
        self.data_df = self._load_and_preprocess_data()
        
        if self.sampling_frequency is None:
            raise ValueError(f"Sampling frequency is None for {os.path.basename(csv_file_path)}. This should not happen.")

        self.X_sequences, self.y_sequences, self.timestamps_sequences = self._create_sequences()
        # Optional filtering by commanded velocity threshold
        if self.max_commanded_vel is not None:
            # target_angle_rad is feature index 1
            commanded_seq = self.X_sequences[:, :, 1]
            times = self.timestamps_sequences[:, :, 0]
            dv = commanded_seq[:, 1:] - commanded_seq[:, :-1]
            dt = times[:, 1:] - times[:, :-1]
            v_cmd = dv / dt
            max_abs_v = torch.max(torch.abs(v_cmd), dim=1)[0]
            mask = max_abs_v <= self.max_commanded_vel
            kept = mask.sum().item()
            total = mask.numel()
            removed = total - kept  # Number of sequences filtered out
            self.X_sequences = self.X_sequences[mask]
            self.y_sequences = self.y_sequences[mask]
            self.timestamps_sequences = self.timestamps_sequences[mask]
            print(
                f"  Filtered sequences by max_commanded_vel ({self.max_commanded_vel:.3f} rad/s): "
                f"kept {kept}/{total}, removed {removed} ({removed/total:.1%})"
            )

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Loads data from CSV and performs all preprocessing and feature engineering."""
        try:
            df = pd.read_csv(self.csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        # Time handling - normalize to start from 0
        df['time_s'] = df['Time_ms'] / 1000.0
        df['time_s'] = df['time_s'] - df['time_s'].min()
        df = df.set_index('time_s')
        df = df.sort_index()
        
        # Remove the first 5 seconds of data
        df = df[df.index >= 5.0]
        print(f"  Removed first 5.0s of data from {os.path.basename(self.csv_file_path)}. Remaining duration: {df.index.max():.2f}s")

        # Compute original sampling frequency
        times = df.index.values
        if len(times) > 1:
            original_sampling_frequency = 1.0 / np.median(np.diff(times))
        else:
            original_sampling_frequency = 1.0

        # Apply sensor bias corrections
        if self.sensor_biases:
            bias_corrections_applied = []
            if 'acc_y_bias' in self.sensor_biases and 'Acc_Y' in df.columns:
                df['Acc_Y'] = df['Acc_Y'] - self.sensor_biases['acc_y_bias']
                bias_corrections_applied.append(f"Acc_Y bias: {self.sensor_biases['acc_y_bias']:.6f}")
            if 'gyro_z_bias' in self.sensor_biases and 'Gyro_Z' in df.columns:
                df['Gyro_Z'] = df['Gyro_Z'] - self.sensor_biases['gyro_z_bias']
                bias_corrections_applied.append(f"Gyro_Z bias: {self.sensor_biases['gyro_z_bias']:.6f}")
            
            if bias_corrections_applied:
                print(f"  Applied sensor bias corrections for {os.path.basename(self.csv_file_path)}: {', '.join(bias_corrections_applied)}")

        # Feature engineering
        df['angle_rad'] = df['Encoder_Angle'] * RAD_PER_DEG
        
        # Apply commanded angle offset correction only for real data (not synthetic)
        # if self.is_synthetic_data:
        df['target_angle_rad'] = df['Commanded_Angle'] * RAD_PER_DEG
        # else:
        #     # Real data: subtract 10.8 degrees offset from commanded angle
        #     df['target_angle_rad'] = (df['Commanded_Angle'] - 10.8) * RAD_PER_DEG
        #     print(f"  Applied 10.8° offset correction to commanded angle for {os.path.basename(self.csv_file_path)} (real data).")
            
        df['ang_vel_rad'] = df['Gyro_Z'] * RAD_PER_DEG
        
        # Compute raw torque
        lin_acc = df[self.main_accel_axis].values
        ang_acc = lin_acc / self.radius_accel
        torque = self.inertia * ang_acc
        # Save raw torque before filtering for debugging
        raw_torque = torque.copy()

        # Prepare feature arrays for filtering
        angle = df['angle_rad'].values
        target = df['target_angle_rad'].values
        ang_vel = df['ang_vel_rad'].values

        # Nyquist frequency (use original sampling)
        nyquist_freq = 0.5 * original_sampling_frequency

        # Determine per-column cutoffs
        if self.filter_cutoff_freqs:
            # Filter only measured/derived noisy signals: encoder angle, angular velocity, torque
            cutoffs = {
                'angle_rad': self.filter_cutoff_freqs.get('Encoder_Angle'),
                'ang_vel_rad': self.filter_cutoff_freqs.get(self.main_gyro_axis),
                'torque': self.filter_cutoff_freqs.get(self.main_accel_axis),
            }
        else:
            cutoffs = {'torque': self.filter_cutoff_freq_hz}

        # Apply filtering
        for key, cutoff in cutoffs.items():
            if cutoff is not None:
                if cutoff < nyquist_freq:
                    Wn = cutoff / nyquist_freq
                    b, a = butter(self.filter_order, Wn, btype='low', analog=False)
                    if key == 'angle_rad':
                        angle = filtfilt(b, a, angle)
                    elif key == 'target_angle_rad':
                        target = filtfilt(b, a, target)
                    elif key == 'ang_vel_rad':
                        ang_vel = filtfilt(b, a, ang_vel)
                    elif key == 'torque':
                        torque = filtfilt(b, a, torque)
                    print(f"  Applied Butterworth low-pass filter on '{key}' (cutoff: {cutoff:.2f} Hz, order: {self.filter_order}) for {os.path.basename(self.csv_file_path)}.")
                else:
                    print(f"  Warning: Cutoff ({cutoff:.2f} Hz) >= Nyquist ({nyquist_freq:.2f} Hz) for '{key}'; no filtering applied.")

        # Save filtered torque before clipping to visualize smoothing effect
        filtered_torque = torque.copy()
        
        # Clip torque to max limit
        if self.max_torque_nm is not None:
            max_torque_found = np.max(np.abs(torque))
            if max_torque_found > self.max_torque_nm:
                torque = np.clip(torque, -self.max_torque_nm, self.max_torque_nm)
                print(f"  Clipped torque from {max_torque_found:.3f} Nm to {self.max_torque_nm:.3f} Nm for {os.path.basename(self.csv_file_path)}")
        
        # Assign filtered data back
        df['angle_rad'] = angle
        df['target_angle_rad'] = target
        df['ang_vel_rad'] = ang_vel
        # Include raw, filtered (pre-clipping), and final torque (post-clipping)
        df['torque_raw'] = raw_torque
        df['torque_filtered'] = filtered_torque
        df['torque'] = torque
        
        # Keep the main acceleration axis for visualization callbacks
        df = df[['torque', 'torque_filtered', 'torque_raw', 'angle_rad', 'target_angle_rad', 'ang_vel_rad', self.main_accel_axis]]
        
        # Resampling after filtering
        if self.resampling_frequency_hz is not None:
            start_time = df.index.min()
            end_time = df.index.max()
            dt = 1.0 / self.resampling_frequency_hz
            new_times = np.arange(start_time, end_time + dt, dt)
            resampled_data = {}
            for col in df.columns:
                f = interpolate.interp1d(df.index.values, df[col].values, kind='linear', bounds_error=False, fill_value='extrapolate')
                resampled_data[col] = f(new_times)
            df = pd.DataFrame(resampled_data, index=new_times)
            df.index.name = 'time_s'
            self.sampling_frequency = self.resampling_frequency_hz
        else:
            self.sampling_frequency = original_sampling_frequency
        
        return df

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates sequences of features, corresponding targets, and timestamps for targets."""
        # Build features including previous torque
        raw_features = self.data_df[['angle_rad', 'target_angle_rad', 'ang_vel_rad']].values
        torque_values = self.data_df['torque'].values
        # Shift torque by one timestep as previous torque; pad start with zero
        torque_prev = np.concatenate(([0.0], torque_values[:-1]))
        feature_data = np.column_stack((raw_features, torque_prev))
        target_data = self.data_df['torque'].values
        # Use the actual timestamps (starting from 5.0s after trim) as float seconds
        timestamps_numeric = self.data_df.index.values

        num_samples = len(feature_data) - self.sequence_length_timesteps + 1
        
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data points ({len(feature_data)}) to create sequences "
                f"of length {self.sequence_length_timesteps} for file {self.csv_file_path}."
            )

        X_np = np.array([feature_data[i : i + self.sequence_length_timesteps] for i in range(num_samples)])
        # The targets correspond to torque sequences for each timestep in the sequence
        y_np = np.array([target_data[i : i + self.sequence_length_timesteps] for i in range(num_samples)])
        # The timestamps correspond to each timestep in the sequence
        timestamps_np = np.array([timestamps_numeric[i : i + self.sequence_length_timesteps] for i in range(num_samples)])
        
        X_tensor = torch.from_numpy(X_np).float()
        # Targets: shape [num_samples, seq_len] → add last dim to get [num_samples, seq_len, 1]
        y_tensor = torch.from_numpy(y_np).float().unsqueeze(-1)
        # Timestamps: same shape as targets
        timestamps_tensor = torch.from_numpy(timestamps_np).float().unsqueeze(-1)

        return X_tensor, y_tensor, timestamps_tensor

    def __len__(self) -> int:
        return len(self.X_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_sequences[idx], self.y_sequences[idx], self.timestamps_sequences[idx]

    def get_sampling_frequency(self) -> float:
        return self.sampling_frequency

    @staticmethod
    def get_input_dim() -> int:
        # Now includes previous torque as an extra input feature
        return 4

    def get_sequence_length_timesteps(self) -> int:
        """Returns the actual sequence length in timesteps for this dataset instance."""
        return self.sequence_length_timesteps
