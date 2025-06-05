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
        filter_order: int = 4,
        is_synthetic_data: bool = False, # Flag to indicate synthetic vs real data
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
            filter_order: Order of the Butterworth filter if used.
            is_synthetic_data: Flag to indicate whether data is synthetic (True) or real (False). For real data, a 10.8° offset is subtracted from the commanded angle.
        """
        self.csv_file_path = csv_file_path
        self.inertia = inertia
        self.radius_accel = radius_accel
        self.main_gyro_axis = main_gyro_axis
        self.main_accel_axis = main_accel_axis
        self.filter_cutoff_freq_hz = filter_cutoff_freq_hz
        self.filter_order = filter_order
        self.resampling_frequency_hz = resampling_frequency_hz # Store for internal use (esp. non-synthetic)
        self.sequence_length_timesteps = sequence_length_timesteps # Set directly from DataModule
        self.is_synthetic_data = is_synthetic_data # Store synthetic flag
        self.sampling_frequency: Optional[float] = None # Will be set during preprocessing
        self.data_df = self._load_and_preprocess_data()
        
        if self.sampling_frequency is None:
            raise ValueError(f"Sampling frequency is None for {os.path.basename(csv_file_path)}. This should not happen.")

        self.X_sequences, self.y_sequences, self.timestamps_sequences = self._create_sequences()

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
        
        # Remove the first 1 second of data
        df = df[df.index >= 1.0]
        print(f"  Removed first 1.0s of data from {os.path.basename(self.csv_file_path)}. Remaining duration: {df.index.max():.2f}s")

        # Resampling using scipy interpolation for better control
        if self.resampling_frequency_hz is not None:
            # Create new time grid starting from 0
            start_time = 0.0  # Now always starts from 0
            end_time = df.index.max()
            dt = 1.0 / self.resampling_frequency_hz
            new_times = np.arange(start_time, end_time + dt, dt)
            
            # Interpolate each column
            resampled_data = {}
            for col in df.columns:
                f = interpolate.interp1d(df.index.values, df[col].values, 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                resampled_data[col] = f(new_times)
            
            # Create new dataframe
            df = pd.DataFrame(resampled_data, index=new_times)
            df.index.name = 'time_s'
            self.sampling_frequency = self.resampling_frequency_hz
        else:
            self.sampling_frequency = 1.0 / (df.index[1] - df.index[0]) if len(df) > 1 else 1.0

        # Feature engineering
        df['angle_rad'] = df['Encoder_Angle'] * RAD_PER_DEG
        
        # Apply commanded angle offset correction only for real data (not synthetic)
        if self.is_synthetic_data:
            df['target_angle_rad'] = df['Commanded_Angle'] * RAD_PER_DEG
        else:
            # Real data: subtract 10.8 degrees offset from commanded angle
            df['target_angle_rad'] = (df['Commanded_Angle'] - 10.8) * RAD_PER_DEG
            print(f"  Applied 10.8° offset correction to commanded angle for {os.path.basename(self.csv_file_path)} (real data).")
            
        df['ang_vel_rad'] = df['Gyro_Z'] * RAD_PER_DEG
        
        lin_acc = df[self.main_accel_axis]
        ang_acc = lin_acc / self.radius_accel
        
        torque = self.inertia * ang_acc

        # Low pass filtering of torque
        if self.filter_cutoff_freq_hz is not None:
            nyquist_freq = 0.5 * self.sampling_frequency
            # Ensure filter_cutoff_freq_hz is less than Nyquist frequency
            if self.filter_cutoff_freq_hz < nyquist_freq:
                Wn = self.filter_cutoff_freq_hz / nyquist_freq
                b, a = butter(self.filter_order, Wn, btype='low', analog=False)
                torque = filtfilt(b, a, torque)
                print(f"  Applied Butterworth low-pass filter (cutoff: {self.filter_cutoff_freq_hz} Hz, order: {self.filter_order}) to tangential acceleration for {os.path.basename(self.csv_file_path)}.")
            else:
                print(f"  Warning: Filter cutoff frequency ({self.filter_cutoff_freq_hz} Hz) is >= Nyquist frequency ({nyquist_freq} Hz) for {os.path.basename(self.csv_file_path)}. Using unfiltered tangential acceleration for torque calculation.")
            
        # Target    
        df['torque'] = torque
        
        # Keep the main acceleration axis for visualization callbacks
        df = df[['torque', 'angle_rad', 'target_angle_rad', 'ang_vel_rad', self.main_accel_axis]]
        
        return df

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates sequences of features, corresponding targets, and timestamps for targets."""
        feature_data = self.data_df[['angle_rad', 'target_angle_rad', 'ang_vel_rad']].values
        target_data = self.data_df['torque'].values
        # Now that time starts from 0, we can use it directly as float seconds
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
        return 3

    def get_sequence_length_timesteps(self) -> int:
        """Returns the actual sequence length in timesteps for this dataset instance."""
        return self.sequence_length_timesteps
