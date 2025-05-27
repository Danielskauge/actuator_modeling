import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List


class ActuatorDataset(Dataset):
    """
    Dataset for actuator modeling.
    
    This dataset processes raw timestamped angle measurements and calculates 
    derived features (velocities, accelerations) and targets (torque).
    """
    
    def __init__(
        self,
        data_file: str,
        inertia: float = 1.0,
        radius_to_accelerometer_center: float = 0.01,
        control_frequency: float = 50.0,
        transform=None, 
        history_length: int = 5
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to data file (CSV expected)
                Expects data to have: timestamp, angle, target_angle, 
                ang_vel_x/y/z, lin_acc_x/y/z columns
            inertia: Inertia of the rod for torque calculation
            radius_to_accelerometer_center: Distance from rotation axis to accelerometer
            control_frequency: Frequency at which control inputs are sent
            transform: Any transforms to apply to the data
            history_length: Number of past timesteps to include in each sample
        """
        self.data_file = data_file
        self.inertia = inertia
        self.radius_to_accelerometer_center = radius_to_accelerometer_center
        self.transform = transform
        self.control_frequency = control_frequency
        self.history_length = history_length
        
        self._load_data()
        self._print_data_info()
        self._calculate_derived_quantities()
        self._prepare_features_and_targets()
        
    def _load_data(self):
        """Load the data from the CSV file."""
        try:
            # Load data
            self.data = pd.read_csv(self.data_file)
            
            # Time is in seconds from start, convert to proper timestamps as datetime
            start_time = pd.Timestamp('2023-01-01')
            self.data['datetime'] = start_time + pd.to_timedelta(self.data['time'], unit='s')
            self.data = self.data.set_index('datetime')
            
            # Downsample the data to the control frequency 
            if self.control_frequency < 1/min(np.diff(self.data.index.values.astype('float64'))):
                print(f"Downsampling data to {self.control_frequency} Hz")
                period = pd.Timedelta(seconds=1/self.control_frequency)
                self.data = self.data.resample(period).mean().dropna()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise  # Re-raise the exception to ensure it's not silently ignored
        
    def _print_data_info(self):
        """Print information about the data."""
        print(f"Data shape: {self.data.shape}")
        print(f"Data columns: {self.data.columns}")
        print(f"Data head: {self.data.head()}")
        
        
    def _calculate_derived_quantities(self):
        """Calculate angular acceleration and torque."""
        self.data['ang_acc'] = self.data['lin_acc_x'] / self.radius_to_accelerometer_center
        self.data['torque'] = self.inertia * self.data['ang_acc']
        
    def _prepare_features_and_targets(self):
        """Prepare feature matrix X and target vector y."""
        data_columns = ['angle', 'target_angle', 'ang_vel_x', 'ang_acc']
        data = self.data[data_columns].values  # shape (num_timesteps, num_data_columns)
        
        # Get torques as targets
        self.targets = self.data['torque'].values[self.history_length-1:]
        
        # Create sliding windows
        num_samples = len(data) - self.history_length + 1
        self.features = np.zeros((num_samples, self.history_length, len(data_columns)))
        
        for i in range(num_samples):
            self.features[i] = torch.tensor(data[i:i + self.history_length], dtype=torch.float32) # shape (history_length, num_data_columns)
        
        self.features = self.features.reshape(num_samples, -1) # Flatten the feature dimension, input layer must be 1d not 2d. Shape (num_samples, history_length * num_data_columns)
        self.features = torch.from_numpy(self.features).float()
        self.targets = torch.from_numpy(self.targets).float()
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        X = self.features[idx]
        y = self.targets[idx]
        
        if self.transform:
            X = self.transform(X)
        
        return X, y 