import os
from typing import Optional, Dict, Any, List

import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit

from src.data.datasets import ActuatorDataset


class ActuatorDataModule(pl.LightningDataModule):
    """
    LightningDataModule for actuator modeling data.
    
    This handles data loading, preparation, and creating dataloaders for training,
    validation, and testing.
    """
    
    def __init__(
        self,
        data_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        history_length: int = 5,
        inertia: float = 1.0,
        radius_to_accelerometer_center: float = 0.01,
        control_frequency: float = 50.0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the ActuatorDataModule.
        
        Args:
            data_path: Path to the data file
            batch_size: Batch size for training/validation/testing
            num_workers: Number of workers for DataLoader
            history_length: Number of past timesteps to include
            inertia: Inertia of the rod for torque calculation
            radius_to_accelerometer_center: Distance from rotation axis to accelerometer
            control_frequency: Frequency at which control inputs are sent
            test_size: Proportion of data to use for testing
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.history_length = history_length
        self.inertia = inertia
        self.radius_to_accelerometer_center = radius_to_accelerometer_center
        self.control_frequency = control_frequency
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        
        # Will be set in setup
        self.dataset = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
    
    
    def setup(self, stage: Optional[str] = None):
        """
        Load and split datasets.
        
        This method is called on every GPU.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Create dataset once
        if self.dataset is None:
            self.dataset = ActuatorDataset(
                data_file=self.data_file,
                inertia=self.inertia,
                radius_to_accelerometer_center=self.radius_to_accelerometer_center,
                control_frequency=self.control_frequency,
                history_length=self.history_length
            )
            
            # Use TimeSeriesSplit for proper time series validation
            n_samples = len(self.dataset)
            val_samples = int(n_samples * self.val_size)
            test_samples = int(n_samples * self.test_size)
            train_samples = n_samples - test_samples - val_samples
            
            train_idx = np.arange(train_samples)
            val_idx = np.arange(train_samples, train_samples + val_samples)
            test_idx = np.arange(train_samples + val_samples, n_samples)
            
            self.train_indices = train_idx
            self.val_indices = val_idx
            self.test_indices = test_idx
    
    def train_dataloader(self):
        """Return the train dataloader."""
        train_subset = Subset(self.dataset, self.train_indices)
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        val_subset = Subset(self.dataset, self.val_indices)
        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Return the test dataloader."""
        test_subset = Subset(self.dataset, self.test_indices)
        return DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self) -> List[int]:
        """Return the input feature dimensions."""
        x_sample, _ = self.dataset[0]
        return list(x_sample.shape)  # Return the last dimension (feature dimension) 