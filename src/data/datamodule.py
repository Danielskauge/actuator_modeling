import os
import glob
from typing import Optional, List, Dict, Any, Tuple
import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
import pandas as pd
import numpy as np

from src.data.datasets import ActuatorDataset

class ActuatorDataModule(pl.LightningDataModule):
    """
    LightningDataModule for actuator modeling data.
    Supports loading multiple CSV files per inertia group from organized subfolders.
    Supports two main operational modes for evaluation:
    1. Global Split Mode: All data is combined and split into train/val/test sets.
    2. LOMO CV Fold Mode: Data is set up for a specific fold of Leave-One-Mass-Out CV.
    """

    def __init__(
        self,
        data_base_dir: str,  # Base directory containing subfolders for each inertia group
        inertia_groups: List[Dict[str, Any]],  # Each dict: {'id': str, 'folder': str, 'inertia': float}
        radius_accel: float, # Radius for tangential acceleration calculation in ActuatorDataset
        filter_order: int,
        filter_cutoff_freq_hz: Optional[float],
        sequence_duration_s: float,
        global_train_ratio: float,
        global_val_ratio: float,
        resampling_frequency_hz: Optional[float] = None,
        synthetic_data_frequency_hz: Optional[float] = None,
        main_gyro_axis: str = 'Gyro_Z',
        main_accel_axis: str = 'Acc_Y',
        # DataLoader params
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if synthetic_data_frequency_hz is not None:
            self.sampling_frequency = synthetic_data_frequency_hz
        else:
            self.sampling_frequency = resampling_frequency_hz
            
        self._normalization_stats_computed_for_global_train = False
        
        self.sequence_length_timesteps = int(round(sequence_duration_s * self.sampling_frequency))
        
        self.datasets_by_group: Dict[str, List[ActuatorDataset]] = {}
        self.ordered_group_ids: List[str] = []
        
        # Global split datasets
        self.global_train_dataset: Optional[Dataset] = None
        self.global_val_dataset: Optional[Dataset] = None
        self.global_test_dataset: Optional[Dataset] = None
        
        # LOMO fold datasets
        self.current_fold_train_dataset: Optional[Dataset] = None
        self.current_fold_val_dataset: Optional[Dataset] = None
        self.current_fold_test_dataset: Optional[Dataset] = None
        
        self._current_mode: Optional[str] = None
        self.input_dim = ActuatorDataset.get_input_dim()
        
        # Normalization stats
        self.input_mean: Optional[torch.Tensor] = None
        self.input_std: Optional[torch.Tensor] = None
        self.target_mean: Optional[torch.Tensor] = None
        self.target_std: Optional[torch.Tensor] = None

    def _save_normalization_stats_to_json(self, file_path: str):
        """Save normalization statistics to JSON file."""
        stats_dict = {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist()
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)

    def _load_all_datasets_once(self):
        if not self.datasets_by_group:
            print(f"DataModule: Discovering and loading CSV files from {len(self.hparams.inertia_groups)} inertia groups...")
            total_csv_files = 0
            for group_config in self.hparams.inertia_groups:
                group_id = group_config.get('id')
                group_folder = group_config.get('folder')
                group_inertia = group_config.get('inertia')
                
                
                # Validation
                if not group_id or not group_folder or group_inertia is None:
                    raise ValueError(f"Group config missing 'id', 'folder', or 'inertia': {group_config}")
                group_path = os.path.join(self.hparams.data_base_dir, group_folder)
                if not os.path.exists(group_path):
                    print(f"Warning: Group folder does not exist: {group_path}. Skipping group '{group_id}'.")
                    continue
                csv_pattern = os.path.join(group_path, "*.csv")
                csv_files = glob.glob(csv_pattern)
                if not csv_files:
                    print(f"Warning: No CSV files found in {group_path}. Skipping group '{group_id}'.")
                    continue
                csv_files.sort()
                print(f"  Group '{group_id}' (inertia={group_inertia}): Found {len(csv_files)} CSV files in {group_path}")
                
                
                group_datasets = []
                for csv_file in csv_files:
                    try:
                        # Determine if data is synthetic based on whether synthetic_data_frequency_hz is set
                        is_synthetic = self.hparams.synthetic_data_frequency_hz is not None
                        
                        dataset_instance = ActuatorDataset(
                            csv_file_path=csv_file,
                            inertia=group_inertia,
                            radius_accel=self.hparams.radius_accel,
                            sequence_length_timesteps=self.sequence_length_timesteps,
                            resampling_frequency_hz=self.hparams.resampling_frequency_hz,
                            main_gyro_axis=self.hparams.main_gyro_axis,
                            main_accel_axis=self.hparams.main_accel_axis,
                            filter_cutoff_freq_hz=self.hparams.filter_cutoff_freq_hz,
                            filter_order=self.hparams.filter_order,
                            is_synthetic_data=is_synthetic,
                        )
                        group_datasets.append(dataset_instance)
                        total_csv_files += 1
                        print(f"    Loaded {os.path.basename(csv_file)}: {len(dataset_instance)} sequences (target Fs: {self.sampling_frequency:.2f} Hz)")
                    except Exception as e:
                        print(f"    Error loading {csv_file}: {e}")
                        continue
                    
                if group_datasets:
                    self.datasets_by_group[group_id] = group_datasets
                    self.ordered_group_ids.append(group_id)
                    print(f"    Total sequences for group '{group_id}': {sum(len(ds) for ds in group_datasets)}")
                else:
                    print(f"    No valid datasets loaded for group '{group_id}'")
            if not self.datasets_by_group:
                raise RuntimeError("No datasets were loaded from any inertia group.")
            print(f"DataModule: Enforcing target sampling frequency: {self.sampling_frequency:.2f} Hz")
            print(f"DataModule: Loaded {total_csv_files} CSV files from {len(self.datasets_by_group)} inertia groups.")
            print(f"  Input dim: {self.input_dim}, Sequence length (timesteps): {self.sequence_length_timesteps}")

    def prepare_data(self):
        # This ensures data (CSVs) are accessible.
        # Actual loading into ActuatorDataset happens in setup methods via _load_all_datasets_once.
        # Normalization stats will be computed in setup based on the training set.
        self._load_all_datasets_once()

    def _calculate_normalization_stats(self, reference_dataset: Dataset):
        """
        Calculates and stores normalization statistics (mean, std) for inputs and targets
        based on the provided reference_dataset.
        This method should ideally be called only once with the global training dataset.
        """
        print(f"  Calculating normalization statistics from {len(reference_dataset)} samples in the reference dataset...")
        if len(reference_dataset) == 0:
            print("  Warning: Reference dataset for normalization is empty. Stats will be 0 for mean and 1 for std.")
            input_dim = ActuatorDataset.get_input_dim()
            self.input_mean = torch.zeros(input_dim)
            self.input_std = torch.ones(input_dim)
            self.target_mean = torch.zeros(1)
            self.target_std = torch.ones(1)
            # If this was intended for the global train set, mark it as "computed" (even if badly)
            if hasattr(self, 'global_train_dataset') and self.global_train_dataset is not None and reference_dataset == self.global_train_dataset: # Check identity
                self._normalization_stats_computed_for_global_train = True
            return

        temp_loader = DataLoader(reference_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
        all_inputs_flattened = []
        all_targets = []

        for x_batch, y_batch, _ in temp_loader: # Unpack and ignore timestamps
            # x_batch: (batch_size, seq_len, input_dim)
            # y_batch: (batch_size, 1) or (batch_size,)
            all_inputs_flattened.append(x_batch.reshape(-1, x_batch.size(-1))) # Flatten seq_len into batch dim
            all_targets.append(y_batch.reshape(-1, 1)) # Ensure target is [N, 1]

        if not all_inputs_flattened or not all_targets:
            print("  Warning: No data collected from reference dataset for normalization. Stats will be 0 for mean and 1 for std.")
            input_dim = ActuatorDataset.get_input_dim()
            self.input_mean = torch.zeros(input_dim)
            self.input_std = torch.ones(input_dim)
            self.target_mean = torch.zeros(1)
            self.target_std = torch.ones(1)
            if hasattr(self, 'global_train_dataset') and self.global_train_dataset is not None and reference_dataset == self.global_train_dataset:
                 self._normalization_stats_computed_for_global_train = True
            return

        stacked_inputs = torch.cat(all_inputs_flattened, dim=0)   # Shape: (total_samples * seq_len, input_dim)
        stacked_targets = torch.cat(all_targets, dim=0) # Shape: (total_samples, 1)

        self.input_mean = torch.mean(stacked_inputs, dim=0)
        self.input_std = torch.std(stacked_inputs, dim=0)
        self.input_std[self.input_std < 1e-6] = 1.0 # Prevent division by zero / very small std

        self.target_mean = torch.mean(stacked_targets, dim=0) # Shape (1,)
        self.target_std = torch.std(stacked_targets, dim=0)   # Shape (1,)
        self.target_std[self.target_std < 1e-6] = 1.0

        print(f"  Input Mean: {self.input_mean.tolist()}")
        print(f"  Input Std:  {self.input_std.tolist()}")
        print(f"  Target Mean: {self.target_mean.tolist()}") # target_mean is a tensor, convert to list
        print(f"  Target Std:  {self.target_std.tolist()}")  # target_std is a tensor, convert to list
        
        # This flag is now specific to when stats are computed for the global_train_dataset
        if hasattr(self, 'global_train_dataset') and self.global_train_dataset is not None and \
           reference_dataset == self.global_train_dataset: 
            self._normalization_stats_computed_for_global_train = True
        # For LOMO, stats are computed per fold, so this global flag isn't set by LOMO's reference_dataset
        
        # Attempt to save stats to JSON if they are computed
        # The actual saving point might be better placed after the specific setup methods
        # (setup_for_global_run or setup_for_lomo_fold) determine the final reference_dataset
        # and path. For now, this demonstrates the calculation.
        # Actual saving will be triggered by those methods.

    def setup(self, stage: Optional[str] = None):
        """
        This method is called by PyTorch Lightning.
        For this DataModule, the actual setup (global split or fold setup)
        is deferred to specific methods like `setup_for_global_run` or `setup_for_lomo_fold`,
        which are expected to be called by the main training script.
        This main setup() currently does not perform split-specific logic itself.
        Ensure `prepare_data()` (which calls `_load_all_datasets_once()`) has been called.
        """
        # _load_all_datasets_once() is called in prepare_data
        # The mode-specific setups (setup_for_global_run or setup_for_lomo_fold)
        # are responsible for setting up datasets and calculating normalization stats if needed.
        # If PL calls this automatically, it doesn't inherently know which mode to set up.
        # The training script should explicitly call the mode-specific setup.
        pass

    def setup_for_global_run(self):
        """Sets up the DataModule for a single global train/val/test run."""
        # _load_all_datasets_once() # Called in prepare_data
        self._current_mode = "global"
        
        # Collect all datasets from all groups
        all_datasets = []
        for group_id in self.ordered_group_ids:
            all_datasets.extend(self.datasets_by_group[group_id])
        
        print(f"DataModule: Setting up for a GLOBAL run with {len(all_datasets)} total datasets from {len(self.datasets_by_group)} groups.")

        if not all_datasets:
            raise RuntimeError("No datasets available to perform a global split.")

        combined_dataset = ConcatDataset(all_datasets)
        total_samples = len(combined_dataset)
        print(f"  Total combined samples for global split: {total_samples}")

        train_ratio = self.hparams.global_train_ratio
        val_ratio = self.hparams.global_val_ratio
        test_ratio = 1.0 - train_ratio - val_ratio

        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1 and abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5):
            raise ValueError(f"Global split ratios are invalid: train={train_ratio}, val={val_ratio}, test={test_ratio}. They must sum to 1 and be > 0.")

        num_train = int(train_ratio * total_samples)
        num_val = int(val_ratio * total_samples)
        num_test = total_samples - num_train - num_val # Ensure all samples are used

        if num_train == 0 or num_val == 0 or num_test == 0:
            raise ValueError(f"Calculated 0 samples for one or more global splits: train={num_train}, val={num_val}, test={num_test}. Check ratios and total samples ({total_samples}).")

        generator = torch.Generator().manual_seed(self.hparams.seed)
        self.global_train_dataset, self.global_val_dataset, self.global_test_dataset = random_split(
            combined_dataset, [num_train, num_val, num_test], generator=generator
        )
        print(f"  Global split: {len(self.global_train_dataset)} train, {len(self.global_val_dataset)} val, {len(self.global_test_dataset)} test samples.")

        # Calculate normalization stats based on the global_train_dataset if not already done
        if not self._normalization_stats_computed_for_global_train:
            if self.global_train_dataset and len(self.global_train_dataset) > 0:
                self._calculate_normalization_stats(self.global_train_dataset)
                # Save stats after computing for global run
                stats_dir = os.path.join(self.hparams.data_base_dir, "processed_stats")
                os.makedirs(stats_dir, exist_ok=True)
                stats_file_path = os.path.join(stats_dir, "normalization_stats_global.json")
                self._save_normalization_stats_to_json(stats_file_path)
            else:
                print("Warning: Global train dataset is empty or not set. Cannot compute global normalization stats.")
                # Fallback to zero mean, one std to prevent errors, though this is not ideal.
                input_dim = ActuatorDataset.get_input_dim()
                self.input_mean = torch.zeros(input_dim)
                self.input_std = torch.ones(input_dim)
                self.target_mean = torch.zeros(1)
                self.target_std = torch.ones(1)
                self._normalization_stats_computed_for_global_train = True # Mark as "done"
        else:
            print("  Global normalization stats already computed. Using existing stats.")

    def setup_for_lomo_fold(self, fold_index: int):
        """Sets up data for a specific LOMO CV fold. No intra-fold splitting of seen data."""
        # _load_all_datasets_once() # Called in prepare_data
        self._current_mode = "lomo_fold"

        num_total_groups = len(self.ordered_group_ids)
        if not 0 <= fold_index < num_total_groups:
            raise ValueError(f"LOMO fold index {fold_index} out of bounds for {num_total_groups} groups.")
        if num_total_groups < 2:
             print(f"Warning: LOMO CV is being set up with only {num_total_groups} group(s). Generalization testing will be limited or not meaningful.")

        print(f"\nDataModule: Setting up for LOMO CV - Fold {fold_index + 1}/{num_total_groups}.")

        # The group at fold_index is for validation and testing (unseen for this fold)
        held_out_group_id = self.ordered_group_ids[fold_index]
        held_out_datasets = self.datasets_by_group[held_out_group_id]
        
        # Combine all datasets from the held-out group for validation and testing
        if len(held_out_datasets) == 1:
            self.current_fold_val_dataset = held_out_datasets[0]
            self.current_fold_test_dataset = held_out_datasets[0]
        else:
            combined_held_out = ConcatDataset(held_out_datasets)
            self.current_fold_val_dataset = combined_held_out
            self.current_fold_test_dataset = combined_held_out
        
        print(f"  Held-out group '{held_out_group_id}' for Val/Test in Fold {fold_index + 1}: {len(self.current_fold_val_dataset)} sequences from {len(held_out_datasets)} CSV files")

        # All other groups form the training set for this fold
        training_datasets = []
        for i, group_id in enumerate(self.ordered_group_ids):
            if i != fold_index:
                training_datasets.extend(self.datasets_by_group[group_id])

        if training_datasets:
            self.current_fold_train_dataset = ConcatDataset(training_datasets)
            num_training_groups = num_total_groups - 1
            print(f"  Training data for Fold {fold_index + 1}: {len(self.current_fold_train_dataset)} sequences from {num_training_groups} group(s).")
            # Calculate normalization stats for THIS FOLD's training data
            if self.current_fold_train_dataset and len(self.current_fold_train_dataset) > 0:
                self._calculate_normalization_stats(self.current_fold_train_dataset)
                # Save stats after computing for LOMO fold
                stats_dir = os.path.join(self.hparams.data_base_dir, "processed_stats")
                os.makedirs(stats_dir, exist_ok=True)
                stats_file_path = os.path.join(stats_dir, f"normalization_stats_fold_{fold_index}.json")
                self._save_normalization_stats_to_json(stats_file_path)
            else:
                print(f"  Warning: LOMO Fold {fold_index + 1} has no training data. Cannot compute fold-specific normalization stats. Model will use zero mean, unit std.")
                input_dim = ActuatorDataset.get_input_dim()
                self.input_mean = torch.zeros(input_dim)
                self.input_std = torch.ones(input_dim)
                self.target_mean = torch.zeros(1)
                self.target_std = torch.ones(1)
        else: # This case implies no training data for the fold (e.g. only 1 group total)
             print(f"  Warning: LOMO Fold {fold_index + 1} has no training data. Cannot compute fold-specific normalization stats. Model will use zero mean, unit std.")
             input_dim = ActuatorDataset.get_input_dim()
             self.input_mean = torch.zeros(input_dim)
             self.input_std = torch.ones(input_dim)
             self.target_mean = torch.zeros(1)
             self.target_std = torch.ones(1)

    def train_dataloader(self) -> DataLoader:
        dataset_to_use = None
        if self._current_mode == "global":
            dataset_to_use = self.global_train_dataset
            if dataset_to_use is None: raise RuntimeError("Global train dataset not set. Call setup_for_global_run().")
        elif self._current_mode == "lomo_fold":
            dataset_to_use = self.current_fold_train_dataset
            if dataset_to_use is None: raise RuntimeError("LOMO fold train dataset not set or is empty. Call setup_for_lomo_fold(). This may occur if only 1 total group is available for LOMO.")
        else:
            raise RuntimeError("DataModule mode not set. Call setup_for_global_run() or setup_for_lomo_fold().")
        
        return DataLoader(dataset_to_use, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def val_dataloader(self) -> DataLoader:
        dataset_to_use = None
        if self._current_mode == "global":
            dataset_to_use = self.global_val_dataset
            if dataset_to_use is None: raise RuntimeError("Global val dataset not set. Call setup_for_global_run().")
        elif self._current_mode == "lomo_fold":
            dataset_to_use = self.current_fold_val_dataset
            if dataset_to_use is None: raise RuntimeError("LOMO fold val dataset not set. Call setup_for_lomo_fold().")
        else:
            raise RuntimeError("DataModule mode not set.")
            
        return DataLoader(dataset_to_use, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self) -> DataLoader:
        dataset_to_use = None
        if self._current_mode == "global":
            dataset_to_use = self.global_test_dataset
            if dataset_to_use is None: raise RuntimeError("Global test dataset not set. Call setup_for_global_run().")
        elif self._current_mode == "lomo_fold":
            dataset_to_use = self.current_fold_test_dataset # This is the unseen inertia for the fold
            if dataset_to_use is None: raise RuntimeError("LOMO fold test dataset not set. Call setup_for_lomo_fold().")
        else:
            raise RuntimeError("DataModule mode not set.")

        return DataLoader(dataset_to_use, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    # Accessors for model to get necessary dimensions
    def get_input_dim(self) -> int: return self.input_dim
    def get_sequence_length_timesteps(self) -> int:
        if self.sequence_length_timesteps is None:
            raise RuntimeError("Sequence length in timesteps has not been set. Ensure data is loaded.")
        return self.sequence_length_timesteps
    def get_sampling_frequency(self) -> Optional[float]: return self.sampling_frequency
    def get_num_lomo_folds(self) -> int:
        # self._load_all_datasets_once() # Called in prepare_data
        return len(self.ordered_group_ids)

    def get_input_normalization_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the computed mean and std for input features."""
        if self.input_mean is not None and self.input_std is not None:
            return self.input_mean, self.input_std
        print("Warning: Input normalization stats requested but not computed or available.")
        return None

    def get_target_normalization_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the computed mean and std for the target variable."""
        if self.target_mean is not None and self.target_std is not None:
            return self.target_mean, self.target_std
        print("Warning: Target normalization stats requested but not computed or available.")
        return None

