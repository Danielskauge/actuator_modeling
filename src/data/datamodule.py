import os
import glob
from typing import Optional, List, Dict, Any, Tuple
import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split # Added random_split
import pandas as pd # Keep for __main__ example
import numpy as np  # Keep for __main__ example

from src.data.datasets import ActuatorDataset # ActuatorDataset now has fixed input_dim and sequence_length

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
        gyro_axis_for_ang_vel: str = 'Gyro_Z',
        accel_axis_for_torque: str = 'Acc_Y',
        target_name: str = 'tau_measured',
        # DataLoader params
        batch_size: int = 32,
        num_workers: int = 4,
        # For Global Split Mode
        global_train_ratio: float = 0.7,
        global_val_ratio: float = 0.15,
        # global_test_ratio is inferred (1 - train_ratio - val_ratio)
        seed: int = 42, # Seed for reproducible splits (both global and LOMO fold selection if applicable)
        **kwargs # Catches any other params from Hydra config (e.g. radius_load)
    ):
        super().__init__()
        # Manually save hyperparameters
        self.save_hyperparameters() # Saves all __init__ args as hparams

        self.datasets_by_group: Dict[str, List[ActuatorDataset]] = {}  # group_id -> list of ActuatorDataset instances
        self.ordered_group_ids: List[str] = []  # Consistent ordering for LOMO CV
        
        # For Global Split Mode
        self.global_train_dataset: Optional[Dataset] = None
        self.global_val_dataset: Optional[Dataset] = None
        self.global_test_dataset: Optional[Dataset] = None

        # For LOMO CV Fold Mode
        self.current_fold_train_dataset: Optional[Dataset] = None
        self.current_fold_val_dataset: Optional[Dataset] = None # Unseen data for validation in a fold
        self.current_fold_test_dataset: Optional[Dataset] = None # Unseen data for testing in a fold

        # Current operation mode (set by setup methods)
        self._current_mode: Optional[str] = None # "global" or "lomo_fold"

        # These are now fixed by ActuatorDataset class definition
        self.input_dim = ActuatorDataset.get_input_dim()
        self.sequence_length = ActuatorDataset.get_sequence_length()
        self.sampling_frequency: Optional[float] = None # Will be determined from the first dataset

        # Normalization statistics
        self.input_mean: Optional[torch.Tensor] = None
        self.input_std: Optional[torch.Tensor] = None
        self.target_mean: Optional[torch.Tensor] = None
        self.target_std: Optional[torch.Tensor] = None
        self._normalization_stats_computed_for_global_train: bool = False

    def _save_normalization_stats_to_json(self, file_path: str):
        """Saves normalization statistics to a JSON file."""
        if self.input_mean is None or self.input_std is None or \
           self.target_mean is None or self.target_std is None:
            print("Warning: Normalization stats are not computed. Cannot save.")
            return

        stats_dict = {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist()
        }
        try:
            with open(file_path, 'w') as f:
                json.dump(stats_dict, f, indent=4)
            print(f"  Normalization statistics saved to {file_path}")
        except Exception as e:
            print(f"  Error saving normalization statistics to {file_path}: {e}")

    def _load_all_datasets_once(self):
        """Helper to load all ActuatorDataset instances if not already loaded."""
        if not self.datasets_by_group:
            print(f"DataModule: Discovering and loading CSV files from {len(self.hparams.inertia_groups)} inertia groups...")
            
            total_csv_files = 0
            dataset_sampling_frequencies = []
            
            for group_config in self.hparams.inertia_groups:
                group_id = group_config.get('id')
                group_folder = group_config.get('folder')
                group_inertia = group_config.get('inertia')
                
                if not group_id or not group_folder or group_inertia is None:
                    raise ValueError(f"Group config missing 'id', 'folder', or 'inertia': {group_config}")
                
                # Build path to group's subfolder
                group_path = os.path.join(self.hparams.data_base_dir, group_folder)
                
                if not os.path.exists(group_path):
                    print(f"Warning: Group folder does not exist: {group_path}. Skipping group '{group_id}'.")
                    continue
                
                # Find all CSV files in this group's folder
                csv_pattern = os.path.join(group_path, "*.csv")
                csv_files = glob.glob(csv_pattern)
                
                if not csv_files:
                    print(f"Warning: No CSV files found in {group_path}. Skipping group '{group_id}'.")
                    continue
                
                # Sort for consistent ordering
                csv_files.sort()
                
                print(f"  Group '{group_id}' (inertia={group_inertia}): Found {len(csv_files)} CSV files in {group_path}")
                
                # Load each CSV file as an ActuatorDataset
                group_datasets = []
                for csv_file in csv_files:
                    try:
                        dataset_instance = ActuatorDataset(
                            csv_file_path=csv_file,
                            inertia=group_inertia,
                            radius_accel=self.hparams.radius_accel,
                            gyro_axis_for_ang_vel=self.hparams.gyro_axis_for_ang_vel,
                            accel_axis_for_torque=self.hparams.accel_axis_for_torque,
                            target_name=self.hparams.target_name
                        )
                        group_datasets.append(dataset_instance)
                        dataset_sampling_frequencies.append(dataset_instance.get_sampling_frequency())
                        total_csv_files += 1
                        print(f"    Loaded {os.path.basename(csv_file)}: {len(dataset_instance)} sequences")
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

            if dataset_sampling_frequencies:
                self.sampling_frequency = dataset_sampling_frequencies[0]
                print(f"DataModule: Base sampling frequency: {self.sampling_frequency:.2f} Hz")
            
            print(f"DataModule: Loaded {total_csv_files} CSV files from {len(self.datasets_by_group)} inertia groups.")
            print(f"  Input dim: {self.input_dim}, Sequence length: {self.sequence_length}")

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

        for x_batch, y_batch in temp_loader:
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
    def get_sequence_length(self) -> int: return self.sequence_length
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


if __name__ == '__main__':
    print("Testing ActuatorDataModule with grouped CSV files structure...")
    
    # Create test directory structure
    test_base_dir = "test_data_dm"
    os.makedirs(test_base_dir, exist_ok=True)
    
    # Create inertia groups configuration
    inertia_groups_config = [
        {'id': 'mass_low', 'folder': 'low_inertia', 'inertia': 0.01},
        {'id': 'mass_med', 'folder': 'med_inertia', 'inertia': 0.02},
        {'id': 'mass_high', 'folder': 'high_inertia', 'inertia': 0.03}
    ]
    
    fs_main_test_dm = 100
    created_files = []
    
    try:
        # Create dummy CSV files for each group
        for group_config in inertia_groups_config:
            group_folder = os.path.join(test_base_dir, group_config['folder'])
            os.makedirs(group_folder, exist_ok=True)
            
            # Create 2-3 CSV files per group
            num_files = 2 if group_config['id'] == 'mass_low' else 3
            for file_idx in range(num_files):
                num_rows_dm = 200 + file_idx * 50
                dt_main_test_dm = 1.0 / fs_main_test_dm
                time_ms_main_test_dm = np.arange(0, num_rows_dm * dt_main_test_dm, dt_main_test_dm) * 1000
                
                inertia_factor = group_config['inertia'] * 10  # Scale for variation
                dummy_data_dm_df = pd.DataFrame({
                    'Time_ms': time_ms_main_test_dm,
                    'Encoder_Angle': 90 * np.sin(2*np.pi*(0.5+inertia_factor)*time_ms_main_test_dm/1000),
                    'Commanded_Angle': 90*np.sin(2*np.pi*(0.5+inertia_factor)*time_ms_main_test_dm/1000 + np.pi/(4+file_idx)),
                    'Acc_X': np.random.randn(num_rows_dm)*(0.5+inertia_factor),
                    'Acc_Y': np.sin(2*np.pi*(0.5+inertia_factor)*time_ms_main_test_dm/1000)*10 + np.random.randn(num_rows_dm)*0.1,
                    'Acc_Z': 9.81 + np.random.randn(num_rows_dm)*(0.5+inertia_factor),
                    'Gyro_X': np.random.randn(num_rows_dm)*(5+inertia_factor),
                    'Gyro_Y': np.random.randn(num_rows_dm)*(5+inertia_factor),
                    'Gyro_Z': (90*0.017453*(0.5+inertia_factor)*2*np.pi)*np.cos(2*np.pi*(0.5+inertia_factor)*time_ms_main_test_dm/1000)/0.017453 + np.random.randn(num_rows_dm)*0.5,
                })
                
                csv_filename = f"run_{file_idx + 1}.csv"
                csv_path = os.path.join(group_folder, csv_filename)
                dummy_data_dm_df.to_csv(csv_path, index=False)
                created_files.append(csv_path)
        
        print(f"Created test directory structure with {len(created_files)} CSV files across {len(inertia_groups_config)} groups.")

        # Test the new ActuatorDataModule
        dm_instance = ActuatorDataModule(
            data_base_dir=test_base_dir,
            inertia_groups=inertia_groups_config,
            radius_accel=0.2,
            gyro_axis_for_ang_vel='Gyro_Z', 
            accel_axis_for_torque='Acc_Y',
            batch_size=16, 
            num_workers=0,
            global_train_ratio=0.7, 
            global_val_ratio=0.15,
            seed=123
        )

        # --- Test Global Split Mode ---
        print("\n--- Testing Global Split Mode ---")
        dm_instance.setup_for_global_run()
        
        train_loader_global = dm_instance.train_dataloader()
        val_loader_global = dm_instance.val_dataloader()
        test_loader_global = dm_instance.test_dataloader()

        print(f"  Global Train loader: {len(train_loader_global.dataset) if train_loader_global else 'N/A'} samples")
        print(f"  Global Val loader: {len(val_loader_global.dataset) if val_loader_global else 'N/A'} samples")
        print(f"  Global Test loader: {len(test_loader_global.dataset) if test_loader_global else 'N/A'} samples")
        
        if train_loader_global and len(train_loader_global) > 0:
            X_batch_train_g, y_batch_train_g = next(iter(train_loader_global))
            print(f"  Global Train X batch shape: {X_batch_train_g.shape}, y batch shape: {y_batch_train_g.shape}")

        # --- Test LOMO CV Fold Mode ---
        print("\n--- Testing LOMO CV Fold Mode ---")
        num_folds_to_test = dm_instance.get_num_lomo_folds()
        
        for fold_idx_dm in range(min(2, num_folds_to_test)):  # Test first 2 folds
            print(f"\n--- Setting up LOMO Fold {fold_idx_dm + 1}/{num_folds_to_test} ---")
            dm_instance.setup_for_lomo_fold(fold_idx_dm)
            
            train_loader_lomo = dm_instance.train_dataloader()
            val_loader_lomo = dm_instance.val_dataloader()
            test_loader_lomo = dm_instance.test_dataloader()

            print(f"  LOMO Fold {fold_idx_dm + 1} Train loader: {len(train_loader_lomo.dataset) if train_loader_lomo and train_loader_lomo.dataset else 'N/A'} samples")
            print(f"  LOMO Fold {fold_idx_dm + 1} Val loader (unseen): {len(val_loader_lomo.dataset) if val_loader_lomo else 'N/A'} samples")
            print(f"  LOMO Fold {fold_idx_dm + 1} Test loader (unseen): {len(test_loader_lomo.dataset) if test_loader_lomo else 'N/A'} samples")

        print("\nActuatorDataModule with grouped CSV structure passes basic tests.")
        
    except Exception as e_dm:
        print(f"Error during ActuatorDataModule test: {e_dm}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test files and directories
        import shutil
        if os.path.exists(test_base_dir):
            shutil.rmtree(test_base_dir)
            print(f"\nCleaned up test directory: {test_base_dir}") 