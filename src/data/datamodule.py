import os
from typing import Optional, List, Dict, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split # Added random_split
import pandas as pd # Keep for __main__ example
import numpy as np  # Keep for __main__ example

from src.data.datasets import ActuatorDataset # ActuatorDataset now has fixed input_dim and sequence_length

class ActuatorDataModule(pl.LightningDataModule):
    """
    LightningDataModule for actuator modeling data.
    Supports two main operational modes for evaluation:
    1. Global Split Mode: All data is combined and split into train/val/test sets.
    2. LOMO CV Fold Mode: Data is set up for a specific fold of Leave-One-Mass-Out CV.
    """

    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]], # Each dict: {'csv_file_path': str, 'inertia': float}
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
        # Manually save hyperparameters as dataset_configs can be complex
        self.save_hyperparameters() # Saves all __init__ args as hparams

        self.all_datasets_loaded: List[ActuatorDataset] = [] # Stores all loaded ActuatorDataset instances
        
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

    def _load_all_datasets_once(self):
        """Helper to load all ActuatorDataset instances if not already loaded."""
        if not self.all_datasets_loaded:
            print(f"DataModule: Lazily loading {len(self.hparams.dataset_configs)} datasets for the first time...")
            dataset_sampling_frequencies = []
            for i, config in enumerate(self.hparams.dataset_configs):
                csv_path = config.get('csv_file_path')
                inertia_val = config.get('inertia')
                if not csv_path or inertia_val is None:
                    raise ValueError(f"Dataset config {i} missing 'csv_file_path' or 'inertia'.")

                dataset_instance = ActuatorDataset(
                    csv_file_path=csv_path,
                    inertia=inertia_val,
                    radius_accel=self.hparams.radius_accel,
                    gyro_axis_for_ang_vel=self.hparams.gyro_axis_for_ang_vel,
                    accel_axis_for_torque=self.hparams.accel_axis_for_torque,
                    target_name=self.hparams.target_name
                )
                self.all_datasets_loaded.append(dataset_instance)
                dataset_sampling_frequencies.append(dataset_instance.get_sampling_frequency())
            
            if not self.all_datasets_loaded:
                raise RuntimeError("No datasets were loaded by _load_all_datasets_once.")

            if dataset_sampling_frequencies:
                self.sampling_frequency = dataset_sampling_frequencies[0]
                # print(f"DataModule: Base sampling frequency: {self.sampling_frequency:.2f} Hz") # Less verbose
                # ... (optional: check for differing fs)
            print(f"DataModule: All {len(self.all_datasets_loaded)} datasets loaded.")
            print(f"  Input dim: {self.input_dim}, Sequence length: {self.sequence_length}, Sampling Freq: {self.sampling_frequency:.2f} Hz")

    def prepare_data(self):
        # This ensures data (CSVs) are accessible.
        # Actual loading into ActuatorDataset happens in setup methods via _load_all_datasets_once.
        pass

    def setup(self, stage: Optional[str] = None):
        """
        This method is called by PyTorch Lightning.
        For this DataModule, the actual setup (global split or fold setup)
        is deferred to specific methods like `setup_for_global_run` or `setup_for_lomo_fold`.
        This main setup ensures all raw datasets are loaded once.
        """
        self._load_all_datasets_once()
        # The actual mode-specific setup will be called externally.

    def setup_for_global_run(self):
        """Sets up the DataModule for a single global train/val/test run."""
        self._load_all_datasets_once()
        self._current_mode = "global"
        print(f"DataModule: Setting up for a GLOBAL run with {len(self.all_datasets_loaded)} total datasets.")

        if not self.all_datasets_loaded:
            raise RuntimeError("No datasets available to perform a global split.")

        combined_dataset = ConcatDataset(self.all_datasets_loaded)
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

    def setup_for_lomo_fold(self, fold_index: int):
        """Sets up data for a specific LOMO CV fold. No intra-fold splitting of seen data."""
        self._load_all_datasets_once()
        self._current_mode = "lomo_fold"

        num_total_datasets = len(self.all_datasets_loaded)
        if not 0 <= fold_index < num_total_datasets:
            raise ValueError(f"LOMO fold index {fold_index} out of bounds for {num_total_datasets} datasets.")
        if num_total_datasets < 2:
             print(f"Warning: LOMO CV is being set up with only {num_total_datasets} dataset(s). Generalization testing will be limited or not meaningful.")

        print(f"\nDataModule: Setting up for LOMO CV - Fold {fold_index + 1}/{num_total_datasets}.")

        # The dataset at fold_index is for validation and testing (unseen for this fold)
        self.current_fold_val_dataset = self.all_datasets_loaded[fold_index]
        self.current_fold_test_dataset = self.all_datasets_loaded[fold_index]
        print(f"  Unseen data for Val/Test in Fold {fold_index + 1}: {self.all_datasets_loaded[fold_index].csv_file_path} ({len(self.current_fold_val_dataset)} sequences)")

        # All other datasets form the training set for this fold
        seen_pool_datasets_for_training = [ds for i, ds in enumerate(self.all_datasets_loaded) if i != fold_index]

        if not seen_pool_datasets_for_training:
            # This would happen if num_total_datasets is 1.
            self.current_fold_train_dataset = None
            print(f"  Warning: No training datasets for LOMO Fold {fold_index + 1} (total datasets: {num_total_datasets}). Training will fail if attempted.")
        else:
            self.current_fold_train_dataset = ConcatDataset(seen_pool_datasets_for_training)
            print(f"  Training data for Fold {fold_index + 1}: {len(self.current_fold_train_dataset)} sequences from {len(seen_pool_datasets_for_training)} dataset(s).")

    def train_dataloader(self) -> DataLoader:
        dataset_to_use = None
        if self._current_mode == "global":
            dataset_to_use = self.global_train_dataset
            if dataset_to_use is None: raise RuntimeError("Global train dataset not set. Call setup_for_global_run().")
        elif self._current_mode == "lomo_fold":
            dataset_to_use = self.current_fold_train_dataset
            if dataset_to_use is None: raise RuntimeError("LOMO fold train dataset not set or is empty. Call setup_for_lomo_fold(). This may occur if only 1 total dataset is available for LOMO.")
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
        self._load_all_datasets_once() # Ensure datasets are counted
        return len(self.all_datasets_loaded)


if __name__ == '__main__':
    print("Testing ActuatorDataModule with two-stage evaluation structure...")
    dummy_files_info_dm = []
    num_total_datasets_dm = 3 # Should be >= 2 for meaningful LOMO seen/unseen split test
    fs_main_test_dm = 100

    # Create dummy CSVs
    for i in range(num_total_datasets_dm):
        num_rows_dm = 300 + i * 50 
        dt_main_test_dm = 1.0 / fs_main_test_dm
        time_ms_main_test_dm = np.arange(0, num_rows_dm * dt_main_test_dm, dt_main_test_dm) * 1000
        dummy_data_dm_df = pd.DataFrame({
            'Time_ms': time_ms_main_test_dm,
            'Encoder_Angle': 90 * np.sin(2*np.pi*(0.5+i*0.1)*time_ms_main_test_dm/1000),
            'Commanded_Angle': 90*np.sin(2*np.pi*(0.5+i*0.1)*time_ms_main_test_dm/1000 + np.pi/(4+i)),
            'Acc_X': np.random.randn(num_rows_dm)*(0.5+i*0.1),
            'Acc_Y': np.sin(2*np.pi*(0.5+i*0.1)*time_ms_main_test_dm/1000)*10 + np.random.randn(num_rows_dm)*0.1,
            'Acc_Z': 9.81 + np.random.randn(num_rows_dm)*(0.5+i*0.1),
            'Gyro_X': np.random.randn(num_rows_dm)*(5+i),
            'Gyro_Y': np.random.randn(num_rows_dm)*(5+i),
            'Gyro_Z': (90*ActuatorDataset.RAD_PER_DEG*(0.5+i*0.1)*2*np.pi)*np.cos(2*np.pi*(0.5+i*0.1)*time_ms_main_test_dm/1000)/ActuatorDataset.RAD_PER_DEG + np.random.randn(num_rows_dm)*0.5,
        })
        dummy_csv_path_main_dm = f"dummy_actuator_data_mass_dm_{i}.csv"
        dummy_data_dm_df.to_csv(dummy_csv_path_main_dm, index=False)
        dummy_files_info_dm.append({'csv_file_path': dummy_csv_path_main_dm, 'inertia': 0.05 + i*0.01})
    print(f"Created {len(dummy_files_info_dm)} dummy CSV files for DataModule test.")

    dm_instance = ActuatorDataModule(
        dataset_configs=dummy_files_info_dm,
        radius_accel=0.2, # m
        gyro_axis_for_ang_vel='Gyro_Z', accel_axis_for_torque='Acc_Y',
        batch_size=16, num_workers=0,
        global_train_ratio=0.7, global_val_ratio=0.15, # Test global ratios
        seed=123
    )
    try:
        # --- Test Global Split Mode ---
        print("\n--- Testing Global Split Mode ---")
        dm_instance.setup_for_global_run() # Explicitly set up for global run
        
        train_loader_global = dm_instance.train_dataloader()
        val_loader_global = dm_instance.val_dataloader()
        test_loader_global = dm_instance.test_dataloader()

        print(f"  Global Train loader: {len(train_loader_global.dataset) if train_loader_global else 'N/A'} samples")
        print(f"  Global Val loader: {len(val_loader_global.dataset) if val_loader_global else 'N/A'} samples")
        print(f"  Global Test loader: {len(test_loader_global.dataset) if test_loader_global else 'N/A'} samples")
        
        if train_loader_global and len(train_loader_global) > 0:
            X_batch_train_g, y_batch_train_g = next(iter(train_loader_global))
            print(f"  Global Train X batch shape: {X_batch_train_g.shape}, y batch shape: {y_batch_train_g.shape}")
        if val_loader_global and len(val_loader_global) > 0:
            X_batch_val_g, y_batch_val_g = next(iter(val_loader_global))
            print(f"  Global Val X batch shape: {X_batch_val_g.shape}, y batch shape: {y_batch_val_g.shape}")
        if test_loader_global and len(test_loader_global) > 0:
            X_batch_test_g, y_batch_test_g = next(iter(test_loader_global))
            print(f"  Global Test X batch shape: {X_batch_test_g.shape}, y batch shape: {y_batch_test_g.shape}")

        # --- Test LOMO CV Fold Mode ---
        print("\n--- Testing LOMO CV Fold Mode ---")
        num_folds_to_test = dm_instance.get_num_lomo_folds()
        if num_folds_to_test == 0 : num_folds_to_test = 1 # ensure loop runs once if only 1 dataset
        
        for fold_idx_dm in range(num_folds_to_test):
            print(f"\n--- Setting up LOMO Fold {fold_idx_dm + 1}/{num_folds_to_test} ---")
            dm_instance.setup_for_lomo_fold(fold_idx_dm)
            
            train_loader_lomo = dm_instance.train_dataloader()
            val_loader_lomo = dm_instance.val_dataloader()
            test_loader_lomo = dm_instance.test_dataloader() # Test is on the unseen data

            print(f"  LOMO Fold {fold_idx_dm + 1} Train loader: {len(train_loader_lomo.dataset) if train_loader_lomo and train_loader_lomo.dataset else 'N/A'} samples")
            print(f"  LOMO Fold {fold_idx_dm + 1} Val loader (unseen): {len(val_loader_lomo.dataset) if val_loader_lomo else 'N/A'} samples")
            print(f"  LOMO Fold {fold_idx_dm + 1} Test loader (unseen): {len(test_loader_lomo.dataset) if test_loader_lomo else 'N/A'} samples")

            if train_loader_lomo and hasattr(train_loader_lomo, 'dataset') and len(train_loader_lomo.dataset) > 0 :
                 X_b_train_l, y_b_train_l = next(iter(train_loader_lomo))
                 print(f"  LOMO Train X batch shape: {X_b_train_l.shape}")
            if val_loader_lomo and len(val_loader_lomo) > 0:
                 X_b_val_l, _ = next(iter(val_loader_lomo))
                 print(f"  LOMO Val X batch shape: {X_b_val_l.shape}")
            if test_loader_lomo and len(test_loader_lomo) > 0:
                 X_b_test_l, _ = next(iter(test_loader_lomo))
                 print(f"  LOMO Test X batch shape: {X_b_test_l.shape}")


        print("\nActuatorDataModule with two-stage (global/LOMO) setup seems to pass basic checks.")
    except Exception as e_dm:
        print(f"Error during ActuatorDataModule test: {e_dm}")
        import traceback
        traceback.print_exc()
    finally:
        for info_dm in dummy_files_info_dm:
            if os.path.exists(info_dm['csv_file_path']):
                os.remove(info_dm['csv_file_path'])
        print("\nCleaned up dummy CSV files for DataModule test.") 