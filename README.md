# Actuator Modeling with PyTorch Lightning and Hydra

This project focuses on developing and evaluating models to predict joint torque for an actuator system. It utilizes PyTorch Lightning for robust training pipelines and Hydra for flexible configuration management. The primary goal is to understand actuator dynamics and develop GRU-based models that can generalize across different physical conditions, such as varying inertias.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Core Functionality](#core-functionality)
    *   [Data Processing](#data-processing)
    *   [Model Architectures](#model-architectures)
    *   [Training and Evaluation](#training-and-evaluation)
    *   [Configuration Management](#configuration-management)
3.  [Codebase Structure](#codebase-structure)
4.  [Setup and Installation](#setup-and-installation)
5.  [Usage](#usage)
    *   [Data Preparation](#data-preparation)
    *   [Training Modes](#training-modes)
        *   [Global Evaluation Mode](#global-evaluation-mode)
        *   [Leave-One-Mass-Out Cross-Validation (LOMO CV) Mode](#leave-one-mass-out-cross-validation-lomo-cv-mode)
    *   [Running Training](#running-training)
    *   [Experiment Tracking](#experiment-tracking)
    *   [Exporting Models](#exporting-models-to-torchscript-jit)
6.  [Key Design Choices](#key-design-choices)
    *   [PyTorch Lightning](#pytorch-lightning)
    *   [Hydra for Configuration](#hydra-for-configuration)
    *   [Two-Stage Evaluation Strategy](#two-stage-evaluation-strategy)
    *   [Feature Engineering in `ActuatorDataset`](#feature-engineering-in-actuatordataset)
    *   [Residual Modeling in `ActuatorModel`](#residual-modeling-in-actuatormodel)
    *   [Normalization Strategy in `ActuatorModel`](#normalization-strategy-in-actuatormodel)
    *   [Asymmetric Torque-Speed Curve (TSC)](#asymmetric-torque-speed-curve-tsc)
    *   [Deployment Considerations](#deployment-considerations)
7.  [Future Work and Extensions](#future-work-and-extensions)

## 1. Project Overview

The project aims to predict the torque applied by an actuator based on its state variables. This involves:
*   Processing raw sensor data from actuator experiments.
*   Defining GRU neural network architectures to learn torque dynamics.
*   Implementing a flexible training and evaluation pipeline.
*   Rigorously evaluating model performance, both on data similar to training conditions and on data from unseen physical conditions (specifically, different inertias).

## 2. Core Functionality

### Data Processing

*   **`src/data/datasets.py:ActuatorDataset`**:
    *   Loads raw data from individual CSV files, each typically representing an experiment with a specific actuator mass/inertia.
    *   Performs crucial preprocessing and feature engineering:
        *   Converts time from milliseconds to seconds and determines sampling frequency.
        *   Converts angles from degrees to radians.
        *   Calculates current angular velocity from gyroscope data.
        *   Calculates the target `tau_measured` (measured torque) using the formula `inertia * angular_acceleration_from_tangential_accelerometer`.
        *   **New**: Optionally applies a low-pass Butterworth filter to the tangential acceleration signal (used to derive `tau_measured`) before torque calculation. This is controlled by `filter_cutoff_freq_hz` and `filter_order` parameters passed from the datamodule, which are configured in `configs/data/default.yaml`. The filter is applied using `scipy.signal.filtfilt`, which performs a forward and backward pass, resulting in a zero-phase (non-causal) filtering operation. This means the filtered signal has no phase distortion and the timing of events in the signal is preserved relative to the original, but it uses future data points for filtering past ones, making it suitable for offline processing like this dataset preparation.
        *   Extracts a fixed set of **3 input features** for the model from the latest timestep of a sequence: `current_angle_rad`, `target_angle_rad`, and `current_ang_vel_rad_s`.
    *   Shapes the data into sequences of a fixed length (`SEQUENCE_LENGTH = 2` by default) suitable for recurrent or time-aware models. The target torque corresponds to the state at the end of the sequence.
    *   **Important**: `ActuatorDataset` returns **raw, unnormalized** feature sequences (`x_sequence`) and target torques (`y_torque`) as PyTorch tensors.
    *   Provides static methods `get_input_dim()` (returns 3) and `get_sequence_length()` (returns 2 by default) for consistent model initialization.

*   **`src/data/datamodule.py:ActuatorDataModule`**:
    *   A PyTorch Lightning `LightningDataModule` orchestrating `ActuatorDataset`.
    *   Loads multiple CSVs per inertia group (defined by `id`, `folder`, `inertia`).
    *   **Normalization Strategy**: Computes normalization statistics (mean, std) for input features and target torque. These statistics are crucial for `ActuatorModel` and are saved by the training script (`scripts/train_model.py`) to a human-readable `normalization_stats.json` file for each run/fold. `ActuatorDataModule` itself calculates these stats based on:
        *   **Global Run Mode**: Stats from the `global_train_dataset`.
        *   **LOMO CV Fold Mode**: Stats independently from each fold's `current_fold_train_dataset` (excluding held-out data) to prevent leakage.
    *   Provides `get_input_normalization_stats()` and `get_target_normalization_stats()`.
    *   Supports two setup modes: `setup_for_global_run` and `setup_for_lomo_fold`.

### Model Architectures

*   **`src/models/gru.py:GRUModel`**: Standard GRU network.
*   **`src/models/model.py:ActuatorModel`**:
    *   The main `LightningModule`, wrapping GRU.
    *   Receives `input_dim` (3) and normalization statistics from `ActuatorDataModule`.
    *   Detailed behavior regarding residual modeling, normalization, and Torque-Speed Curve (TSC) is described in [Key Design Choices](#key-design-choices).

### Training and Evaluation

*   **`scripts/train_model.py`**: Main Hydra-driven training script.
    *   Instantiates `ActuatorDataModule` and `ActuatorModel`.
    *   Passes normalization stats from datamodule to model.
    *   Manages evaluation strategy (`global` or `lomo_cv`).

### Configuration Management

*   **Hydra (`configs/`)**:
    *   `configs/config.yaml`: Main configuration.
    *   `configs/data/default.yaml`: Configures `ActuatorDataModule`.
    *   `configs/model/default.yaml`: Configures `ActuatorModel`. Includes parameters for the physics-based model (`k_spring`, `theta0`, `kp_phys`, `kd_phys`) and, critically, the new parameters for the optional Torque-Speed Curve (TSC) applied to the PD component during training: `pd_stall_torque_phys_training` and `pd_no_load_speed_phys_training`.
    *   `configs/train/default.yaml`: Training loop settings.

## 3. Codebase Structure

```
actuator_modeling/
├── configs/                   # Hydra configuration files
│   ├── data/                  # Data module configurations
│   │   ├── datasets.py        # ActuatorDataset class
│   │   ├── datamodule.py      # ActuatorDataModule class
│   │   └── __init__.py
│   ├── models/
│   │   ├── gru.py             # GRUModel class
│   │   ├── model.py           # ActuatorModel (LightningModule)
│   │   └── __init__.py
│   ├── utils/
│   │   ├── callbacks.py       # Custom PyTorch Lightning callbacks (e.g., plotting)
│   │   └── __init__.py
│   └── __init__.py
├── .gitignore
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── setup.py                   # For installing the src package
```

## 4. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd actuator_modeling
    ```
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install the `src` package in editable mode:** This allows you to use the modules from `src` (e.g., `from src.data import ActuatorDataModule`) and have changes reflected immediately.
    ```bash
    pip install -e .
    ```

## 5. Usage

### Data Preparation

1.  **Directory Structure**: Organize your actuator data into a structured directory format where each inertia group has its own subfolder containing multiple CSV files:
    ```
    data/real/  # or your chosen base directory
    ├── 0.01kgm2/           # Folder for first inertia group
    │   ├── run_001.csv
    │   ├── run_002.csv
    │   ├── experiment_A.csv
    │   └── ...
    ├── 0.015kgm2/          # Folder for second inertia group
    │   ├── trial_1.csv
    │   ├── trial_2.csv
    │   └── ...
    └── 0.02kgm2/           # Folder for third inertia group
        ├── test_alpha.csv
        ├── test_beta.csv
        └── ...
    ```

2.  **CSV File Format**: Each CSV file should represent data from a specific experimental run. All files within the same subfolder should correspond to the same inertia value.
    *   **Required columns for `ActuatorDataset`**:
        *   `Time_ms`: Timestamp in milliseconds.
        *   `Encoder_Angle`: Current angle of the actuator (degrees).
        *   `Commanded_Angle`: Desired/target angle of the actuator (degrees).
        *   `Acc_X`, `Acc_Y`, `Acc_Z`: Accelerometer readings (m/s²). The specific axis used for torque calculation (`accel_axis_for_torque`) is configured.
        *   `Gyro_X`, `Gyro_Y`, `Gyro_Z`: Gyroscope readings (deg/s). The specific axis used for angular velocity (`gyro_axis_for_ang_vel`) is configured.

3.  **Configure `configs/data/default.yaml`**:
    *   Update `data_base_dir` to point to your base directory (e.g., `"data/real"`).
    *   Update `inertia_groups` with your inertia group configurations. Example:
        ```yaml
        inertia_groups:
          - {id: "mass_low", folder: "0.01kgm2", inertia: 0.01}
          - {id: "mass_medium", folder: "0.015kgm2", inertia: 0.015}
          - {id: "mass_high", folder: "0.02kgm2", inertia: 0.02}
        ```
    *   Set other relevant parameters like `radius_accel`, `gyro_axis_for_ang_vel`, `accel_axis_for_torque`.
    *   **New**: Configure target smoothing parameters if desired:
        *   `filter_cutoff_freq_hz`: (float, optional, default: `null`) Cutoff frequency in Hz for the Butterworth filter applied to the acceleration signal for `tau_measured`. Set to `null` or omit to disable filtering.
        *   `filter_order`: (int, default: 4) Order of the Butterworth filter if `filter_cutoff_freq_hz` is active.
    *   For "Global Evaluation Mode", configure `global_train_ratio` and `global_val_ratio`.
    *   Ensure `fallback_sampling_frequency` is set if needed (though `ActuatorModel` does not use it directly, `ActuatorDataset` does for its internal calculations).

### Training Modes

The `ActuatorDataModule` and the main training script (`scripts/train_model.py`) support two primary evaluation strategies, selectable via `evaluation_mode` in `configs/config.yaml` or command line:

#### Global Evaluation Mode (`evaluation_mode=global`)

*   **Purpose**: To assess the model's fundamental ability to learn the actuator dynamics from the entirety of the collected data. This is a good first step to ensure the model and features are viable.
*   **Mechanism**: All datasets from all inertia groups are combined. This combined dataset is then split into training, validation, and test sets according to `global_train_ratio` and `global_val_ratio`.

#### Leave-One-Mass-Out Cross-Validation (LOMO CV) Mode (`evaluation_mode=lomo_cv`)

*   **Purpose**: To rigorously evaluate the model's ability to generalize to unseen inertias.
*   **Mechanism**: The training script iterates through each inertia group. In each iteration (fold):
    *   One entire inertia group (all CSV files for that inertia) is held out as the validation set and the test set for that fold.
    *   The model is trained on datasets from all other inertia groups.
    *   Performance metrics are collected for the held-out inertia group.
    *   The final LOMO CV performance is typically the average of metrics across all folds.

### Running Training

Commands are executed from the root of the `actuator_modeling` directory.
The `scripts/train_model.py` script orchestrates the training process. It:
*   Instantiates `ActuatorDataModule`.
*   Calls the appropriate setup method on the datamodule (`setup_for_global_run` or, within a loop, `setup_for_lomo_fold`). These setup methods now also trigger the calculation of the relevant (global or fold-specific) normalization statistics within the datamodule.
*   Retrieves these normalization statistics from the datamodule.
*   Instantiates `ActuatorModel`, passing the retrieved normalization statistics to it.
*   Initializes and runs the PyTorch Lightning `Trainer`.

```bash
# Example for Global Evaluation Mode (assuming it's the default in config.yaml or set via CLI)
python scripts/train_model.py # uses evaluation_mode from config
python scripts/train_model.py evaluation_mode=global # explicit override

# Example for LOMO CV Mode
python scripts/train_model.py evaluation_mode=lomo_cv

# Override other parameters (e.g., model parameters, training epochs)
python scripts/train_model.py evaluation_mode=global model.gru_hidden_dim=256 train.max_epochs=100
```

*   Hydra will create an output directory for each run (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/` or `logs/YYYY-MM-DD/HH-MM-SS/` based on Hydra's setup), containing:
    *   `.hydra` directory with the run's specific configuration.
    *   Log files.
    *   Checkpoints saved by PyTorch Lightning (e.g., in a `checkpoints` subfolder).
    *   `training_config_summary.json`: A summary of key training parameters, including those for the physics-based model, GRU architecture, and input dimensions.
    *   `normalization_stats.json`: Normalization statistics (mean, std) for input features and the target, saved in a human-readable JSON format, preserving the feature order from `ActuatorDataset.FEATURE_NAMES`.

### Exporting Models to TorchScript (JIT)

After training, the best model checkpoint can be automatically exported to a TorchScript (`.pt`) file. This JIT-compiled model is self-contained (includes normalization) and can be useful for deployment or optimized inference.

*   **Configuration**: The export is controlled by the `train.export_jit_model` flag in your Hydra configuration.
    *   Default: `false` (as set in `configs/train/default.yaml`).
    *   To enable, set `train.export_jit_model=true` either in a config file or via command-line override.

*   **Behavior**:
    *   **Global Mode**: If enabled, the best model from the global run will be exported to the root of the Hydra run's output directory (e.g., `outputs/RUN_ID/model_name_best_jit.pt`).
    *   **LOMO CV Mode**: If enabled, for each fold, the best model from that fold will be exported to a `jit_models` subfolder within that fold's checkpoint directory (e.g., `outputs/RUN_ID/checkpoints/fold_X/jit_models/model_name_fold_X_best_jit.pt`).
    *   If WandB is active, the exported JIT model will also be logged as a WandB artifact.

*   **Example CLI Override to Enable JIT Export**:
    ```bash
    # For a global run with JIT export
    python scripts/train_model.py evaluation_mode=global train.export_jit_model=true

    # For a LOMO CV run with JIT export for each fold
    python scripts/train_model.py evaluation_mode=lomo_cv train.export_jit_model=true
    ```

### Experiment Tracking

*   The project is set up to integrate with Weights & Biases (WandB).
*   Configure WandB settings in `configs/config.yaml` (e.g., `wandb.project`, `wandb.entity`, `wandb.active`).
*   PyTorch Lightning's `WandbLogger` will automatically log metrics, hyperparameters, and potentially model checkpoints.

### Filter Effects Analysis

A dedicated script (`scripts/analyze_filter_effects.py`) is provided to visualize the effects of the Butterworth filter on accelerometer data. This tool helps you understand how the filter impacts both the time-domain signals and their frequency content.

#### Usage

```bash
# Use default filter settings from config
python scripts/analyze_filter_effects.py

# Override filter cutoff frequency
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=30.0

# Change filter order
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=50.0 data.filter_order=6

# Use a different data directory
python scripts/analyze_filter_effects.py data.data_base_dir="path/to/your/data"

# Disable filtering to see unfiltered data only
python scripts/analyze_filter_effects.py data.filter_cutoff_freq_hz=null
```

#### Generated Analysis

The script creates comprehensive plots for each processed CSV file showing:

1. **Accelerometer Data Analysis**:
   - **Timeseries comparison**: Raw vs. filtered accelerometer signals
   - **Power Spectral Density (PSD)**: Frequency content before and after filtering
   - **Difference plot**: What the filter removes from the signal

2. **Derived Torque Analysis**:
   - **Timeseries comparison**: Unfiltered vs. filtered torque calculations
   - **PSD comparison**: Frequency content of the derived torque signals
   - **Difference plot**: Impact of filtering on the final torque values

3. **Quantitative Metrics**:
   - RMS differences between filtered and unfiltered signals
   - Visual indication of the filter cutoff frequency on PSD plots

#### Output

- Individual analysis plots saved to `filter_analysis_plots/`
- Summary file (`analysis_summary.txt`) with configuration details
- Each plot shows 6 subplots arranged in a 2×3 grid for comprehensive comparison

#### Configuration

Filter parameters are configured in `configs/data/default.yaml`:
```yaml
# Filter Parameters for ActuatorDataset
filter_cutoff_freq_hz: 50.0  # Cutoff frequency in Hz. Set to null to disable filtering
filter_order: 4               # Order of the Butterworth filter
```

This analysis tool uses the same filter implementation as the training pipeline (`scipy.signal.filtfilt` with zero-phase filtering), ensuring consistency between analysis and actual model training.

## 6. Key Design Choices

*   **PyTorch Lightning**: Simplifies training code, enables multi-GPU, checkpointing, logging.
*   **Hydra for Configuration**: Flexible management of data, model, training parameters.
*   **Two-Stage Evaluation Strategy**: `global` mode for overall learning, `lomo_cv` for generalization to unseen inertias with fold-specific normalization.
*   **Feature Engineering in `ActuatorDataset`**: Centralized calculation of 3 core input features (`current_angle_rad`, `target_angle_rad`, `current_ang_vel_rad_s`) in a **defined order** as specified by `ActuatorDataset.FEATURE_NAMES`. This order is critical as it's implicitly used for calculating and applying normalization statistics and by the input layer of models, including the deployed `CombinedGRUAndPDActuator`. Raw, unnormalized data is passed to `ActuatorDataModule`.

*   **Residual Modeling in `ActuatorModel`**:
    *   Controlled by `use_residual: True` in model configuration.
    *   If true, the model learns to predict a residual to an analytical physics-based model (`tau_phys`).
    *   **`_calculate_tau_phys` Method**:
        *   Input: Uses **raw (unnormalized)** features (`current_angle_k`, `target_angle_k`, `current_ang_vel_k`) from the latest timestep of the input sequence `x_sequence`.
        *   PD Component (`tau_pd_only`): `kp_phys * error_pos_k + kd_phys * error_vel_k`.
            *   `error_pos_k = target_angle_k - current_angle_k`.
            *   `error_vel_k = 0 - current_ang_vel_k` (target angular velocity for this PD term is assumed to be zero).
        *   **Optional Torque-Speed Curve (TSC) on PD Component**: If `pd_stall_torque_phys_training` and `pd_no_load_speed_phys_training` are configured (and valid > 0), an asymmetric TSC (see [Asymmetric Torque-Speed Curve (TSC)](#asymmetric-torque-speed-curve-tsc) below) is applied *exclusively* to `tau_pd_only` via the `_apply_tsc_to_pd_component` method, resulting in `saturated_tau_pd_only`.
        *   Spring Component (`tau_spring_comp`): `k_spring * (current_angle_k - theta0)`.
        *   Final `tau_phys` (physical units): `saturated_tau_pd_only` (if TSC applied, else raw `tau_pd_only`) + `tau_spring_comp`.
    *   This `tau_phys` is then normalized for calculating the residual target.

*   **Normalization Strategy in `ActuatorModel`**:
    *   **Input Normalization**: Raw input sequences `x` from `ActuatorDataset` are normalized using `input_mean` and `input_std` (from `ActuatorDataModule`) before being fed into the neural network.
    *   **Neural Network Output**: The direct output of the neural network (`y_hat_model_output_normalized`) is always in the normalized scale.
    *   **Target Normalization for Loss**: The raw target torque `y_measured_torque_raw` is normalized to `y_target_for_loss_normalized` using `target_mean` and `target_std`.
    *   **Loss Calculation**:
        *   If `use_residual: True`:
            1.  `tau_phys_raw` is calculated (potentially with TSC on its PD part) using unnormalized `x`.
            2.  `tau_phys_raw` is normalized to `tau_phys_normalized` (using `target_mean`, `target_std`).
            3.  The learning target for the NN is `residual_normalized = y_target_for_loss_normalized - tau_phys_normalized`.
            4.  Loss is computed between `y_hat_model_output_normalized` (NN's prediction of `residual_normalized`) and `residual_normalized`.
        *   If `use_residual: False`:
            1.  The learning target for the NN is `y_target_for_loss_normalized`.
            2.  Loss is computed between `y_hat_model_output_normalized` and `y_target_for_loss_normalized`.
    *   **Denormalization for Metrics**:
        *   If `use_residual: True`, the full torque prediction in normalized scale is `y_pred_final_normalized = y_hat_model_output_normalized + tau_phys_normalized`.
        *   If `use_residual: False`, `y_pred_final_normalized = y_hat_model_output_normalized`.
        *   This `y_pred_final_normalized` is then denormalized to physical scale using `target_mean` and `target_std`.
        *   Performance metrics (MSE, RMSE, MAE, R2) are computed by comparing this denormalized prediction against the original, raw `y_measured_torque_raw`.

*   **Asymmetric Torque-Speed Curve (TSC)**:
    *   The TSC implemented (both in `ActuatorModel._apply_tsc_to_pd_component` for training and in the Isaac Lab `CombinedGRUAndPDActuator` for deployment) is asymmetric.
    *   This means that when the motor is moving:
        *   The maximum torque it can produce *in the direction of that motion* is limited by the curve: `stall_torque * (1 - |velocity| / no_load_speed)`.
        *   The maximum torque it can produce *in the direction opposing that motion* (braking) is limited by the full `stall_torque`.
    *   At zero speed, the motor can apply up to `stall_torque` in either direction.

*   **Deployment Considerations (Connection to Isaac Lab Actuator)**:
    *   The parameters configured in `ActuatorModel` for the physics-based component (`kp_phys`, `kd_phys`, `k_spring`, `theta0`), its optional TSC (`pd_stall_torque_phys_training`, `pd_no_load_speed_phys_training`), GRU architecture (`gru_num_layers`, `gru_hidden_dim`, `gru_sequence_length`), and other vital details like `use_residual` and `input_dim` are saved in the `training_config_summary.json` during the training pipeline.
    *   The Isaac Lab actuator class (e.g., `CombinedGRUAndPDActuator`) reads this `training_config_summary.json`. If `use_residual` is true in the summary, the Isaac Lab actuator replicates the same analytical base model by using these parameters from the summary. This includes applying the same TSC to its PD component *before* adding spring torque.
    *   Crucially, `CombinedGRUAndPDActuator` also loads the corresponding `normalization_stats.json` file. These statistics (input means and standard deviations) are used to normalize the input features (current angle, target angle, current angular velocity) before feeding them to its GRU component. This ensures that the data seen by the GRU at deployment time is processed in the exact same way as during training. The order of features in the `normalization_stats.json` (derived from `ActuatorDataset.FEATURE_NAMES`) must align with how `CombinedGRUAndPDActuator` constructs its input sequence for the GRU.

## 7. Future Work and Extensions

*   Implement more sophisticated model architectures (e.g., Transformers, other RNN variants).
*   Explore advanced data augmentation techniques.
*   Incorporate uncertainty estimation in torque predictions.
*   Develop more detailed analysis scripts for LOMO CV results.
*   Expand the physics-based model in `_calculate_tau_phys` to include more known effects (e.g., friction models).
*   Optimize for real-time inference if deployment is a goal (e.g., using exported ONNX models).

---

This README provides a detailed starting point. You should adapt and expand it as your project evolves. 