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
    *   [Stateful GRU for Time-Series Modeling](#stateful-gru-for-time-series-modeling)
    *   [Two-Stage Evaluation Strategy](#two-stage-evaluation-strategy)
    *   [Feature Engineering in `ActuatorDataset`](#feature-engineering-in-actuatordataset)
    *   [Residual Modeling in `ActuatorModel`](#residual-modeling-in-actuatormodel)
    *   [Normalization Strategy](#normalization-strategy)
    *   [Asymmetric Torque-Speed Curve (TSC)](#asymmetric-torque-speed-curve-tsc)
    *   [Deployment to Isaac Lab Actuator](#deployment-to-isaac-lab-actuator)
7.  [Future Work and Extensions](#future-work-and-extensions)

## 1. Project Overview

The project aims to predict the torque applied by an actuator based on its state variables. This involves:
*   Processing raw sensor data from actuator experiments.
*   Defining stateful GRU neural network architectures to learn torque dynamics over time.
*   Implementing a flexible training and evaluation pipeline.
*   Rigorously evaluating model performance, both on data similar to training conditions and on data from unseen physical conditions (specifically, different inertias).

## 2. Core Functionality

### Data Processing

*   **`src/data/datasets.py:ActuatorDataset`**:
    *   Loads raw data from individual CSV files.
    *   Performs crucial preprocessing and feature engineering:
        *   Calculates sampling frequency, converts units.
        *   Calculates `tau_measured` (target torque).
        *   Optionally applies a Butterworth filter to the acceleration signal used for `tau_measured`.
        *   Extracts **3 input features**: `current_angle_rad`, `target_angle_rad`, `current_ang_vel_rad_s`.
    *   Shapes data into sequences. The length of these sequences (`sequence_length_timesteps`) is now configurable via `sequence_duration_s` (e.g., 1.0 seconds) and the dataset's calculated `sampling_frequency`.
    *   Returns raw, unnormalized feature sequences and target torques.
    *   Provides `get_input_dim()` (static) and `get_sequence_length_timesteps()` (instance method).

*   **`src/data/datamodule.py:ActuatorDataModule`**:
    *   Orchestrates `ActuatorDataset` instances.
    *   Configurable via `sequence_duration_s` in `configs/data/default.yaml`, which is passed to `ActuatorDataset`.
    *   Determines and stores `sequence_length_timesteps` and `sampling_frequency` from the first loaded dataset instance.
    *   Computes and provides normalization statistics (mean, std) for inputs and targets, crucial for the `ActuatorModel`. These are saved to `normalization_stats.json`.

### Model Architectures

*   **`src/models/gru.py:GRUModel`**:
    *   A standard `nn.GRU` wrapped in an `nn.Module`.
    *   **Stateful Forward Pass**: Its `forward` method is now `forward(x, h_prev=None) -> (prediction, h_next)`.
        *   `x`: Input sequence tensor `(batch, seq_len, features)`.
        *   `h_prev`: Optional previous hidden state `(num_layers, batch, hidden_dim)`. If `None`, it's initialized to zeros.
        *   Returns `prediction` `(batch, output_dim)` (from the last time step of the sequence) and `h_next` `(num_layers, batch, hidden_dim)`.

*   **`src/models/model.py:ActuatorModel`**:
    *   The main `LightningModule`, wrapping `GRUModel`.
    *   **Stateful Forward Pass**: Mirrors `GRUModel`'s stateful signature: `forward(x, h_prev=None) -> (prediction, h_next)`.
    *   **Training**: During `training_step`, `validation_step`, and `test_step`:
        *   The `forward` method is called with `h_prev=None` (or the hidden state is implicitly managed by `nn.GRU` if the input `x` is a sequence). The returned `h_next` is typically ignored for loss calculation within these steps, as state propagation *between batches* is not standard. The stateful signature is primarily for ensuring the JIT-exported model can be used statefully during inference.
    *   Detailed behavior regarding residual modeling, normalization, and Torque-Speed Curve (TSC) is described in [Key Design Choices](#key-design-choices).

### Training and Evaluation

*   **`scripts/train_model.py`**: Main Hydra-driven training script.
    *   Instantiates `ActuatorDataModule` and `ActuatorModel`.
    *   Retrieves `sequence_length_timesteps` from the datamodule for logging purposes (e.g., in `training_config_summary.json`).
    *   Passes normalization stats from datamodule to the model.
    *   Manages evaluation strategy (`global` or `lomo_cv`).
    *   **JIT Export**: Exports the trained `ActuatorModel` to TorchScript. The exported model will have the stateful `forward(x, h_prev)` signature, making it suitable for continuous, stateful inference.

### Configuration Management

*   **Hydra (`configs/`)**:
    *   `configs/data/default.yaml`: Configures `ActuatorDataModule`, including the new `sequence_duration_s` parameter.
    *   `configs/model/default.yaml`: Configures `ActuatorModel`.
    *   The `training_config_summary.json` output now includes `gru_sequence_length_timesteps`.

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
    *   **New/Updated**: Configure `sequence_duration_s` (e.g., `1.0` for 1-second sequences). The actual number of timesteps per sequence will be `sequence_duration_s * sampling_frequency`.

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

## 6. Key Design Choices

*   **PyTorch Lightning**: Simplifies training code, enables multi-GPU, checkpointing, logging.
*   **Hydra for Configuration**: Flexible management of data, model, training parameters.
*   **Stateful GRU for Time-Series Modeling**:
    *   The core `GRUModel` and the encapsulating `ActuatorModel` are designed to be stateful. Their `forward` methods accept a previous hidden state and return the prediction along with the next hidden state: `(prediction, h_next) = model.forward(x, h_prev)`.
    *   **Training**: For training, sequences of `sequence_length_timesteps` (e.g., 1 second of data) are fed to the model. The `nn.GRU` layer internally propagates the hidden state across the timesteps *within* that sequence. The explicit `h_prev` argument to `ActuatorModel.forward` is typically `None` at the start of processing a batch/sequence during training steps, relying on the `nn.GRU`'s internal handling or default zero initialization.
    *   **Inference**: The JIT-exported model retains this stateful signature. This allows external components, like the `CombinedGRUAndPDActuator` in Isaac Lab, to manage the hidden state explicitly from one inference call (e.g., one simulation step) to the next, enabling continuous memory.
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

*   **Normalization Strategy**:
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

*   **Deployment to Isaac Lab Actuator (`CombinedGRUAndPDActuator`)**:
    *   The `CombinedGRUAndPDActuator` is designed to work with the JIT-exported stateful `ActuatorModel`.
    *   **Stateful Inference Loop**:
        1.  At each simulation step, the actuator prepares a **single time-step input** for the GRU: `current_step_input` of shape `(batch_size_effective, 1, num_input_features)`. `batch_size_effective` is typically `num_envs * num_joints`.
        2.  This `current_step_input` is normalized using the loaded `input_mean` and `input_std` from `normalization_stats.json`.
        3.  The normalized input is passed to the JIT model along with the currently stored hidden state: `prediction_normalized, h_next = jit_model(normalized_input, self.sea_hidden_state)`.
        4.  The actuator updates its stored hidden state: `self.sea_hidden_state = h_next`.
        5.  The `prediction_normalized` is then denormalized (if `use_residual=False`) or used to compute the total torque (if `use_residual=True` by adding to the normalized `tau_phys`).
    *   **Configuration Files**:
        *   `training_config_summary.json`: Still critical. It provides `gru_num_layers`, `gru_hidden_dim`, parameters for the residual physics model (`kp_phys`, `kd_phys`, etc., if `use_residual=True`), and `gru_sequence_length_timesteps` (which informs about the training context of the hidden state).
        *   `normalization_stats.json`: Essential for normalizing the single time-step inputs at inference time to match the training distribution. The order of features defined in `ActuatorDataset.FEATURE_NAMES` and reflected in this file must be strictly adhered to by the actuator when constructing its input.
    *   The `gru_sequence_length` parameter in `CombinedGRUAndPDActuatorCfg` now serves primarily as a reference to the sequence length the model was trained with, ensuring consistency in understanding the model's architecture, rather than dictating the input shape for per-step inference (which is always 1).

## 7. Future Work and Extensions

*   Implement more sophisticated model architectures (e.g., Transformers, other RNN variants).
*   Explore advanced data augmentation techniques.
*   Incorporate uncertainty estimation in torque predictions.
*   Develop more detailed analysis scripts for LOMO CV results.
*   Expand the physics-based model in `_calculate_tau_phys` to include more known effects (e.g., friction models).
*   Optimize for real-time inference if deployment is a goal (e.g., using exported ONNX models).

---

This README provides a detailed starting point. You should adapt and expand it as your project evolves. 