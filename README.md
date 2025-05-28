# Actuator Modeling with PyTorch Lightning and Hydra

This project focuses on developing and evaluating models to predict joint torque for an actuator system. It utilizes PyTorch Lightning for robust training pipelines and Hydra for flexible configuration management. The primary goal is to understand actuator dynamics and develop models that can generalize across different physical conditions, such as varying inertias.

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
6.  [Key Design Choices](#key-design-choices)
    *   [PyTorch Lightning](#pytorch-lightning)
    *   [Hydra for Configuration](#hydra-for-configuration)
    *   [Two-Stage Evaluation Strategy](#two-stage-evaluation-strategy)
    *   [Feature Engineering in `ActuatorDataset`](#feature-engineering-in-actuatordataset)
    *   [Residual Modeling](#residual-modeling)
7.  [Future Work and Extensions](#future-work-and-extensions)

## 1. Project Overview

The project aims to predict the torque applied by an actuator based on its state variables. This involves:
*   Processing raw sensor data from actuator experiments.
*   Defining neural network architectures (initially GRU, adaptable to MLP and others) to learn torque dynamics.
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
        *   Extracts a fixed set of **3 input features** for the model: `current_angle_rad`, `target_angle_rad`, and `current_ang_vel_rad_s`. The explicit `target_ang_vel_rad_s` feature has been removed, relying on the sequence of target angles to implicitly convey this information to the neural network.
    *   Shapes the data into sequences of a fixed length (`SEQUENCE_LENGTH = 2` by default) suitable for recurrent or time-aware models. The target torque corresponds to the state at the end of the sequence.
    *   **Important**: `ActuatorDataset` returns **raw, unnormalized** feature sequences (`x_sequence`) and target torques (`y_torque`) as PyTorch tensors. It does not perform statistical normalization (e.g., z-scoring).
    *   Provides static methods `get_input_dim()` (which will return 3) and `get_sequence_length()` for consistent model initialization, and `get_sampling_frequency()` (though this is not directly used by `ActuatorModel` anymore).

*   **`src/data/datamodule.py:ActuatorDataModule`**:
    *   A PyTorch Lightning `LightningDataModule` that orchestrates the use of `ActuatorDataset`.
    *   Supports loading multiple CSV files per inertia group from organized subfolders. Each inertia group is defined by:
        *   `id`: A unique identifier for the group
        *   `folder`: The subfolder name under the base data directory
        *   `inertia`: The numerical inertia value for all CSV files in this group
    *   Automatically discovers and loads all `*.csv` files within each group's subfolder.
    *   **Normalization Strategy**: `ActuatorDataModule` is responsible for calculating and providing the necessary normalization statistics (mean and standard deviation) for both input features and the target torque. This is handled differently based on the evaluation mode:
        *   **Global Run Mode (`setup_for_global_run`)**:
            *   After combining all datasets and performing a single train/validation/test split on this aggregated data, normalization statistics are computed **exclusively from the `global_train_dataset`**.
            *   These global statistics are stored within the datamodule instance.
        *   **LOMO CV Fold Mode (`setup_for_lomo_fold`)**:
            *   For each fold, one entire inertia group is held out as the validation and test set.
            *   Normalization statistics are computed **independently for each fold, using only the `current_fold_train_dataset`** (i.e., data from all inertia groups *except* the one held out for that fold). This ensures that no information from the fold's validation/test set influences the normalization process, preventing data leakage.
            *   These fold-specific statistics are stored within the datamodule instance for the current fold.
    *   Provides accessor methods (`get_input_normalization_stats()`, `get_target_normalization_stats()`) for the training script to retrieve the relevant statistics.
    *   Supports two primary setup modes for evaluation (controlled by the main training script via `cfg.evaluation_mode`):
        1.  **Global Run Mode (`setup_for_global_run`)**: Combines all individual datasets from all groups and performs a single train/validation/test split on this aggregated data. This is used to assess overall model learning capability. Normalization stats are derived from this mode's training split.
        2.  **LOMO CV Fold Mode (`setup_for_lomo_fold`)**: Sets up data for a specific fold in Leave-One-Mass-Out Cross-Validation. In each fold, one entire inertia group (all CSV files for that inertia) is held out as the validation and test set (unseen inertia), while datasets from all other groups are used for training. Normalization stats are derived from this fold's specific training set. This rigorously tests generalization to new inertias.
    *   Provides the necessary `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` based on the active mode.

### Model Architectures

*   **`src/models/gru.py:GRUModel`**:
    *   Implements a Gated Recurrent Unit (GRU) network.
    *   Configurable `input_dim` (which will be 3 per timestep), `hidden_dim`, `num_layers`, `output_dim`, and `dropout`.
    *   Uses `batch_first=True` and processes sequences, taking the GRU output from the last timestep for prediction.
*   **`src/models/mlp.py:MLP`**: (If used, or as a reference for other architectures)
    *   A standard Multi-Layer Perceptron. Input dimension will be `input_dim * sequence_length` (e.g., 3 * 2 = 6).
    *   Configurable hidden layer dimensions, activation function, dropout, and batch normalization.
*   **`src/models/model.py:ActuatorModel`**:
    *   The main `LightningModule` that wraps the chosen neural network (e.g., `GRUModel`).
    *   **Dynamically configured `input_dim` (now 3)** is passed during instantiation from the `ActuatorDataModule` via the training script.
    *   **Normalization Handling**:
        *   Accepts pre-calculated `input_mean`, `input_std`, `target_mean`, and `target_std` during initialization. These statistics are provided by the `ActuatorDataModule` (and are specific to the global training set or the current LOMO CV fold's training set) and are registered as buffers within the model.
        *   **Input Normalization**: In its `step` method, raw input sequences `x` (from `ActuatorDataset`) are normalized using the provided `input_mean` and `input_std` before being fed into the underlying neural network (MLP or GRU).
        *   **Target Normalization for Loss**: The raw target torque `y_measured_torque` is also normalized using `target_mean` and `target_std`. The loss function is computed against this normalized target.
        *   **Residual Modeling with Normalization**:
            *   If `use_residual: True`, the `_calculate_tau_phys` method computes the physics-based torque component using the **original, raw (unnormalized)** input features `x`.
            *   The resulting `tau_phys` (in physical units) is then **normalized** using `target_mean` and `target_std`.
            *   The model is trained to predict the difference between the *normalized* `y_measured_torque` and this *normalized* `tau_phys`. The model's direct output is thus a normalized residual.
        *   **Denormalization for Metrics**: For calculating performance metrics (MSE, RMSE, MAE, R2), the model's final prediction (which is in the normalized scale, and includes the re-addition of the *normalized* `tau_phys` if in residual mode) is **denormalized** back to the original physical scale using `target_mean` and `target_std`. Metrics are then computed by comparing this denormalized prediction against the original, raw `y_measured_torque`. This ensures metrics are interpretable in their physical units.
    *   Handles the training, validation, and test steps.
    *   Calculates loss (MSE by default, with an optional first-difference torque smoothing term) on normalized values.
    *   Computes and logs various metrics (MSE, RMSE, MAE, R2) on denormalized, physical-scale values.
    *   Implements optimizer (AdamW) and learning rate scheduler (linear warmup followed by cosine decay).

### Training and Evaluation

*   **`scripts/train_model.py`**:
    *   The main script for initiating training using Hydra.
    *   Instantiates `ActuatorDataModule` and `ActuatorModel`, passing the dynamic `input_dim` (3) from the datamodule to the model.
    *   Instantiates PyTorch Lightning `Trainer` with callbacks.
    *   Manages the selected evaluation strategy (`global` or `lomo_cv`) based on `cfg.evaluation_mode`.

### Configuration Management

*   **Hydra (`configs/`)**:
    *   `configs/config.yaml`: Main configuration. Includes `evaluation_mode` to switch between global and LOMO CV runs.
    *   `configs/data/default.yaml`: Configures `ActuatorDataModule` with the new grouped structure. Includes `data_base_dir` (base directory path) and `inertia_groups` (list of group configurations). `input_dim` is now implicitly 3 due to `ActuatorDataset` changes. Includes `fallback_sampling_frequency` (though `ActuatorModel` no longer uses it directly).
    *   `configs/model/default.yaml`: Configures `ActuatorModel`. `input_dim` is no longer set here but passed dynamically. `sampling_frequency` is not used.
    *   `configs/train/default.yaml`: Includes `callbacks` section to toggle common callbacks and settings for `gradient_clip_val` and `early_stopping.active`.

## 3. Codebase Structure

```
actuator_modeling/
├── configs/                   # Hydra configuration files
│   ├── data/                  # Data module configurations
│   ├── model/                 # Model configurations (GRU, MLP, etc.)
│   └── train/                 # Training loop configurations
│   └── config.yaml            # Main Hydra configuration
├── data/                      # Placeholder for raw and processed data
│   └── synthetic_raw/         # Example: Directory for raw synthetic CSVs
├── logs/                      # Output directory for Hydra runs (contains logs, checkpoints)
├── models/                    # (Potentially for manually saved/exported models, distinct from Hydra logs)
├── scripts/                   # Executable scripts
│   └── train_model.py         # Main training script
│   └── generate_synthetic_csv.py # Utility for generating synthetic data (example)
├── src/                       # Source code for the actuator_modeling package
│   ├── data/
│   │   ├── datasets.py        # ActuatorDataset class
│   │   ├── datamodule.py      # ActuatorDataModule class
│   │   └── __init__.py
│   ├── models/
│   │   ├── gru.py             # GRUModel class
│   │   ├── mlp.py             # MLPModel class (if used)
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

# Change model type for a run
python scripts/train_model.py model.model_type=mlp # Ensure mlp specific params are well-defined in model config
```

*   Hydra will create an output directory for each run (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/` or `logs/YYYY-MM-DD/HH-MM-SS/` based on Hydra's setup), containing:
    *   `.hydra` directory with the run's specific configuration.
    *   Log files.
    *   Checkpoints saved by PyTorch Lightning (e.g., in a `checkpoints` subfolder).

### Experiment Tracking

*   The project is set up to integrate with Weights & Biases (WandB).
*   Configure WandB settings in `configs/config.yaml` (e.g., `wandb.project`, `wandb.entity`, `wandb.active`).
*   PyTorch Lightning's `WandbLogger` will automatically log metrics, hyperparameters, and potentially model checkpoints.

## 6. Key Design Choices

*   **PyTorch Lightning**: Chosen to simplify boilerplate training code, enable easy multi-GPU training, provide robust checkpointing, and integrate well with logging and callbacks.
*   **Hydra for Configuration**: Provides a highly flexible and powerful way to manage all aspects of the project (data, model, training, logging) through YAML files and command-line overrides. This facilitates experimentation and reproducibility.
*   **Two-Stage Evaluation Strategy**: Controlled by `evaluation_mode`.
    1.  **Global Mode**: First, verify the model can learn from the entire dataset distribution. Normalization statistics are derived from the training split of this global dataset.
    2.  **LOMO CV Mode**: Second, specifically test generalization to unseen inertias, which is critical for real-world applicability. Normalization statistics are derived *independently for each fold* from that fold's specific training data (excluding the held-out inertia). This separation allows for a more structured approach to model development and ensures valid generalization assessment.
*   **Input and Target Normalization**:
    *   Input features (after initial unit conversions by `ActuatorDataset`) and target torque values are statistically normalized (z-scored) before being used by the neural network for training.
    *   The normalization statistics (mean and standard deviation) are calculated by `ActuatorDataModule`, ensuring they are derived only from the appropriate training set (either the global training set or the fold-specific training set in LOMO CV).
    *   `ActuatorModel` performs the normalization of inputs and targets, and denormalizes its output predictions before metric calculation, ensuring metrics are in interpretable physical units. This is crucial for stable training and proper model evaluation.
*   **Feature Engineering in `ActuatorDataset`**: Centralizing feature calculation (angles to radians, derivatives, `tau_measured`) within the dataset class ensures consistency. The model now uses 3 core input features per timestep (`current_angle_rad`, `target_angle_rad`, `current_ang_vel_rad_s`).
*   **Residual Modeling (`ActuatorModel`)**: The option to predict a residual torque (actual torque minus a physics-based model component) can sometimes help the model focus on learning the more complex, unmodeled dynamics. For the physics-based component, the target angular velocity (for the `kd_phys` term) is assumed to be zero. The normalization strategy correctly handles the scales when combining learned residuals with the physics-based component.

## 7. Future Work and Extensions

*   Implement more sophisticated model architectures (e.g., Transformers, other RNN variants).
*   Explore advanced data augmentation techniques.
*   Incorporate uncertainty estimation in torque predictions.
*   Develop more detailed analysis scripts for LOMO CV results.
*   Expand the physics-based model in `_calculate_tau_phys` to include more known effects (e.g., friction models).
*   Optimize for real-time inference if deployment is a goal (e.g., using exported ONNX models).

---

This README provides a detailed starting point. You should adapt and expand it as your project evolves. 