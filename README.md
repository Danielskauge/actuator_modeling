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
        *   Converts time from milliseconds to seconds.
        *   Converts angles from degrees to radians.
        *   Calculates angular velocities and accelerations using numerical differentiation (`np.gradient`) if not directly available or to ensure consistency.
        *   Calculates the target `tau_measured` (measured torque) using the formula `inertia * angular_acceleration_from_tangential_accelerometer`.
        *   Extracts a fixed set of input features for the model: `current_angle_rad`, `target_angle_rad`, `current_ang_vel_rad_s`, `target_ang_vel_rad_s`.
    *   Shapes the data into sequences of a fixed length (`SEQUENCE_LENGTH = 2` by default) suitable for recurrent or time-aware models. The target torque corresponds to the state at the end of the sequence.
    *   Provides static methods `get_input_dim()` and `get_sequence_length()` for consistent model initialization.

*   **`src/data/datamodule.py:ActuatorDataModule`**:
    *   A PyTorch Lightning `LightningDataModule` that orchestrates the use of `ActuatorDataset`.
    *   Manages multiple `ActuatorDataset` instances, one for each CSV/inertia defined in the configuration.
    *   Supports two primary setup modes for evaluation (controlled by the main training script):
        1.  **Global Run Mode (`setup_for_global_run`)**: Combines all individual datasets and performs a single train/validation/test split on this aggregated data. This is used to assess overall model learning capability.
        2.  **LOMO CV Fold Mode (`setup_for_lomo_fold`)**: Sets up data for a specific fold in Leave-One-Mass-Out Cross-Validation. In each fold, one dataset (mass) is held out as the validation and test set (unseen inertia), while the remaining datasets are used for training. This rigorously tests generalization to new inertias.
    *   Provides the necessary `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` based on the active mode.

### Model Architectures

*   **`src/models/gru.py:GRUModel`**:
    *   Implements a Gated Recurrent Unit (GRU) network.
    *   Configurable `input_dim`, `hidden_dim`, `num_layers`, `output_dim`, and `dropout`.
    *   Uses `batch_first=True` and processes sequences, taking the GRU output from the last timestep for prediction.
*   **`src/models/mlp.py:MLP`**: (If used, or as a reference for other architectures)
    *   A standard Multi-Layer Perceptron.
    *   Configurable hidden layer dimensions, activation function, dropout, and batch normalization.
*   **`src/models/model.py:ActuatorModel`**:
    *   The main `LightningModule` that wraps the chosen neural network (e.g., `GRUModel`).
    *   Handles the training, validation, and test steps.
    *   Calculates loss (MSE by default, with an optional first-difference torque smoothing term).
    *   Computes and logs various metrics (MSE, RMSE, MAE, R2).
    *   **Residual Modeling**: Can be configured to predict either the full torque or a residual torque (`tau_measured - tau_phys`).
        *   `_calculate_tau_phys`: Calculates a physics-based torque component using provided spring constant (`k_spring`), equilibrium angle (`theta0`), and PD gains (`kp_phys`, `kd_phys`). This component is subtracted from the target if `use_residual=True`.
    *   Implements optimizer (AdamW) and learning rate scheduler (linear warmup followed by cosine decay).

### Training and Evaluation

*   **`scripts/train_model.py`** (Conceptual - you will create or adapt this):
    *   The main script for initiating training.
    *   Uses Hydra to load configurations for data, model, training parameters, logging, etc.
    *   Instantiates `ActuatorDataModule` and `ActuatorModel`.
    *   Instantiates PyTorch Lightning `Trainer` with callbacks (e.g., for checkpointing, logging).
    *   Manages the selected evaluation strategy:
        *   If "global mode": Calls `datamodule.setup_for_global_run()` and then `trainer.fit()` and `trainer.test()`.
        *   If "LOMO CV mode": Loops through each dataset/mass, calls `datamodule.setup_for_lomo_fold(fold_idx)`, then `trainer.fit()` and `trainer.test()` for that fold. Aggregates LOMO CV results.

### Configuration Management

*   **Hydra (`configs/`)**:
    *   The project heavily relies on Hydra for managing configurations.
    *   `configs/config.yaml`: Main configuration file, sets defaults and can include other config files.
    *   `configs/data/default.yaml`: Configures `ActuatorDataModule`, including paths to CSVs, inertias, and splitting ratios for global mode.
    *   `configs/model/default.yaml` (and e.g., `configs/model/gru.yaml`): Configures `ActuatorModel` and the specific network (GRU/MLP) parameters, including `model_type`, dimensions, `use_residual`, and physics parameters.
    *   `configs/train/default.yaml`: Configures training parameters like epochs, learning rate, batch size, accelerator, etc.
    *   Allows overriding any configuration parameter via the command line.

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
│   └── train_model.py         # Main training script (to be developed based on existing structure)
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

1.  **CSV Files**: Prepare your actuator data in CSV format. Each file should represent data from a specific experimental condition (e.g., a particular inertia).
    *   **Required columns for `ActuatorDataset`**:
        *   `Time_ms`: Timestamp in milliseconds.
        *   `Encoder_Angle`: Current angle of the actuator (degrees).
        *   `Commanded_Angle`: Desired/target angle of the actuator (degrees).
        *   `Acc_X`, `Acc_Y`, `Acc_Z`: Accelerometer readings (m/s²). The specific axis used for torque calculation (`accel_axis_for_torque`) is configured.
        *   `Gyro_X`, `Gyro_Y`, `Gyro_Z`: Gyroscope readings (deg/s). The specific axis used for angular velocity (`gyro_axis_for_ang_vel`) is configured.
    *   Place these CSV files in a directory (e.g., `data/my_experiment_csvs/`).

2.  **Configure `configs/data/default.yaml`**:
    *   Update `dataset_configs` with a list of your CSV files and their corresponding `inertia` values. Example:
        ```yaml
        dataset_configs:
          - {csv_file_path: "data/my_experiment_csvs/run_inertia_A.csv", inertia: 0.01}
          - {csv_file_path: "data/my_experiment_csvs/run_inertia_B.csv", inertia: 0.015}
          # ... and so on for all your datasets
        ```
    *   Set other relevant parameters like `radius_accel`, `gyro_axis_for_ang_vel`, `accel_axis_for_torque`.
    *   For "Global Evaluation Mode", configure `global_train_ratio` and `global_val_ratio`.

### Training Modes

The `ActuatorDataModule` and the main training script are designed to support two primary evaluation strategies:

#### Global Evaluation Mode

*   **Purpose**: To assess the model's fundamental ability to learn the actuator dynamics from the entirety of the collected data. This is a good first step to ensure the model and features are viable.
*   **Mechanism**: All datasets listed in `dataset_configs` are combined. This combined dataset is then split into training, validation, and test sets according to `global_train_ratio` and `global_val_ratio`.
*   **How to run**: Your `scripts/train_model.py` will need a way to trigger this mode, likely via a Hydra configuration parameter (e.g., `evaluation_mode=global`).

#### Leave-One-Mass-Out Cross-Validation (LOMO CV) Mode

*   **Purpose**: To rigorously evaluate the model's ability to generalize to unseen inertias.
*   **Mechanism**: The training script iterates through each dataset. In each iteration (fold):
    *   One dataset (representing one inertia) is held out as the validation set and the test set for that fold.
    *   The model is trained on all other N-1 datasets.
    *   Performance metrics are collected for the held-out dataset.
    *   The final LOMO CV performance is typically the average of metrics across all folds.
*   **How to run**: Your `scripts/train_model.py` will need to trigger this mode (e.g., `evaluation_mode=lomo_cv`). The script will then call `datamodule.setup_for_lomo_fold(fold_idx)` in a loop.

### Running Training

The exact command will depend on your `scripts/train_model.py` implementation, but it will generally look like this (using Hydra):

```bash
# Example for Global Evaluation Mode
python scripts/train_model.py evaluation_mode=global # Add other overrides as needed

# Example for LOMO CV Mode
python scripts/train_model.py evaluation_mode=lomo_cv

# Override other parameters
python scripts/train_model.py evaluation_mode=global model.gru_hidden_dim=256 train.max_epochs=100
```

*   Hydra will create an output directory for each run (usually in `logs/` or `outputs/` based on Hydra's setup), containing:
    *   `.hydra` directory with the run's configuration.
    *   Log files.
    *   Checkpoints saved by PyTorch Lightning.

### Experiment Tracking

*   The project is set up to integrate with experiment tracking tools like Weights & Biases (WandB).
*   Configure WandB settings in `configs/config.yaml` (e.g., `wandb.project`, `wandb.entity`).
*   PyTorch Lightning's `WandbLogger` will automatically log metrics, hyperparameters, and potentially model checkpoints.

## 6. Key Design Choices

*   **PyTorch Lightning**: Chosen to simplify boilerplate training code, enable easy multi-GPU training, provide robust checkpointing, and integrate well with logging and callbacks.
*   **Hydra for Configuration**: Provides a highly flexible and powerful way to manage all aspects of the project (data, model, training, logging) through YAML files and command-line overrides. This facilitates experimentation and reproducibility.
*   **Two-Stage Evaluation Strategy**:
    1.  **Global Mode**: First, verify the model can learn from the entire dataset distribution.
    2.  **LOMO CV Mode**: Second, specifically test generalization to unseen inertias, which is critical for real-world applicability. This separation allows for a more structured approach to model development.
*   **Feature Engineering in `ActuatorDataset`**: Centralizing feature calculation (angles to radians, derivatives, `tau_measured`) within the dataset class ensures consistency and makes the data loading process cleaner.
*   **Residual Modeling (`ActuatorModel`)**: The option to predict a residual torque (actual torque minus a physics-based model component) can sometimes help the model focus on learning the more complex, unmodeled dynamics. Parameters for the physics-based component (`k_spring`, `theta0`, `kp_phys`, `kd_phys`) are configurable.

## 7. Future Work and Extensions

*   Implement more sophisticated model architectures (e.g., Transformers, other RNN variants).
*   Explore advanced data augmentation techniques.
*   Incorporate uncertainty estimation in torque predictions.
*   Develop more detailed analysis scripts for LOMO CV results.
*   Expand the physics-based model in `_calculate_tau_phys` to include more known effects (e.g., friction models).
*   Optimize for real-time inference if deployment is a goal (e.g., using exported ONNX models).

---

This README provides a detailed starting point. You should adapt and expand it as your project evolves, particularly the "Usage" section once `scripts/train_model.py` is finalized. 