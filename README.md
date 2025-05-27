# Actuator Modeling with PyTorch Lightning

This project implements an MLP (Multi-Layer Perceptron) model for actuator torque prediction using PyTorch Lightning. The model predicts torque based on current angle, desired angle, and their derivatives.

## Project Structure

```
project_root/
├── data/                      # Raw and processed data
├── logs/                      # Training logs
├── models/                    # Saved models
│   └── checkpoints/           # Model checkpoints
├── src/                       # Source code
│   ├── data/                  # Data processing
│   ├── models/                # Model definitions
│   └── utils/                 # Utilities
├── scripts/                   # Training and evaluation scripts
├── configs/                   # Configuration files
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

## Features

- **PyTorch Lightning**: Clean, scalable deep learning code
- **Multi-GPU Training**: Support for training on multiple GPUs
- **Model Export**: Export to TorchScript and ONNX for deployment
- **Visualization**: Visualize model predictions during training
- **Configurable**: YAML-based configuration for easy experimentation

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd actuator_modeling
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Data Format

The raw data should be a CSV file with the following columns:
- `timestamp`: Time of the measurement
- `current_angle`: Current angle of the actuator
- `desired_angle`: Desired angle of the actuator

Place your data file in the `data` directory with the name `actuator_data.csv` or specify a different path in the configuration.

## Usage

### Training

To train the model with default configuration:

```bash
python scripts/train.py --config configs/train/default.yaml
```

You can customize training by modifying the YAML configuration files or by passing command-line arguments:

```bash
python scripts/train.py --config configs/train/default.yaml --max_epochs 200 --gpus 2
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py <path_to_checkpoint> --data_dir data
```

### Export

To export a trained model for deployment:

```bash
python scripts/export.py <path_to_checkpoint> --test_speed
```

## Configuration

The project uses YAML configuration files for easy customization:

- **Model Configuration**: `configs/model/mlp.yaml`
- **Data Configuration**: `configs/data/actuator_data.yaml`
- **Training Configuration**: `configs/train/default.yaml`

## Multi-GPU Training

The project supports training on multiple GPUs using PyTorch Lightning's DDP (Distributed Data Parallel) strategy. To train on all available GPUs:

```bash
python scripts/train.py --gpus -1
```

To train on a specific number of GPUs:

```bash
python scripts/train.py --gpus 2
```

## Model Architecture

The MLP model consists of configurable fully-connected layers with optional batch normalization, dropout, and residual connections. The default architecture has 5 hidden layers with sizes [64, 128, 256, 128, 64].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 