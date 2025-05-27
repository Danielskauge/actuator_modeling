#!/usr/bin/env python

import os
import argparse
from typing import Dict, List, Tuple

import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.datamodule import ActuatorDataModule
from src.models.model import ActuatorModel
from src.utils.callbacks import (
    ModelExporter,
    EarlySummary,
    TestPredictionPlotter,
)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration from file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(
    config_path: str = None,
    data_file: str = "data/synthetic/joint_data/joint-data-1.csv",
    max_epochs: int = 100,
    gpu: int = -1,  # Use all available GPUs by default
    precision: int = 16,  # Use mixed precision by default
    seed: int = 42,
    log_dir: str = "logs",
    save_dir: str = "models/checkpoints",
):
    """
    Train the actuator model.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing data
        max_epochs: Maximum number of epochs
        gpu: The gpu to use. -1 for all available, 0 for gpu 0, 1 for gpu 1, etc.
        precision: Precision for training
        seed: Random seed
        log_dir: Directory for logs
        save_dir: Directory for saving checkpoints
    """
    # Set seed for reproducibility
    pl.seed_everything(seed)
    print(f"Seed set to {seed}")
    
    # Load configuration
    if config_path:
        train_config = load_config(config_path)
        model_config_path = os.path.join(os.path.dirname(config_path), train_config.get("model", ""))
        data_config_path = os.path.join(os.path.dirname(config_path), train_config.get("data", ""))
        
        model_config = load_config(model_config_path) if os.path.exists(model_config_path) else {}
        data_config = load_config(data_config_path) if os.path.exists(data_config_path) else {}
    else:
        model_config = {}
        data_config = {}
        train_config = {}
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize datamodule
    datamodule = ActuatorDataModule(
        data_file=data_file,
        batch_size=train_config.get("batch_size", 64),
        num_workers=train_config.get("num_workers", 4),
        **data_config
    )
    
    # Setup datamodule to get feature_dim
    datamodule.setup()
    feature_dim = datamodule.get_feature_dim()
    print(f"feature_dim: {feature_dim}")

    # Log the number of training batches
    num_train_batches = len(datamodule.train_dataloader())
    print(f"Number of training batches per epoch: {num_train_batches}")

    # Initialize model
    model = ActuatorModel(
        input_dim=int(np.prod(feature_dim)), # Flatten the feature dimension, input layer must be 1d not 2d. Also needs to be standard python int not numpy int.
        learning_rate=train_config.get("learning_rate", 1e-3),
        weight_decay=train_config.get("weight_decay", 1e-6),
        batch_size=train_config.get("batch_size", 64),
        **model_config
    )

    wandb_logger = WandbLogger(
        project=train_config["wandb"]["project_name"],  # Get project name from config
        log_model="all",  # Log all checkpoints
        config={**train_config, **model_config, **data_config},  # Log the config files
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"models/checkpoints/{wandb_logger.version}",  # Save in wandb versioned folder
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping( 
            monitor="val_loss",
            patience=train_config.get("patience", 10),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
   
        ModelExporter(output_dir=os.path.join(save_dir, "../exported")),
        EarlySummary(),
        TestPredictionPlotter(),
    ]
    
    # Configure multi-GPU strategy
    strategy = "ddp" if gpu == -1 else "auto" # Auto-select DDP (Distributed Data Parallel) for multiple GPUs
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=gpu,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=train_config.get("log_frequency", 10),
        val_check_interval=train_config.get("val_check_interval", 1.0),
        deterministic=True,
    )
    
    try:
        trainer.fit(model, datamodule=datamodule)
        
        trainer.test(model, datamodule=datamodule)
    
        wandb.finish()
        
        return model, datamodule
    
    # Handle keyboard interrupts (Ctrl+C)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. ")
        # Finish the wandb run properly
        wandb.finish()
        print("Wandb run finished.")
        return None, None
    


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train actuator model")
    parser.add_argument(
        "--config", type=str, default="configs/train/default.yaml", 
        help="Path to config file"
    )
    parser.add_argument(
        "--data_file", type=str, default="data/synthetic/joint_data/joint-data-1.csv", 
        help="Directory with data"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, 
        help="Maximum epochs for training"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, 
        help="The gpu to use. -1 for all available, 0 for gpu 0, 1 for gpu 1, etc."
    )
    parser.add_argument(
        "--precision", type=int, default=16, 
        choices=[16, 32], 
        help="Precision for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Call the training function
    train(
        config_path=args.config,
        data_file=args.data_file,
        max_epochs=args.max_epochs,
        gpu=args.gpu,
        precision=args.precision,
        seed=args.seed,
    ) 