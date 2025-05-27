import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from src.data.datamodule import ActuatorDataModule
from src.models.model import ActuatorModel

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, stopping_threshold: float = None, **kwargs):
        super().__init__(**kwargs)
        self.stopping_threshold = stopping_threshold
        print(f"CustomEarlyStopping initialized. Monitor: {self.monitor}, Patience: {self.patience}, Threshold: {self.stopping_threshold}")

    def _evaluate_stopping_criteria(self, current_value: torch.Tensor) -> Tuple[bool, str]:
        should_stop, reason = super()._evaluate_stopping_criteria(current_value)
        if not should_stop and self.stopping_threshold is not None:
            if current_value <= self.stopping_threshold:
                should_stop = True
                reason = f"Metric {self.monitor}={current_value:.6f} was <= {self.stopping_threshold:.6f}. Signaling Trainer to stop."
                print(reason)
        return should_stop, reason

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_actuator_model(cfg: DictConfig) -> None:
    """Main training loop with LOMO CV using Hydra configurations."""
    print("Starting training script...")
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # --- DataModule Setup ---
    # The ActuatorDataModule now gets its input_dim and sequence_length from ActuatorDataset static properties.
    # Model hparams like k_spring are linked from model config to data config via Hydra.
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup() # Load all datasets once

    num_folds = len(datamodule.all_datasets)
    if num_folds == 0:
        raise ValueError("No datasets found in datamodule. Check data.dataset_configs.")
    print(f"Starting Leave-One-Mass-Out Cross-Validation with {num_folds} folds.")

    all_fold_test_metrics = []

    for fold_idx in range(num_folds):
        print(f"\n===== FOLD {fold_idx + 1} / {num_folds} =====")
        run_name = f"{cfg.model.model_type}-fold_{fold_idx + 1}_{cfg.wandb.get('name_suffix', '')}"
        
        # Setup datamodule for the current fold
        datamodule.setup_fold(fold_idx)

        # --- Model Instantiation ---
        # input_dim is now sourced from the datamodule after it interrogates ActuatorDataset
        model_params = OmegaConf.to_container(cfg.model, resolve=True) # Convert to dict
        model_params['input_dim'] = datamodule.get_input_dim()
        # Other ActuatorModel params like mlp_hidden_dims, gru_hidden_dim, etc., are directly in cfg.model
        model = ActuatorModel(**model_params)

        # --- Callbacks ---
        callbacks = []
        # Model Checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.train.checkpointing.monitor, # Should be val_rmse_epoch
            mode=cfg.train.checkpointing.mode,
            save_top_k=cfg.train.checkpointing.save_top_k,
            filename=f"{run_name}-{{epoch:02d}}-{{{cfg.train.checkpointing.monitor}:.4f}}",
            dirpath=os.path.join(cfg.outputs_dir, "checkpoints", f"fold_{fold_idx + 1}")
        )
        callbacks.append(checkpoint_callback)

        # Early Stopping (with custom threshold logic)
        early_stop_callback = CustomEarlyStopping(
            monitor=cfg.train.early_stopping.monitor, # Should be val_rmse_epoch
            mode=cfg.train.early_stopping.mode,
            patience=cfg.train.early_stopping.patience,
            min_delta=cfg.train.early_stopping.min_delta,
            verbose=cfg.train.early_stopping.verbose,
            stopping_threshold=0.05 # Target val_rmse_epoch <= 0.05 Nm
        )
        callbacks.append(early_stop_callback)

        # Learning Rate Monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))

        # --- Logger (WandB) ---
        wandb_logger = None
        if cfg.wandb.get("active", False): # Check if wandb is active
            wandb_logger = WandbLogger(
                name=run_name,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity if cfg.wandb.entity else None, # Handle empty string for default entity
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                save_dir=cfg.outputs_dir, # Save wandb files within hydra outputs
                group=f"LOMO_CV_{cfg.model.model_type}_{cfg.wandb.get('group_suffix', '')}",
                job_type="train_fold"
            )
            wandb_logger.watch(model, log='all', log_freq=100)
        else:
            print("WandB logging is disabled.")

        # --- Trainer --- 
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            precision=cfg.train.precision,
            callbacks=callbacks,
            logger=wandb_logger if wandb_logger else True, # Use default logger if wandb is off
            gradient_clip_val=5.0, # As requested
            gradient_clip_algorithm="norm", # As requested
            deterministic=True, # For reproducibility with seed_everything
            # val_check_interval=cfg.train.val_check_interval, # Already default
            # log_every_n_steps=cfg.train.log_frequency # Already default
        )

        print(f"Starting training for fold {fold_idx + 1}...")
        trainer.fit(model, datamodule=datamodule)

        print(f"Starting testing for fold {fold_idx + 1}...")
        fold_test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best') # Test best model
        
        if fold_test_results:
            all_fold_test_metrics.append(fold_test_results[0]) # trainer.test returns a list of dicts
            print(f"Fold {fold_idx + 1} Test Metrics: {fold_test_results[0]}")
        else:
            print(f"No test results returned for fold {fold_idx + 1}")

        if wandb_logger:
            wandb.finish() # Finish wandb run for the fold

    # --- Aggregate and Print LOMO CV Results ---
    if all_fold_test_metrics:
        print("\n===== LOMO Cross-Validation Summary =====")
        # Example: Average test_rmse_epoch across folds
        avg_metrics = {}
        metric_keys = all_fold_test_metrics[0].keys()
        for key in metric_keys:
            try:
                avg_metrics[key] = np.mean([m[key] for m in all_fold_test_metrics if key in m])
            except TypeError:
                print(f"Could not average metric '{key}' as it might not be numeric.")
        
        print("Average Test Metrics Across Folds:")
        for key, val in avg_metrics.items():
            print(f"  Avg {key}: {val:.6f}")
        
        # Log aggregated results to a summary file or a final WandB summary if desired
        # E.g., if using wandb, you could do a final summary log:
        # if cfg.wandb.get("active", False):
        #     wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=f"LOMO_CV_Summary_{cfg.model.model_type}", job_type="summary")
        #     wandb.log(avg_metrics)
        #     wandb.finish()
    else:
        print("\nNo test metrics collected across folds.")

    print("Training script finished.")

if __name__ == "__main__":
    train_actuator_model() 