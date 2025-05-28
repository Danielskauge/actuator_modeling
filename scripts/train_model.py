import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import wandb # Explicit import for wandb.finish()

from src.data.datamodule import ActuatorDataModule
from src.models.model import ActuatorModel
from src.utils.callbacks import TestPredictionPlotter, EarlySummary # Assuming these exist and are compatible

from typing import Tuple # For CustomEarlyStopping type hints

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, stopping_threshold: float = None, monitor_op=None, **kwargs):
        super().__init__(**kwargs)
        self.stopping_threshold = stopping_threshold
        # If monitor_op is not provided, deduce from mode
        if monitor_op is None:
            if self.mode == "min":
                self.monitor_op = torch.less_equal
            elif self.mode == "max":
                self.monitor_op = torch.greater_equal
            else:
                self.monitor_op = torch.less_equal # Default for safety
                print(f"Warning: CustomEarlyStopping mode '{self.mode}' not 'min' or 'max'. Defaulting monitor_op to less_equal for threshold check.")
        else:
            self.monitor_op = monitor_op
        print(f"CustomEarlyStopping initialized. Monitor: {self.monitor}, Mode: {self.mode}, Patience: {self.patience}, Threshold: {self.stopping_threshold}")

    def _evaluate_stopping_criteria(self, current_value: torch.Tensor) -> Tuple[bool, str]:
        should_stop, reason = super()._evaluate_stopping_criteria(current_value)
        if not should_stop and self.stopping_threshold is not None:
            # Ensure threshold is a tensor on the same device for comparison
            threshold_tensor = torch.tensor(self.stopping_threshold, device=current_value.device, dtype=current_value.dtype)
            if self.monitor_op(current_value, threshold_tensor):
                should_stop = True
                reason = f"Metric {self.monitor}={current_value:.6f} reached stopping threshold {self.stopping_threshold:.6f} (op: {self.monitor_op.__name__}). Signaling Trainer to stop."
                print(reason)
        return should_stop, reason

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_actuator_model(cfg: DictConfig) -> None:
    """Main training script supporting global and LOMO CV evaluation modes."""
    print("Starting training script...")
    evaluation_mode = cfg.get('evaluation_mode', 'lomo_cv') # Default to lomo_cv if not set
    print(f"Evaluation mode: {evaluation_mode}")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    datamodule: ActuatorDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup() # This loads all_datasets_loaded (internal list of ActuatorDataset objects)

    # --- Model Instantiation Args Preparation (input_dim is dynamic) ---
    # All other model parameters (like gru_hidden_dim, use_residual, etc.) 
    # are expected to be directly in cfg.model and will be passed via **model_cfg_dict
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg_dict['input_dim'] = datamodule.get_input_dim() # Should be 3
    # model_cfg_dict['sampling_frequency'] = sampling_freq # REMOVED - ActuatorModel no longer takes this
    
    # Common callbacks are defined as types and instantiated per run/fold
    common_callbacks_types = []
    if cfg.callbacks.get('learning_rate_monitor', True):
        common_callbacks_types.append(LearningRateMonitor)
    if cfg.callbacks.get('early_summary', False):
        common_callbacks_types.append(EarlySummary)
    if cfg.callbacks.get('test_prediction_plotter', False):
        common_callbacks_types.append(TestPredictionPlotter)

    # --- Training Execution --- 
    if evaluation_mode == "global":
        print("\n===== GLOBAL EVALUATION MODE =====")
        datamodule.setup_for_global_run()
        
        # Instantiate model for global run
        model_global = ActuatorModel(**model_cfg_dict)

        run_name_global = f"{model_global.hparams.model_type}-global_{cfg.wandb.get('name_suffix', '')}"

        callbacks_global = [CB() for CB in common_callbacks_types] # Instantiate common callbacks

        checkpoint_callback_global = ModelCheckpoint(
            monitor=cfg.train.checkpointing.monitor,
            mode=cfg.train.checkpointing.mode,
            save_top_k=cfg.train.checkpointing.save_top_k,
            filename=f"{run_name_global}-{{epoch:02d}}-{{{cfg.train.checkpointing.monitor}:.4f}}",
            dirpath=os.path.join(cfg.outputs_dir, "checkpoints", "global_run")
        )
        callbacks_global.append(checkpoint_callback_global)
        
        if cfg.train.early_stopping.get("active", True):
            early_stop_callback_global = CustomEarlyStopping(
                monitor=cfg.train.early_stopping.monitor,
                mode=cfg.train.early_stopping.mode,
                patience=cfg.train.early_stopping.patience,
                min_delta=cfg.train.early_stopping.min_delta,
                verbose=cfg.train.early_stopping.verbose,
                stopping_threshold=cfg.train.early_stopping.get("stopping_threshold", None) 
            )
            callbacks_global.append(early_stop_callback_global)

        wandb_logger_global = None
        if cfg.wandb.get("active", False):
            wandb_logger_global = WandbLogger(
                name=run_name_global,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity if cfg.wandb.entity else None,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                save_dir=cfg.outputs_dir,
                group=f"GLOBAL_{model_global.hparams.model_type}_{cfg.wandb.get('group_suffix', '')}",
                job_type="train_global"
            )
            wandb_logger_global.watch(model_global, log='all', log_freq=cfg.wandb.get("watch_log_freq", 100))
        
        trainer_global = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            precision=cfg.train.precision,
            callbacks=callbacks_global,
            logger=wandb_logger_global if wandb_logger_global else True,
            gradient_clip_val=cfg.train.get("gradient_clip_val", 0.0), 
            gradient_clip_algorithm=cfg.train.get("gradient_clip_algorithm", "norm"),
            deterministic=cfg.get("deterministic_trainer", True)
        )
        
        print("Starting global training...")
        trainer_global.fit(model_global, datamodule=datamodule)
        print("Starting global testing...")
        global_test_results = trainer_global.test(model_global, datamodule=datamodule, ckpt_path='best')
        if global_test_results:
            print(f"Global Test Metrics: {global_test_results[0]}")
            if wandb_logger_global: wandb_logger_global.log_metrics({"global_best_" + k: v for k,v in global_test_results[0].items()})
        if wandb_logger_global: wandb.finish()

    elif evaluation_mode == "lomo_cv":
        print("\n===== LOMO CROSS-VALIDATION MODE =====")
        num_folds = datamodule.get_num_lomo_folds()
        if num_folds == 0:
            raise ValueError("No datasets found for LOMO CV. Check data.dataset_configs.")
        if num_folds < 2 and num_folds > 0: # If only 1 dataset, LOMO CV doesn't make sense for generalization testing.
            print("Warning: Only 1 dataset found. LOMO CV will run for 1 fold. This is similar to a global run but uses LOMO data setup logic. Consider using 'global' evaluation_mode for a single dataset.")
        elif num_folds == 0:
             raise ValueError("LOMO CV requires at least one dataset.")

        all_fold_test_metrics = []

        for fold_idx in range(num_folds):
            print(f"\n===== FOLD {fold_idx + 1} / {num_folds} =====")
            
            # Instantiate a fresh model for each fold
            # model_cfg_dict already has input_dim correctly set
            model_fold = ActuatorModel(**model_cfg_dict)
            run_name_fold = f"{model_fold.hparams.model_type}-fold_{fold_idx + 1}_{cfg.wandb.get('name_suffix', '')}"
            
            datamodule.setup_for_lomo_fold(fold_idx)

            callbacks_fold = [CB() for CB in common_callbacks_types]
            checkpoint_callback_fold = ModelCheckpoint(
                monitor=cfg.train.checkpointing.monitor,
                mode=cfg.train.checkpointing.mode,
                save_top_k=cfg.train.checkpointing.save_top_k,
                filename=f"{run_name_fold}-{{epoch:02d}}-{{{cfg.train.checkpointing.monitor}:.4f}}",
                dirpath=os.path.join(cfg.outputs_dir, "checkpoints", f"fold_{fold_idx + 1}")
            )
            callbacks_fold.append(checkpoint_callback_fold)

            if cfg.train.early_stopping.get("active", True):
                early_stop_callback_fold = CustomEarlyStopping(
                    monitor=cfg.train.early_stopping.monitor,
                    mode=cfg.train.early_stopping.mode,
                    patience=cfg.train.early_stopping.patience,
                    min_delta=cfg.train.early_stopping.min_delta,
                    verbose=cfg.train.early_stopping.verbose,
                    stopping_threshold=cfg.train.early_stopping.get("stopping_threshold", None)
                )
                callbacks_fold.append(early_stop_callback_fold)

            wandb_logger_fold = None
            if cfg.wandb.get("active", False):
                wandb_logger_fold = WandbLogger(
                    name=run_name_fold,
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity if cfg.wandb.entity else None,
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                    save_dir=cfg.outputs_dir,
                    group=f"LOMO_CV_{model_fold.hparams.model_type}_{cfg.wandb.get('group_suffix', '')}",
                    job_type="train_fold",
                    resume="allow", 
                    id=wandb.util.generate_id() 
                )
                wandb_logger_fold.watch(model_fold, log='all', log_freq=cfg.wandb.get("watch_log_freq", 100))
            
            trainer_fold = pl.Trainer(
                max_epochs=cfg.train.max_epochs,
                accelerator=cfg.train.accelerator,
                devices=cfg.train.devices,
                precision=cfg.train.precision,
                callbacks=callbacks_fold,
                logger=wandb_logger_fold if wandb_logger_fold else True,
                gradient_clip_val=cfg.train.get("gradient_clip_val", 0.0),
                gradient_clip_algorithm=cfg.train.get("gradient_clip_algorithm", "norm"),
                deterministic=cfg.get("deterministic_trainer", True)
            )

            print(f"Starting training for fold {fold_idx + 1}...")
            trainer_fold.fit(model_fold, datamodule=datamodule)
            print(f"Starting testing for fold {fold_idx + 1}...")
            fold_test_results = trainer_fold.test(model_fold, datamodule=datamodule, ckpt_path='best')
            
            if fold_test_results:
                all_fold_test_metrics.append(fold_test_results[0])
                print(f"Fold {fold_idx + 1} Test Metrics: {fold_test_results[0]}")
            else:
                print(f"No test results returned for fold {fold_idx + 1}")

            if wandb_logger_fold: 
                # Log per-fold metrics to the current wandb run for this fold
                if fold_test_results:
                    wandb_logger_fold.log_metrics({"fold_best_" + k: v for k,v in fold_test_results[0].items()})
                wandb.finish() # Finish wandb run for the fold

        # --- Aggregate and Print LOMO CV Results --- (after all folds)
        if all_fold_test_metrics:
            print("\n===== LOMO Cross-Validation Summary =====")
            avg_metrics = {}
            # Assuming all metric dicts have the same keys as the first one
            if all_fold_test_metrics[0]:
                metric_keys = all_fold_test_metrics[0].keys()
                for key in metric_keys:
                    try:
                        # Collect values for this key from all folds that have it
                        key_values = [m[key] for m in all_fold_test_metrics if m and key in m]
                        if key_values: # Ensure list is not empty
                             avg_metrics["avg_" + key] = np.mean(key_values)
                    except TypeError:
                        print(f"Could not average metric '{key}' as it might not be numeric.")
            
            if avg_metrics:
                print("Average Test Metrics Across Folds:")
                for key, val in avg_metrics.items():
                    print(f"  {key}: {val:.6f}")
                
                # Optional: Log aggregated LOMO CV results to a new, summary WandB run
                if cfg.wandb.get("active", False) and cfg.wandb.get("log_lomo_summary", True):
                    summary_run_name = f"{model_cfg_dict.get('model_type', 'model')}-LOMO_CV_Summary_{cfg.wandb.get('name_suffix', '')}"
                    wandb.init(
                        project=cfg.wandb.project,
                        entity=cfg.wandb.entity if cfg.wandb.entity else None,
                        name=summary_run_name,
                        group=f"LOMO_CV_SUMMARY_{model_cfg_dict.get('model_type', 'model')}_{cfg.wandb.get('group_suffix', '')}",
                        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                        job_type="lomo_summary",
                        reinit=True # Important if previous fold runs were in the same process
                    )
                    wandb.log(avg_metrics)
                    wandb.finish()
            else:
                print("No numeric metrics found to average for LOMO CV summary.")
        else:
            print("\nNo test metrics collected across folds for LOMO CV.")
    else:
        raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}. Choose 'global' or 'lomo_cv'.")

    print("Training script finished.")

if __name__ == "__main__":
    train_actuator_model() 