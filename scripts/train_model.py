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
import json # For saving config summary

from src.data.datamodule import ActuatorDataModule
from src.models.model import ActuatorModel
from src.utils.callbacks import TestPredictionPlotter, EarlySummary, DatasetVisualizationCallback, DebugTestTimeSeriesPlotter # Assuming these exist and are compatible
# from src.data.datasets import ActuatorDataset # No longer needed for SEQUENCE_LENGTH

from typing import Tuple, Optional # For CustomEarlyStopping type hints

# Helper function for JIT export
def _export_model_to_jit(
    checkpoint_path: str,
    model_class: type,  # Should be ActuatorModel
    jit_filename: str,
    output_dir: str,
    # Example inputs for scripting/tracing
    model_example_input: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    wandb_logger: Optional[WandbLogger] = None,
    run_name_for_artifact: Optional[str] = None,
    trainer: Optional[pl.Trainer] = None  # Trainer to attach for to_torchscript
):
    """Loads a model from a checkpoint, converts to TorchScript, saves, and logs to WandB."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path not found: {checkpoint_path}. Skipping JIT export.")
        return

    print(f"\nExporting model from {checkpoint_path} to TorchScript (JIT)...")
    try:
        # Prefer instantiating model then loading state_dict for robustness
        # First, load the hparams from the checkpoint to instantiate the model correctly.
        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        hparams = ckpt.get('hyper_parameters')
        if hparams is None:
            raise ValueError("Hyperparameters not found in checkpoint.")
        
        # Instantiate model with hparams from checkpoint
        model_to_jit = model_class(**hparams)
        model_to_jit.load_state_dict(ckpt['state_dict'])
        model_to_jit.eval()  # Ensure model is in eval mode
        # Flatten RNN weights to avoid fragmentation warnings
        if hasattr(model_to_jit, 'model') and hasattr(model_to_jit.model, 'gru'):
            print("Flattening GRUModel parameters in JIT export...")
            model_to_jit.model.gru.flatten_parameters()
        elif hasattr(model_to_jit, 'flatten_parameters'):
            print("Flattening parameters on top-level JIT model...")
            model_to_jit.flatten_parameters()
        # Attach provided Trainer (or a temporary one) so LightningModule.to_torchscript won't error
        if trainer:
            model_to_jit.trainer = trainer
        else:
            print("Warning: No Trainer provided for JIT export; attaching a temporary Trainer.")
            model_to_jit.trainer = pl.Trainer()

        jit_model_path = os.path.join(output_dir, jit_filename)
        os.makedirs(os.path.dirname(jit_model_path), exist_ok=True) # Ensure dir exists

        # Use Lightning's to_torchscript API
        if model_example_input:
            scripted_model = model_to_jit.to_torchscript(
                method='script', example_inputs=[model_example_input]
            )
        else:
            print("Warning: No example_input provided for JIT scripting. Proceeding without it.")
            scripted_model = model_to_jit.to_torchscript(method='script')
            
        torch.jit.save(scripted_model, jit_model_path)
        print(f"Model successfully exported to TorchScript: {jit_model_path}")

        if wandb_logger and run_name_for_artifact: # wandb_logger is passed, implying wandb is active for the trainer
            if wandb.run is not None: # Check if there is an active wandb run
                try:
                    artifact_name = f"{run_name_for_artifact}_jit"
                    # Use wandb.log_artifact directly
                    artifact = wandb.Artifact(name=artifact_name, type="model")
                    artifact.add_file(jit_model_path)
                    wandb.log_artifact(artifact)
                    print(f"JIT model logged as WandB artifact: {artifact_name}")
                except Exception as e:
                    print(f"Error logging JIT model to WandB: {e}")
            else:
                print("WandB run not active. Skipping artifact logging for JIT model.")

    except Exception as e:
        print(f"Error during JIT export for {checkpoint_path}: {e}")

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, stopping_threshold: float = None, monitor_op=None, **kwargs):
        super().__init__(**kwargs)
        self.stopping_threshold = stopping_threshold
        # Store the custom monitor_op logic in a separate attribute
        if monitor_op is None:
            if self.mode == "min":
                self._custom_monitor_op = torch.less_equal
            elif self.mode == "max":
                self._custom_monitor_op = torch.greater_equal
            else:
                self._custom_monitor_op = torch.less_equal # Default for safety
                print(f"Warning: CustomEarlyStopping mode '{self.mode}' not 'min' or 'max'. Defaulting _custom_monitor_op to less_equal for threshold check.")
        else:
            self._custom_monitor_op = monitor_op
        print(f"CustomEarlyStopping initialized. Monitor: {self.monitor}, Mode: {self.mode}, Patience: {self.patience}, Threshold: {self.stopping_threshold}, Custom Op: {self._custom_monitor_op.__name__ if hasattr(self._custom_monitor_op, '__name__') else 'N/A'}")

    def _evaluate_stopping_criteria(self, current_value: torch.Tensor) -> Tuple[bool, str]:
        should_stop, reason = super()._evaluate_stopping_criteria(current_value)
        if not should_stop and self.stopping_threshold is not None:
            threshold_tensor = torch.tensor(self.stopping_threshold, device=current_value.device, dtype=current_value.dtype)
            # Use the _custom_monitor_op for the threshold check
            if self._custom_monitor_op(current_value, threshold_tensor):
                should_stop = True
                # Use self.monitor (from parent) for the logging message, as it's the metric being monitored.
                reason = f"Metric {self.monitor}={current_value:.6f} reached stopping threshold {self.stopping_threshold:.6f} (op: {self._custom_monitor_op.__name__ if hasattr(self._custom_monitor_op, '__name__') else 'N/A'}). Signaling Trainer to stop."
                print(reason)
        return should_stop, reason

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_actuator_model(cfg: DictConfig) -> None:
    """Main training script supporting global and LOMO CV evaluation modes."""
    print("Starting training script...")
    evaluation_mode = cfg.get('evaluation_mode', 'lomo_cv') # Default to lomo_cv if not set
    print(f"Evaluation mode: {evaluation_mode}")
    # print(OmegaConf.to_yaml(cfg)) # Already logged by WandB if active

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('high') # Utilize Tensor Cores

    datamodule: ActuatorDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data() # Ensures all raw datasets are loaded.

    # --- Model Instantiation Args Preparation ---
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg_dict['input_dim'] = datamodule.get_input_dim()

    if isinstance(model_cfg_dict.get('model_type'), dict) and 'name' in model_cfg_dict['model_type']:
        model_type_name = model_cfg_dict['model_type']['name']
        model_cfg_dict['model_type'] = model_type_name
    
    # --- Call setup_for_global_run ONCE to compute normalization stats if LOMO CV or Global ---
    # This ensures that normalization stats are based on a consistent (global) training set view.
    # setup_for_global_run internally computes and stores these stats.
    # print("\n--- Ensuring normalization statistics are computed based on global training data ---")
    # datamodule.setup_for_global_run() # This will define global_train_dataset and compute stats from it.
    
    # input_stats = datamodule.get_input_normalization_stats()
    # target_stats = datamodule.get_target_normalization_stats()

    # if input_stats is None or target_stats is None:
    #     raise RuntimeError("Normalization statistics were not computed by the DataModule. Cannot proceed.")
    
    # model_cfg_dict['input_mean'], model_cfg_dict['input_std'] = input_stats
    # model_cfg_dict['target_mean'], model_cfg_dict['target_std'] = target_stats
    # print("Normalization stats retrieved and will be passed to ActuatorModel.")
    # model_cfg_dict['sampling_frequency'] = sampling_freq # REMOVED

    # Common callbacks are defined as types and instantiated per run/fold
    common_callbacks_types = []
    if cfg.train.callbacks.get('learning_rate_monitor', True):
        common_callbacks_types.append(LearningRateMonitor)
    if cfg.train.callbacks.get('early_summary', False):
        common_callbacks_types.append(EarlySummary)
    if cfg.train.callbacks.get('test_prediction_plotter', False):
        common_callbacks_types.append(TestPredictionPlotter)
        common_callbacks_types.append(DebugTestTimeSeriesPlotter)
    if cfg.train.callbacks.get('dataset_visualization', True): # Default to True
        common_callbacks_types.append(DatasetVisualizationCallback)

    # --- Training Execution --- 
    if evaluation_mode == "global":
        print("\n===== GLOBAL EVALUATION MODE =====")
        print("Setting up data and computing normalization statistics for global run...")
        datamodule.setup_for_global_run() # Computes and stores global normalization stats
        
        input_stats_global = datamodule.get_input_normalization_stats()
        target_stats_global = datamodule.get_target_normalization_stats()
        if input_stats_global is None or target_stats_global is None:
            raise RuntimeError("Global normalization statistics were not computed by the DataModule. Cannot proceed.")
        
        # Save normalization stats for global run
        output_dir_global = cfg.outputs_dir # Hydra's output directory for this run
        os.makedirs(output_dir_global, exist_ok=True)
        
        norm_stats_global_dict = {
            "input_mean": input_stats_global[0].tolist(), # Convert to list
            "input_std": input_stats_global[1].tolist(),  # Convert to list
            "target_mean": target_stats_global[0].tolist(),# Convert to list
            "target_std": target_stats_global[1].tolist()  # Convert to list
        }
        # Save as JSON instead of .pt
        norm_stats_file_path_global = os.path.join(output_dir_global, "normalization_stats.json")
        with open(norm_stats_file_path_global, 'w') as f:
            json.dump(norm_stats_global_dict, f, indent=4)
        print(f"Saved global normalization statistics to {norm_stats_file_path_global}")

        model_cfg_dict_global = model_cfg_dict.copy() # Use a copy for global model
        model_cfg_dict_global['input_mean'], model_cfg_dict_global['input_std'] = input_stats_global
        model_cfg_dict_global['target_mean'], model_cfg_dict_global['target_std'] = target_stats_global
        print("Global normalization stats retrieved and will be passed to ActuatorModel for global run.")

        # Ensure we are using the datasets set up by the prior call to setup_for_global_run()
        if datamodule.global_train_dataset is None:
             raise RuntimeError("Global train dataset not available after setup_for_global_run. This should not happen.")
        # print("Using pre-established global data splits and normalization stats for global run.") # Redundant now
        
        # Instantiate model for global run (model_cfg_dict_global now includes normalization stats)
        model_global = ActuatorModel(**model_cfg_dict_global)

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
        
        # --- Export JIT model for global run ---
        if cfg.train.get("export_jit_model", False):
            if checkpoint_callback_global and checkpoint_callback_global.best_model_path:
                # Create example input for JIT scripting/tracing based on the model's new forward signature
                # (x, h_prev) -> (y, h_next)
                # x shape: (batch_size, seq_len, input_dim)
                # h_prev shape: (num_layers, batch_size, hidden_dim) or None
                example_batch_size = 1 # For inference, usually batch size is 1
                example_seq_len = datamodule.get_sequence_length_timesteps() # Use training sequence length for example
                example_input_dim = datamodule.get_input_dim()
                example_gru_num_layers = model_global.hparams.gru_num_layers
                example_gru_hidden_dim = model_global.hparams.gru_hidden_dim

                example_x = torch.randn(example_batch_size, example_seq_len, example_input_dim)
                example_h_prev_tensor = torch.randn(example_gru_num_layers, example_batch_size, example_gru_hidden_dim)
                
                # JIT needs a list of tuples if multiple example inputs are to be tested, or one tuple for one call signature
                # For a single forward signature like forward(self, x, h_prev=None)
                # we can provide one example call with h_prev as a tensor, and one with h_prev as None.
                # However, torch.jit.script might be able to infer Optional better.
                # Let's provide one example where h_prev is a tensor.
                # The JIT scripter should then understand that h_prev can also be None.
                example_model_input_for_jit = (example_x, example_h_prev_tensor)

                _export_model_to_jit(
                    checkpoint_path=checkpoint_callback_global.best_model_path,
                    model_class=ActuatorModel,
                    jit_filename=f"{run_name_global}_best_jit.pt",
                    output_dir=cfg.outputs_dir,  # Save in the run's output dir
                    model_example_input=example_model_input_for_jit,
                    wandb_logger=wandb_logger_global,
                    run_name_for_artifact=run_name_global,
                    trainer=trainer_global
                )
            else:
                print("No best model checkpoint path available from callback for global run. Skipping JIT export.")

        if wandb_logger_global: wandb.finish()

    elif evaluation_mode == "lomo_cv":
        print("\n===== LOMO CROSS-VALIDATION MODE =====")
        num_folds = datamodule.get_num_lomo_folds()
        if num_folds == 0:
            raise ValueError("No inertia groups found for LOMO CV. Check data.inertia_groups.")
        if num_folds < 2: # If only 1 group, LOMO CV doesn't make sense for generalization.
            print("Warning: Only 1 inertia group found. LOMO CV will run for 1 fold. Generalization to unseen inertias cannot be meaningfully tested. This is similar to a global run.")
        
        all_fold_test_metrics = []

        for fold_idx in range(num_folds):
            print(f"\n===== FOLD {fold_idx + 1} / {num_folds} =====")
            
            datamodule.setup_for_lomo_fold(fold_idx)

            input_stats_fold = datamodule.get_input_normalization_stats()
            target_stats_fold = datamodule.get_target_normalization_stats()
            if input_stats_fold is None or target_stats_fold is None:
                raise RuntimeError(f"Fold-specific normalization statistics were not computed for fold {fold_idx + 1}. Cannot proceed.")

            # Save normalization stats for this LOMO fold
            output_dir_fold_stats = os.path.join(cfg.outputs_dir, "checkpoints", f"fold_{fold_idx + 1}", "norm_stats_and_config")
            os.makedirs(output_dir_fold_stats, exist_ok=True)
            
            norm_stats_fold_dict = {
                "input_mean": input_stats_fold[0].tolist(), # Convert to list
                "input_std": input_stats_fold[1].tolist(),  # Convert to list
                "target_mean": target_stats_fold[0].tolist(),# Convert to list
                "target_std": target_stats_fold[1].tolist()  # Convert to list
            }
            # Save as JSON instead of .pt
            norm_stats_file_path_fold = os.path.join(output_dir_fold_stats, "normalization_stats.json")
            with open(norm_stats_file_path_fold, 'w') as f:
                json.dump(norm_stats_fold_dict, f, indent=4)
            print(f"Saved fold {fold_idx + 1} normalization statistics to {norm_stats_file_path_fold}")

            model_cfg_dict_fold = model_cfg_dict.copy()
            model_cfg_dict_fold['input_mean'], model_cfg_dict_fold['input_std'] = input_stats_fold
            model_cfg_dict_fold['target_mean'], model_cfg_dict_fold['target_std'] = target_stats_fold
            print(f"Fold {fold_idx+1} normalization stats retrieved and will be passed to ActuatorModel.")

            # Instantiate a fresh model for each fold, passing the FOLD-SPECIFIC global normalization stats
            model_fold = ActuatorModel(**model_cfg_dict_fold)
            run_name_fold = f"{model_fold.hparams.model_type}-fold_{fold_idx + 1}_{cfg.wandb.get('name_suffix', '')}"
            
            # Setup data for the current LOMO fold. This uses the pre-computed global norm stats.
            # datamodule.setup_for_lomo_fold(fold_idx) # Already called above

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
                # Log held-out group info to WandB config and print after run start
                held_out_group_id = datamodule.ordered_group_ids[fold_idx]
                held_out_csv_files = len(datamodule.datasets_by_group[held_out_group_id])
                held_out_sequences = len(datamodule.current_fold_val_dataset)
                wandb_logger_fold.experiment.config.update({
                    'held_out_group': held_out_group_id,
                    'held_out_csv_files': held_out_csv_files,
                    'held_out_sequences': held_out_sequences
                })
                print(f"  Held-out group '{held_out_group_id}' for Val/Test in Fold {fold_idx + 1}: {held_out_sequences} sequences from {held_out_csv_files} CSV files")
            
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

            # --- Export JIT model for LOMO CV fold ---
            if cfg.train.get("export_jit_model", False):
                if checkpoint_callback_fold and checkpoint_callback_fold.best_model_path:
                    # Create example input for JIT scripting/tracing for the fold model
                    example_batch_size_fold = 1 
                    example_seq_len_fold = datamodule.get_sequence_length_timesteps()
                    example_input_dim_fold = datamodule.get_input_dim()
                    example_gru_num_layers_fold = model_fold.hparams.gru_num_layers
                    example_gru_hidden_dim_fold = model_fold.hparams.gru_hidden_dim

                    example_x_fold = torch.randn(example_batch_size_fold, example_seq_len_fold, example_input_dim_fold)
                    example_h_prev_tensor_fold = torch.randn(example_gru_num_layers_fold, example_batch_size_fold, example_gru_hidden_dim_fold)
                    example_model_input_for_jit_fold = (example_x_fold, example_h_prev_tensor_fold)

                    fold_jit_output_dir = os.path.join(cfg.outputs_dir, "checkpoints", f"fold_{fold_idx + 1}", "jit_models")
                    _export_model_to_jit(
                        checkpoint_path=checkpoint_callback_fold.best_model_path,
                        model_class=ActuatorModel,
                        jit_filename=f"{run_name_fold}_best_jit.pt",
                        output_dir=fold_jit_output_dir,
                        model_example_input=example_model_input_for_jit_fold,
                        wandb_logger=wandb_logger_fold,
                        run_name_for_artifact=run_name_fold,
                        trainer=trainer_fold
                    )
                else:
                    print(f"No best model checkpoint path available from callback for fold {fold_idx + 1}. Skipping JIT export.")

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