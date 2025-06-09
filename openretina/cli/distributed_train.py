#!/usr/bin/env python3

import logging
import os

import hydra
import lightning.pytorch
import torch
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.base import compute_data_info
from openretina.data_io.session_combined_dataset import SessionAwareDataModule
from openretina.models.core_readout import load_core_readout_model
from openretina.utils.log_to_mlflow import log_to_mlflow

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="hoefling_2024_distributed_example",
)
def main(cfg: DictConfig) -> float | None:
    """Standalone entry point for running distributed training directly."""
    score = train_model(cfg)
    return score


def train_model(cfg: DictConfig) -> float | None:
    """Main training function that can be called from CLI or standalone."""
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

    movies_dict = hydra.utils.call(cfg.data_io.stimuli)
    neuron_data_dict = hydra.utils.call(cfg.data_io.responses)

    if cfg.check_stimuli_responses_match:
        for session, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict[session])

    dataloaders = hydra.utils.instantiate(
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )

    data_info = compute_data_info(neuron_data_dict, movies_dict)

    # Create DataModule for session-aware distributed training
    data_module = SessionAwareDataModule(
        dataloaders_dict=dataloaders,
        batch_size=cfg.dataloader.batch_size,
        seed=cfg.seed,
    )

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ### Model init
    load_model_path = cfg.paths.get("load_model_path")
    if load_model_path:
        log.info(f"Loading model from <{load_model_path}>")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_gru_model = "gru" in cfg.model._target_.lower() if hasattr(cfg.model, "_target_") else False
        model = load_core_readout_model(load_model_path, device, is_gru_model=is_gru_model)

        # add new readouts and modify stored data in model
        model.readout.add_sessions(data_info["n_neurons_dict"])  # type: ignore
        model.update_model_data_info(data_info)

    else:
        # Assign missing n_neurons_dict to model
        cfg.model.n_neurons_dict = data_info["n_neurons_dict"]
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model, data_info=data_info)

    if cfg.get("only_train_readout") is True:
        log.info("Only training readout, core model parameters will be frozen.")
        model.core.requires_grad_(False)

    ### Logging
    log.info("Instantiating loggers...")
    logger_array = []
    for logger_name, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params)
        logger_array.append(logger)

    ### Callbacks
    log.info("Instantiating callbacks...")
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]

    ### Trainer init
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, datamodule=data_module)

    ### Convert DeepSpeed checkpoint to regular PyTorch format if needed
    is_deepspeed = "deepspeed" in str(trainer.strategy).lower()
    if is_deepspeed:
        try:
            import glob

            from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

            # Find the latest checkpoint directory
            checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
            checkpoint_patterns = [
                os.path.join(checkpoint_dir, "epoch=*_val_correlation=*.ckpt"),
                os.path.join(checkpoint_dir, "epoch*.ckpt"),
                os.path.join(checkpoint_dir, "*.ckpt"),
            ]

            latest_checkpoint = None
            for pattern in checkpoint_patterns:
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    # Get the most recent checkpoint
                    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
                    break

            if latest_checkpoint and os.path.isdir(latest_checkpoint):
                # Convert DeepSpeed checkpoint to regular PyTorch format
                output_path = os.path.join(cfg.paths.output_dir, "model_final.pt")
                log.info(f"Converting DeepSpeed checkpoint from {latest_checkpoint} to {output_path}")
                convert_zero_checkpoint_to_fp32_state_dict(latest_checkpoint, output_path)
                log.info(f"Successfully converted DeepSpeed checkpoint to {output_path}")
            else:
                log.warning(f"No DeepSpeed checkpoint directory found in {checkpoint_dir}")

        except ImportError:
            log.warning(
                "DeepSpeed checkpoint conversion not available - lightning.pytorch.utilities.deepspeed not found"
            )
        except Exception as e:
            log.warning(f"Failed to convert DeepSpeed checkpoint: {e}")

    ### Testing
    log.info("Starting testing!")

    # Add test dataloaders to DataModule
    data_module.test_dataloaders_dict = dataloaders

    # Use the is_deepspeed variable defined above
    if is_deepspeed:
        # For DeepSpeed, use the current model state instead of loading checkpoint
        # since DeepSpeed checkpoints are directories, not .ckpt files
        log.info("Using current model state for testing (DeepSpeed strategy detected)")
        trainer.test(model=model, datamodule=data_module)
    else:
        # For regular strategies, load the best checkpoint
        log.info("Loading best checkpoint for testing")
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    # Check if MLflow is one of the loggers and save model and datasets as artifacts

    mlflow_logger_array = [logger for logger in logger_array if "mlflow" in str(type(logger)).lower()]
    if len(mlflow_logger_array) > 1:
        raise ValueError(f"Multiple mlflow loggers defined:  {[str(type(logger)) for logger in mlflow_logger_array]}")
    elif len(mlflow_logger_array) == 1:
        logger = mlflow_logger_array[0]
        log_to_mlflow(logger, model, cfg, data_info, data_module.val_dataloader())

    if cfg.get("objective_target") is not None:
        ### Final validation for optuna
        log.info("Starting validation for Optuna")
        target_score = trainer.validate(model, dataloaders=[data_module.val_dataloader()], ckpt_path="best")[0][
            cfg.objective_target
        ]
        if target_score is None:
            log.error(f"Score for objective target '{cfg.objective_target}' is None!")
        return target_score
    else:
        return None


if __name__ == "__main__":
    main()
