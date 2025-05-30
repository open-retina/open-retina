#!/usr/bin/env python3

import logging
import os

import hydra
import lightning.pytorch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.base import compute_data_info
from openretina.data_io.cyclers import LongCycler, ShortCycler
from openretina.utils.log_to_mlflow import log_to_mlflow
import platform
import sys
import mlflow

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="hoefling_2024_core_readout_high_res",
)
def main(cfg: DictConfig) -> float | None:
    score = train_model(cfg)
    return score


def train_model(cfg: DictConfig) -> float | None:  ## normally train_model
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

    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True),
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ### Model init
    # Assign missing n_neurons_dict to the model
    cfg.model.n_neurons_dict = data_info["n_neurons_dict"]

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, data_info=data_info)

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
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ### Testing
    log.info("Starting testing!")
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    dataloader_mapping = {f"DataLoader {i}": x[0] for i, x in enumerate(short_cyclers)}
    log.info(f"Dataloader mapping: {dataloader_mapping}")
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path="best")
    # Check if MLflow is one of the loggers and save model and datasets as artifacts

    mlflow_logger_array = [logger for logger in logger_array if "mlflow" in str(type(logger)).lower()]
    if len(mlflow_logger_array) > 1:
        raise ValueError(f"Multiple mlflow loggers defined:  {[str(type(logger)) for logger in mlflow_logger_array]}")
    elif len(mlflow_logger_array == 1):
        logger = mlflow_logger_array[0]
        log_to_mlflow(logger, model, cfg, data_info, valid_loader)

    if cfg.objective_target is not None:
        ### Final validation for optuna
        log.info("Starting validation for Optuna")
        target_score = trainer.validate(model, dataloaders=[valid_loader], ckpt_path="best")[0][cfg.objective_target]
        if target_score is None:
            log.error(f"Score for objective target '{cfg.objective_target}' is None!")
        return target_score
    else:
        return None


if __name__ == "__main__":
    main()
