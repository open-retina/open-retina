#!/usr/bin/env python3

import logging
import os

import hydra
import lightning.pytorch
import torch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.base import compute_data_info
from openretina.data_io.cyclers import LongCycler, ShortCycler
from openretina.models.core_readout import UnifiedCoreReadout, load_core_readout_model
from openretina.utils.log_to_mlflow import log_to_mlflow

log = logging.getLogger(__name__)
logging.captureWarnings(True)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="hoefling_2024_core_readout_high_res",
)
def main(cfg: DictConfig) -> float | None:
    score = train_model(cfg)
    return score


def train_model(cfg: DictConfig) -> float | None:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

    ### Set matmul precision if defined
    if "matmul_precision" in cfg:
        hydra.utils.call(cfg.matmul_precision)

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

    data_info = compute_data_info(neuron_data_dict, movies_dict, partial_data_info=cfg.data_io.get("data_info"))

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
    load_model_path = cfg.paths.get("load_model_path")
    if load_model_path:
        log.info(f"Loading model from <{load_model_path}>")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_core_readout_model(load_model_path, device)

        new_session_keys = {k for k in data_info["n_neurons_dict"].keys() if k not in model.readout.keys()}
        # add new readouts and modify stored data in model
        if len(new_session_keys) > 0:
            log.info(f"Adding additional sessions to pretrained model: {new_session_keys}")
            n_neurons_dict_subset = {k: v for k, v in data_info["n_neurons_dict"].items() if k in new_session_keys}
            if "mean_activity_dict" in data_info:
                mean_activity_dict = {k: v for k, v in data_info["mean_activity_dict"].items() if k in new_session_keys}
            else:
                mean_activity_dict = None
            model.readout.add_sessions(n_neurons_dict_subset, mean_activity_dict)  # type: ignore
            for k in ["n_neurons_dict", "mean_activity_dict", "sessions_kwargs"]:
                if model.data_info.get(k) is not None:
                    data_info[k] = model.data_info[k] | data_info[k]
                else:
                    log.warning(f"data_info key {k} when loading model: {model.data_info.keys()=}")
            data_info["n_neurons_dict"] = model.data_info
            model.update_model_data_info(data_info)
    else:
        # Assign missing n_neurons_dict to model
        cfg.model.n_neurons_dict = data_info["n_neurons_dict"]
        if hasattr(cfg.model, "_target_"):
            log.info(f"Instantiating model <{cfg.model._target_}>")
            model = hydra.utils.instantiate(cfg.model, data_info=data_info)
        else:
            log.info("Instantiating model <UnifiedCoreReadout>")
            model = UnifiedCoreReadout(data_info=data_info, **cfg.model)

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
    elif len(mlflow_logger_array) == 1:
        logger = mlflow_logger_array[0]
        log_to_mlflow(logger, model, cfg, data_info, valid_loader)

    ### Final validation for optuna if objective target is set
    if cfg.get("objective_target") is None:
        return None

    log.info("Starting validation for Optuna")
    target_score = trainer.validate(model, dataloaders=[valid_loader], ckpt_path="best")[0][cfg.objective_target]
    if target_score is None:
        log.error(f"Score for objective target '{cfg.objective_target}' is None!")
    return target_score


if __name__ == "__main__":
    main()
