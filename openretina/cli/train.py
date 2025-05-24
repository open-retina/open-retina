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
from openretina.models.core_readout import load_core_readout_model



log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="hoefling_2024_core_readout_low_res") # low res for fast testing
def main(cfg: DictConfig) -> None:
    train_model(cfg)


def train_model(cfg: DictConfig) -> None:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")

    ### Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    ### Display log directory for ease of access
    log.info(f"Saving run logs at: {cfg.paths.output_dir}")

    ### Import data
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
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ### Model init
    load_model_path = cfg.paths.get("load_model_path", None)
    if load_model_path:
        log.info(f"Loading model from <{load_model_path}>")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_gru_model = 'gru' in cfg.model._target_.lower() if hasattr(cfg.model, '_target_') else False
        model = load_core_readout_model(load_model_path, device, is_gru_model=is_gru_model)
        
        for key in data_info.keys():
            if key == 'input_shape':
                assert all(model.data_info[key][dim] == data_info[key][dim] for dim in range(len(data_info[key]))), \
                    f"Input shapes don't match: model has {model.data_info[key]}, new data has {data_info[key]}"
            else:
                model.data_info[key].update(data_info[key])
        
        
        for session_key, n_neurons in data_info["n_neurons_dict"].items():
            model.readout.add_readout_session(session_key, n_neurons)
        
        # update hyperparameters such that they are saved in the checkpoint
        if hasattr(model, 'hparams'):
            if 'n_neurons_dict' in model.hparams:
                model.hparams["n_neurons_dict"].update(data_info["n_neurons_dict"]) #type: ignore
        
    else:

        cfg.model.n_neurons_dict = data_info["n_neurons_dict"]
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model, data_info=data_info)

    if cfg.get("only_train_readout", False) == True:
        model.only_train_readout = True
        log.info(f"Only training readout.")
    else:
        model.only_train_readout = False

    ### Logging
    log.info("Instantiating loggers...")
    logger_array = []
    for _, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params)
        logger_array.append(logger)

    ### Callbacks
    log.info("Instantiating callbacks...")
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]
    log_parameter_snapshot(model, prefix="Before training")
    ### Trainer init
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: lightning.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    log_parameter_snapshot(model, prefix="After training")

    ### Testing
    log.info("Starting testing!")
    short_cyclers = [(n, ShortCycler(dl)) for n, dl in dataloaders.items()]
    dataloader_mapping = {f"DataLoader {i}": x[0] for i, x in enumerate(short_cyclers)}
    log.info(f"Dataloader mapping: {dataloader_mapping}")
    trainer.test(model, dataloaders=[c for _, c in short_cyclers], ckpt_path="best")



def log_parameter_snapshot(model, prefix=""):
    """Take a compact snapshot of parameter values to check for changes"""
    snapshots = {}
    
    # Core parameters (sample a few)
    core_params = list(model.core.parameters())
    if core_params:
        # Get first 3 parameters, first 5 values of each
        core_samples = {}
        for i, param in enumerate(core_params[:3]):
            values = param.flatten()[:5].detach().cpu().tolist()
            core_samples[f"core_param_{i}"] = [f"{v:.6f}" for v in values]
        snapshots["core"] = core_samples
    
    # Readout parameters (sample a few)
    readout_params = list(model.readout.parameters())
    if readout_params:
        readout_samples = {}
        for i, param in enumerate(readout_params[:3]):
            values = param.flatten()[:5].detach().cpu().tolist()
            readout_samples[f"readout_param_{i}"] = [f"{v:.6f}" for v in values]
        snapshots["readout"] = readout_samples
    
    log.info(f"{prefix} Parameter snapshot:")
    for module, params in snapshots.items():
        log.info(f"  {module}:")
        for name, values in params.items():
            log.info(f"    {name}: {values}")
    


if __name__ == "__main__":
    main()
