#!/usr/bin/env python3

import logging
import os

import hydra
import lightning.pytorch
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.cyclers import LongCycler, ShortCycler

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="hoefling_2024_core_readout_high_res")
def main(cfg: DictConfig) -> None:
    train_model(cfg)


def train_model(cfg: DictConfig) -> None:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

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

    train_loader = data.DataLoader(
        LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0, pin_memory=True
    )
    valid_loader = ShortCycler(dataloaders["validation"])

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    ### Model init
    # Assign missing n_neurons_dict to the model
    cfg.model.n_neurons_dict = {name: data_point.targets.shape[-1] for name, data_point in iter(train_loader)}

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    ### Logging
    log.info("Instantiating loggers...")
    log_folder = os.path.join(cfg.data.output_dir, cfg.exp_name)
    os.makedirs(log_folder, exist_ok=True)
    logger_array = []
    for _, logger_params in cfg.logger.items():
        logger = hydra.utils.instantiate(logger_params, save_dir=log_folder)
        logger_array.append(logger)

    ### Callbacks
    log.info("Instantiating callbacks...")
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]

    ### Trainer init
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ### Testing
    log.info("Starting testing!")
    test_loader = ShortCycler(dataloaders["test"])
    trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader], ckpt_path="best")


if __name__ == "__main__":
    main()
