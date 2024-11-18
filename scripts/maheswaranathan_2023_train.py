#!/usr/bin/env python3

import os

import hydra
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.cyclers import LongCycler
from openretina.data_io.maheswaranathan_2023.constants import CLIP_LENGTH
from openretina.data_io.maheswaranathan_2023.dataloader import multiple_movies_dataloaders
from openretina.data_io.maheswaranathan_2023.neuron_data_io import load_all_sessions


@hydra.main(version_base=None, config_path="../configs", config_name="maheswaranathan_core_readout")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_path = os.path.join(cfg.data.base_folder, "baccus_data/neural_code_data/ganglion_cell_data/")

    neuron_data_dict, movies_dict = load_all_sessions(data_path, fr_normalization=1)

    dataloaders = multiple_movies_dataloaders(
        neuron_data_dict,
        movies_dict,  # type: ignore
        train_chunk_size=50,
        batch_size=64,
        clip_length=CLIP_LENGTH,
        num_val_clips=20,
    )

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), **cfg.dataloader)
    valid_loader = data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), **cfg.dataloader)

    ### Model init
    # Assign missing n_neurons_dict to the model
    cfg.model.n_neurons_dict = {name: data_point.targets.shape[-1] for name, data_point in iter(train_loader)}
    model = hydra.utils.instantiate(cfg.model)

    ### Logging
    log_folder = os.path.join(cfg.data.save_folder, cfg.exp_name)
    os.makedirs(log_folder, exist_ok=True)
    logger_array = []
    for logger_name, logger_params in cfg.loggers.items():
        logger = hydra.utils.instantiate(logger_params, save_dir=log_folder)
        logger_array.append(logger)

    ### Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_correlation", save_top_k=3, mode="max", verbose=True)
    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]
    callbacks.append(model_checkpoint)

    ### Trainer init
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ### Testing
    test_loader = data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), **cfg.dataloader)
    trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader], ckpt_path="best")
    trainer.save_checkpoint(os.path.join(log_folder, "model.ckpt"))


if __name__ == "__main__":
    main()
