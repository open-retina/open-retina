#!/usr/bin/env python3

import os
import pickle

import torch
import lightning
import hydra
from omegaconf import DictConfig

from openretina.cyclers import LongCycler
from openretina.hoefling_2024.data_io import (
    natmov_dataloaders_v2,
)
from openretina.neuron_data_io import make_final_responses
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.models.core_readout import CoreReadout


@hydra.main(version_base=None, config_path="../example_configs", config_name="train_core_readout")
def main(conf: DictConfig) -> None:
    data_folder = os.path.expanduser(conf.data_folder)
    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    data_path_responses = os.path.join(data_folder, "2024-08-14_neuron_data_responses_484c12d_djimaging.h5")
    responses = load_h5_into_dict(data_path_responses)

    data_dict = make_final_responses(responses, response_type="natural")  # type: ignore
    dataloaders = natmov_dataloaders_v2(data_dict, movies_dictionary=movies_dict, train_chunk_size=100, seed=1000)

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), **conf.dataloader)
    valid_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), **conf.dataloader)
    test_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), **conf.dataloader)

    n_neurons_dict = {
       name: data_point.targets.shape[-1] for name, data_point in iter(train_loader)
    }
    model = CoreReadout(
        n_neurons_dict=n_neurons_dict,
        **conf.core_readout,
    )

    log_folder = os.path.join(conf.save_folder, conf.exp_name)
    os.makedirs(log_folder, exist_ok=True)
    logger_array = []
    for logger_name, logger_params in conf.loggers.items():
        logger = hydra.utils.instantiate(logger_params, save_dir=log_folder, name=logger_name)
        logger_array.append(logger)
    callbacks = []
    for callback_params in conf.get("training_callbacks", {}).values():
        callbacks.append(hydra.utils.instantiate(callback_params))

    trainer = lightning.Trainer(max_epochs=conf.max_epochs, default_root_dir=conf.save_folder,
                                logger=logger_array, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
    trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader])
    trainer.save_checkpoint(os.path.join(log_folder, "model.ckpt"))


if __name__ == "__main__":
    main()
