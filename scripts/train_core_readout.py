#!/usr/bin/env python3

import os
from typing import Literal

import hydra
import lightning
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.data_io.cyclers import LongCycler, ShortCycler
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.models.core_readout import CoreReadout
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.utils.model_utils import OptimizerResetCallback


@hydra.main(version_base=None, config_path="../configs", config_name="example_train_core_readout")
def main(conf: DictConfig) -> None:
    hydra.utils.call(conf.matmul_precision)

    movies_path = get_local_file_path(conf.movies_path, conf.cache_folder)
    movies_dict = MoviesTrainTestSplit.from_pickle(movies_path)

    data_path_responses = get_local_file_path(conf.responses_path, conf.cache_folder)
    responses = load_h5_into_dict(data_path_responses)
    filtered_responses = filter_responses(responses, **OmegaConf.to_object(conf.quality_checks))  # type: ignore

    data_dict = make_final_responses(filtered_responses, response_type="natural")  # type: ignore

    dataloaders = hydra.utils.call(
        conf.natmov_dataloader, neuron_data_dictionary=data_dict, movies_dictionary=movies_dict
    )

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = DataLoader(LongCycler(dataloaders["train"], shuffle=True), **conf.dataloader)
    valid_loader = DataLoader(ShortCycler(dataloaders["validation"]), **conf.dataloader)

    # max_pool3d_with_indices does not have a deterministic implementation in pytorch yet
    deterministic: bool | Literal["warn"] = "warn" if conf.seed is not None else False
    if conf.seed is not None:
        seed_everything(conf.seed)

    n_neurons_dict = {name: data_point.targets.shape[-1] for name, data_point in iter(train_loader)}
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
    model_checkpoint = ModelCheckpoint(monitor="val_correlation", save_top_k=3, mode="max", verbose=True)
    callbacks: list = [model_checkpoint, OptimizerResetCallback()]
    for callback_params in conf.get("training_callbacks", {}).values():
        callbacks.append(hydra.utils.instantiate(callback_params))

    trainer = lightning.Trainer(
        default_root_dir=conf.save_folder,
        logger=logger_array,
        callbacks=callbacks,
        deterministic=deterministic,
        **conf.trainer,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # test
    test_loader = DataLoader(ShortCycler(dataloaders["test"]), **conf.dataloader)
    trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader], ckpt_path="best")
    trainer.save_checkpoint(os.path.join(log_folder, "model.ckpt"))


if __name__ == "__main__":
    main()
