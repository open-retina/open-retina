#!/usr/bin/env python3

import os
import pickle

import hydra
import lightning
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from openretina.cyclers import LongCycler
from openretina.hoefling_2024.data_io import (
    natmov_dataloaders_v2,
)
from openretina.models.core_readout import CoreReadout
from openretina.neuron_data_io import filter_responses, make_final_responses
from openretina.utils.h5_handling import load_h5_into_dict


class OptimizerResetCallback(Callback):
    def __init__(self):
        super().__init__()
        self.prev_lr = None  # This will store the previous learning rate

    def on_validation_end(self, trainer, pl_module):
        # Get the current learning rate from the optimizer
        optims = pl_module.optimizers()
        try:
            optim = optims[0]
        except:  # noqa
            optim = optims
        current_lr = optim.param_groups[0]["lr"]

        # Compare with the previous learning rate
        if self.prev_lr is not None and current_lr < self.prev_lr:
            print(f"Learning rate decreased from {self.prev_lr} to {current_lr}. Resetting optimizer.")
            # Reset the optimizer if the learning rate has decreased
            params_dict = optim.param_groups[0]
            # below could be written shorter
            new_optimizer = torch.optim.AdamW(
                pl_module.parameters(),
                lr=current_lr,
                betas=params_dict["betas"],
                eps=params_dict["eps"],
                weight_decay=params_dict["weight_decay"],
                amsgrad=params_dict["amsgrad"],
                maximize=params_dict["maximize"],
                foreach=params_dict["foreach"],
                capturable=params_dict["capturable"],
                differentiable=params_dict["differentiable"],
                fused=params_dict["fused"],
            )
            trainer.optimizers = [new_optimizer]  # Replace the optimizer in the trainer

        self.prev_lr = current_lr


@hydra.main(version_base=None, config_path="../example_configs", config_name="train_core_readout")
def main(conf: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    data_folder = os.path.expanduser(conf.data_folder)
    movies_path = os.path.join(data_folder, conf.movies_filename)
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    data_path_responses = os.path.join(data_folder, conf.responses_filename)
    responses = load_h5_into_dict(data_path_responses)
    filtered_responses = filter_responses(responses, **OmegaConf.to_object(conf.quality_checks))  # type: ignore

    data_dict = make_final_responses(filtered_responses, response_type="natural")  # type: ignore
    dataloaders = natmov_dataloaders_v2(data_dict, movies_dictionary=movies_dict, train_chunk_size=100, seed=1000)

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), **conf.dataloader)
    valid_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), **conf.dataloader)

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
        max_epochs=conf.max_epochs,
        default_root_dir=conf.save_folder,
        precision="16-mixed",
        logger=logger_array,
        callbacks=callbacks,
        accumulate_grad_batches=1,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # test
    test_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), **conf.dataloader)
    trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader], ckpt_path="best")
    trainer.save_checkpoint(os.path.join(log_folder, "model.ckpt"))


if __name__ == "__main__":
    main()
