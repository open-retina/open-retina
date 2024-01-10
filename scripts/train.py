#!/usr/bin/env python3

import collections.abc
import pickle
import pprint

import numpy as np
import torch
import wandb

from openretina.dataloaders import dataloaders_from_dictionaries
from openretina.hoefling_2022_configs import model_config, trainer_config
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout
from openretina.misc import CustomPrettyPrinter
from openretina.plotting import play_stimulus
from openretina.training import standard_early_stop_trainer as trainer


def main() -> None:
    print("Main")

    dataloader_store_folder = "/gpfs01/euler/data/SharedFiles/projects/TP12/"
    with open(dataloader_store_folder + "dataloaders_stim_8c18928_responses_99c71a0.pkl", "rb") as f:
        stim_dataloaders_dict = pickle.load(f)

    with open(dataloader_store_folder + "movies_8c18928.pkl", "rb") as f:
        movies_dict = pickle.load(f)

    dataloaders = dataloaders_from_dictionaries(stim_dataloaders_dict, movies_dict)
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)
    print("Init model")

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=dataloaders,
        seed=1000,
        **trainer_config,
        wandb_logger=None,
    )
    print(f"Training finished with test_score: {test_score} and val_score: {val_score}")


if __name__ == "__main__":
    main()

