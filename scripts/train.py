#!/usr/bin/env python3

import os
import pickle

import torch
import numpy as np
from openretina.hoefling_2022_configs import model_config, trainer_config
from openretina.hoefling_2022_data_io import natmov_dataloaders_v2
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout
from openretina.training import standard_early_stop_trainer as trainer


def main() -> None:
    print("Main")

    data_folder = "/gpfs01/euler/data/SharedFiles/projects/TP12/"
    data_path = os.path.join(data_folder, "2024-01-11_neuron_data_stim_8c18928_responses_99c71a0.pkl")
    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(data_path, "rb") as f:
        neuron_data_dict = pickle.load(f)

    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    dataloaders = natmov_dataloaders_v2(neuron_data_dict, movies_dict)
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)
    print("Init model")
    trainer_config["max_iter"] = 75

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=dataloaders,
        seed=1000,
        **trainer_config,
        wandb_logger=None,
    )
    print(f"Training finished with test_score: {test_score} and val_score: {val_score}")

    state_dict = model.state_dict()
    state_dict_path = "model_state_dict.tmp"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved state dict to {state_dict_path}")


if __name__ == "__main__":
    main()
