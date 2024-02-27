#!/usr/bin/env python3

import argparse
import os

import numpy as np

from openretina.hoefling_2022_configs import model_config, trainer_config
from openretina.hoefling_2022_data_io import natmov_dataloaders_v2
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout
from openretina.maheswaranathan_2023_data_io import CLIP_LENGTH, load_all_sessions
from openretina.training import save_model
from openretina.training import standard_early_stop_trainer as trainer


def main(data_folder) -> None:
    data_path = os.path.join(data_folder, "baccus_data/neural_code_data/ganglion_cell_data/")

    neuron_data_dict, movies_dict = load_all_sessions(data_path, fr_normalization=1)

    movie_length = movies_dict["train"].shape[1]

    dataloaders = natmov_dataloaders_v2(
        neuron_data_dict,
        movies_dict,
        train_chunk_size=50,
        batch_size=32,
        clip_length=90,
        num_clips=len(np.arange(0, movie_length // CLIP_LENGTH)),
        num_val_clips=20,
    )
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

    save_model(
        model=model, save_folder=os.path.join(data_folder, "models"), model_name="SFB3d_core_SxF3d_readout_salamander"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/")

    args = parser.parse_args()

    main(**vars(args))
