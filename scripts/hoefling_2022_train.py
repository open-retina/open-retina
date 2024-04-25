#!/usr/bin/env python3

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from openretina.hoefling_2022_configs import model_config, trainer_config
from openretina.hoefling_2022_data_io import natmov_dataloaders_v2
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout
from openretina.plotting import save_figure
from openretina.training import save_model
from openretina.training import standard_early_stop_trainer as trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

    return parser.parse_args()


def main(data_folder: str, save_folder: str, device: str) -> None:
    data_path = os.path.join(data_folder, "2024-01-11_neuron_data_stim_8c18928_responses_99c71a0.pkl")
    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(data_path, "rb") as f:
        neuron_data_dict = pickle.load(f)

    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    dataloaders = natmov_dataloaders_v2(neuron_data_dict, movies_dict)
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)
    model.to(device)
    print(f"Init model for {device=}")

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=dataloaders,
        seed=1000,
        **trainer_config,
        wandb_logger=None,
    )
    print(f"Training finished with test_score: {test_score} and val_score: {val_score}")

    save_model(
        model=model,
        save_folder=os.path.join(save_folder, "models"),
        model_name="SFB3d_core_SxF3d_readout_hoefling_2022",
    )

    ## Plotting an example field
    sample_loader = dataloaders.get("train", dataloaders)
    sample_session = list(sample_loader.keys())[0]
    test_sample = next(iter(dataloaders["test"][sample_session]))

    input_samples = test_sample.inputs
    targets = test_sample.targets

    model.eval()
    model.cpu()

    with torch.no_grad():
        reconstructions = model(input_samples.cpu(), sample_session)
    reconstructions = reconstructions.cpu().numpy().squeeze()

    targets = targets.cpu().numpy().squeeze()
    window = 500
    neuron = 2
    plt.plot(np.arange(0, window), targets[:window, neuron], label="target")
    plt.plot(np.arange(30, window + 30), reconstructions[:window, neuron], label="prediction")
    plt.legend()
    sns.despine()
    save_figure("mouse_reconstruction_example.pdf", os.path.join(save_folder, "figures"))


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
