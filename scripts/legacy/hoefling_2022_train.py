#!/usr/bin/env python3

import argparse
import functools
import operator
import os
import pickle
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from openretina.data_io.hoefling_2024 import (
    SFB3d_core_SxF3d_readout,
    get_chirp_dataloaders,
    get_mb_dataloaders,
    natmov_dataloaders_v2,
)
from openretina.data_io.hoefling_2024.responses import make_final_responses
from openretina.legacy.configs import model_config, trainer_config
from openretina.legacy.training import save_model
from openretina.legacy.training import standard_early_stop_trainer as trainer
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.utils.plotting import save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="natural",
        help="Underscore separated list of datasets, " "e.g. 'natural', 'chirp', 'mb', or 'natural_mb'",
    )

    return parser.parse_args()


def main(
    data_folder: str,
    save_folder: str,
    device: str,
    datasets: str,
) -> None:
    dataset_names_list = datasets.split("_")
    for name in dataset_names_list:
        if name not in {"natural", "chirp", "mb"}:
            raise ValueError(f"Unsupported dataset name {name}")

    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging.h5")
    responses = load_h5_into_dict(data_path_responses)

    dataloader_list = []

    dataloader_name_to_function: dict[str, Callable] = {
        "chirp": get_chirp_dataloaders,
        "mb": get_mb_dataloaders,
        "natural": functools.partial(natmov_dataloaders_v2, movies_dictionary=movies_dict, seed=1000),
    }
    for dataset_name in dataset_names_list:
        data_dict = make_final_responses(responses, response_type=dataset_name)  # type: ignore
        dataloader_fn = dataloader_name_to_function[dataset_name]
        dataloader = dataloader_fn(data_dict, train_chunk_size=100)
        dataloader_list.append(dataloader)

    def get_joint_dataloader(dataloader_list: list, set_name: str):
        dict_list = [d[set_name] for d in dataloader_list if set_name in d]
        if len(dict_list) == 0:
            print(f"Warn: Using training data for {set_name=}")
            dict_list = [dataloader_list[0]["train"]]

        dataloader = functools.reduce(operator.or_, dict_list)
        return dataloader

    joint_dataloaders = {
        "train": get_joint_dataloader(dataloader_list, "train"),
        "validation": get_joint_dataloader(dataloader_list, "validation"),
        "test": get_joint_dataloader(dataloader_list, "test"),
    }
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=joint_dataloaders, seed=42)  # type: ignore
    print("Init model")

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=joint_dataloaders,
        seed=1000,
        **trainer_config,  # type: ignore
        wandb_logger=None,
        device=device,
    )
    print(f"Training finished with test_score: {test_score} and val_score: {val_score}")

    save_model(
        model=model,
        dataloaders=joint_dataloaders,
        save_folder=os.path.join(save_folder, "models"),
        model_name="SFB3d_core_SxF3d_readout_hoefling_2022",
    )

    # Plotting an example field
    sample_loader = joint_dataloaders["train"]
    sample_session = list(sample_loader.keys())[0]
    test_sample = next(iter(joint_dataloaders["test"][sample_session]))

    input_samples = test_sample.inputs
    targets = test_sample.targets

    model.eval()
    model.cpu()

    with torch.no_grad():
        reconstructions = model(input_samples.cpu(), sample_session)
    reconstructions = reconstructions.cpu().numpy().squeeze()

    targets = targets.cpu().numpy().squeeze()
    if len(targets.shape) == 3:
        print("Targets still have a batch dimensions, taking the first element")
        targets = targets[0]
        assert len(reconstructions.shape) == 3
        reconstructions = reconstructions[0]

    window = min(targets.shape[0], 500)
    neuron = 2
    plt.plot(np.arange(0, window), targets[:window, neuron], label="target")
    window = min(reconstructions.shape[0], 500)
    plt.plot(np.arange(30, window + 30), reconstructions[:window, neuron], label="prediction")
    plt.legend()
    sns.despine()
    save_figure("mouse_reconstruction_example.pdf", os.path.join(save_folder, "figures"))


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
