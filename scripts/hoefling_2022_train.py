#!/usr/bin/env python3

from typing import Callable
import argparse
import functools
import operator
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from openretina.hoefling_2024.configs import model_config, trainer_config
from openretina.hoefling_2024.data_io import (
    get_chirp_dataloaders,
    get_mb_dataloaders,
    natmov_dataloaders_v2,
)
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
import openretina.neuron_data_io
from openretina.neuron_data_io import make_final_responses
from openretina.plotting import save_figure
from openretina.training import save_model
from openretina.training import standard_early_stop_trainer as trainer
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.hoefling_2024.constants import RGC_GROUP_NAMES_DICT
from openretina.metrics import correlation_numpy as corr


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--datasets",
        type=str,
        default="natural",
        help="Underscore separated list of datasets, " "e.g. 'natural', 'chirp', 'mb', or 'natural_mb'",
    )
    parser.add_argument(
        "--cells",
        default="all",
        choices=["all", "on", "off", "on-off"],
    )
    parser.add_argument(
        "--max_id",
        type=int,
        default=99,
    )

    return parser.parse_args()


def plot_examples(
    dataloaders,
    example_field: str,
    ensemble_model,
    save_folder: str,
    device: str,
) -> None:
    test_sample = next(iter(dataloaders["test"][example_field]))

    inputs, targets = test_sample[:-1], test_sample[-1]

    ensemble_model.eval()
    ensemble_model.to(device)

    with torch.no_grad():
        reconstructions = ensemble_model(inputs[0].to(device), example_field)
    reconstructions = reconstructions.cpu().numpy().squeeze()

    targets_numpy = targets.cpu().numpy().squeeze()
    window = 750
    neuron = 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(np.arange(0, window), targets_numpy[:window, neuron], label="target")
    ax.plot(np.arange(30, window), reconstructions[:window, neuron], label="prediction")
    ax.set_title(f"Neuron {neuron} in field {example_field}")
    ax.legend()
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Firing rate (a.u.)")
    sns.despine()
    fig.savefig(f"{save_folder}/{example_field}_neuron_{neuron}.jpg", bbox_inches="tight", facecolor="w", dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(reconstructions, alpha=0.15)
    axes[0].plot(reconstructions.mean(axis=1), color="black", linestyle="--", label="mean reconstruction")
    axes[0].legend()
    axes[0].set_title("Reconstructed responses")
    axes[0].set_xlabel("Time (frames)")
    axes[0].set_ylabel("Firing rate (a.u.)")

    axes[1].plot(targets_numpy, alpha=0.15)
    axes[1].plot(targets_numpy.mean(axis=1), label="mean target", color="black", linestyle="--")
    axes[1].legend()
    axes[1].set_title("Target responses")
    axes[1].set_xlabel("Time (frames)")

    session_performance = corr(
        reconstructions,
        targets_numpy[30:],
        axis=1,
    ).mean()

    plt.suptitle(
        f"Reconstructed vs target responses for session {example_field} (n = {reconstructions.shape[1]} neurons). "
        f"\n Model average test correlation for the session: {session_performance:.3g}"
    )
    plt.tight_layout()
    fig.savefig(f"{save_folder}/{example_field}_all.jpg", bbox_inches="tight", facecolor="w", dpi=300)


def main(
    data_folder: str,
    save_folder: str,
    device: str,
    datasets: str,
    cells: str,
    max_id: int,
) -> None:

    if cells == "all":
        relevant_ids = np.array([id_ for id_ in RGC_GROUP_NAMES_DICT.keys() if id_ <= max_id])
    elif cells == "on":
        relevant_ids = np.array([id_ for id_, name in RGC_GROUP_NAMES_DICT.items() if "ON" in name and "OFF" not in name and id_ <= max_id])
    elif cells == "off":
        relevant_ids = np.array([id_ for id_, name in RGC_GROUP_NAMES_DICT.items() if "OFF" in name and "ON" not in name and id_ <= max_id])
    elif cells == "on-off":
        relevant_ids = np.array([id_ for id_, name in RGC_GROUP_NAMES_DICT.items() if "ON-OFF" in name and id_ <= max_id])
    else:
        raise ValueError(f"Unsupported option {cells=}")
    openretina.neuron_data_io.relevant_rgc_ids = relevant_ids
    print(f"Overwrote {openretina.neuron_data_io.relevant_rgc_ids=}")

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
    print(f"Training finished with test_score: {test_score:.5f} and val_score: {val_score:.5f}")

    save_model(
        model=model,
        save_folder=os.path.join(save_folder, "models"),
        model_name="SFB3d_core_SxF3d_readout_hoefling_2022",
    )

    # Plotting example fields
    model.eval()
    model.cpu()
    plot_folder = f"{save_folder}/plots_natural"
    os.makedirs(plot_folder, exist_ok=True)
    for example_field in model.readout_keys():
        plot_examples(joint_dataloaders, example_field, model, plot_folder, device)

    chirp_data_dict = make_final_responses(responses, response_type="chirp")  # type: ignore
    chirp_dataloaders = get_chirp_dataloaders(chirp_data_dict, train_chunk_size=100)
    plot_folder = f"{save_folder}/plots_chirp"
    os.makedirs(plot_folder, exist_ok=True)
    for example_field in model.readout_keys():
        plot_examples(chirp_dataloaders, example_field, model, plot_folder, device)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
