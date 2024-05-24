#!/usr/bin/env python3

import argparse
import yaml
import functools
import operator
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from neuralpredictors.measures import corr

from openretina.h5_handling import load_h5_into_dict
from openretina.neuron_data_io import make_final_responses
from openretina.hoefling_2022_configs import model_config, trainer_config
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout
from openretina.hoefling_2022_data_io import (
    get_chirp_dataloaders,
    get_mb_dataloaders,
    natmov_dataloaders_v2,
)
from openretina.plotting import save_figure
from openretina.training import save_model
from openretina.training import standard_early_stop_trainer as trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--config", type=str, help="Path to the train config", default="configs/hoefling_2022.yaml")
    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--datasets", type=str, default="natural",
                        help="Underscore separated list of datasets, "
                             "e.g. 'natural', 'chirp', 'mb', or 'natural_mb'")

    return parser.parse_args()


def main(
        config: str,
        data_folder: str,
        save_folder: str,
        device: str,
        datasets: str,
) -> None:
    dataset_names_list = datasets.split("_")
    for name in dataset_names_list:
        if name not in {"natural", "chirp", "mb"}:
            raise ValueError(f"Unsupported dataset name {name}")

    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    #data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging.h5")
    data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging_single_neuron.h5")
    responses = load_h5_into_dict(data_path_responses)

    dataloader_list = []

    dataloader_name_to_function = {
        "chirp": get_chirp_dataloaders,
        "mb": get_mb_dataloaders,
        "natural": functools.partial(natmov_dataloaders_v2, movies_dictionary=movies_dict, seed=1000),
    }
    for dataset_name in dataset_names_list:
        data_dict = make_final_responses(responses, response_type=dataset_name)
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
    if False:
        print(f"Set validation/test dataloader to train dataloader")
        joint_dataloaders["validation"] = joint_dataloaders["train"]
        joint_dataloaders["test"] = joint_dataloaders["train"]

    model = SFB3d_core_SxF3d_readout(**config_dict["model_config"], dataloaders=joint_dataloaders, seed=42)
    print(f"Init model")

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=joint_dataloaders,
        seed=1000,
        **config_dict["trainer_config"],
        wandb_logger=None,
        device=device,
    )
    print(f"Training finished with test_score: {test_score} and val_score: {val_score}")

    save_model(
        model=model,
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
    reconstructions = reconstructions.cpu().numpy().squeeze(axis=0)

    targets = targets.cpu().numpy().squeeze(axis=0)
    if len(targets.shape) == 3:
        print("Targets still have a batch dimensions, taking the first element")
        targets = targets[0]
        assert len(reconstructions.shape) == 3
        reconstructions = reconstructions[0]

    for neuron in range(targets.shape[1]):
        plt.plot(np.arange(0, targets.shape[0]), targets[:, neuron], label="target")
        assert targets.shape[0] > reconstructions.shape[0]
        time_lag_reconstructions = targets.shape[0] - reconstructions.shape[0]
        time_rec = np.arange(0, reconstructions.shape[0]) + time_lag_reconstructions
        plt.plot(time_rec, reconstructions[:, neuron], label="prediction")
        neuron_corr = corr(reconstructions[:, neuron], targets[time_lag_reconstructions:, neuron])
        plt.title(f"Correlation: {neuron_corr}")
        plt.legend()
        sns.despine()
        save_figure(f"corr_{int(neuron_corr*100):02}_neuron_{neuron}.pdf", os.path.join(save_folder, "figures"))
        plt.clf()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
