#!/usr/bin/env python3
from typing import Callable
import argparse
import functools
import operator
import time
import os
import pickle

import torch
import lightning
from lightning.pytorch.loggers import CSVLogger

from openretina.neuron_data_io import make_final_responses
from openretina.models.autoencoder import SparsityMSELoss, Autoencoder, ActivationsDataset
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.hoefling_2024.configs import model_config
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
from openretina.hoefling_2024.data_io import (
    get_chirp_dataloaders,
    get_mb_dataloaders,
    natmov_dataloaders_v2,
)
from openretina.cyclers import LongCycler
from openretina.hoefling_2024.nnfabrik_model_loading import Center


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--sparsity_factor", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=1000, help="Hidden dim for autoencoder")
    parser.add_argument(
        "--datasets",
        type=str,
        default="natural",
        help="Underscore separated list of datasets, " "e.g. 'natural', 'chirp', 'mb', or 'natural_mb'",
    )

    return parser.parse_args()


def generate_neuron_activations(data_folder: str, dataset_names_list: list[str], device: str) -> list[torch.Tensor]:
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
    state_dict_path = "models/debug_2024-07-01/models/SFB3d_core_SxF3d_readout_hoefling_2022_2024-07-01_model_weights.pt"
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    center_class = Center(target_mean=[0.0, 0.0])
    center_class(model)
    model.to(device)
    print(f"Initialized model from {state_dict_path}")

    # generate model outputs
    # We currently generate outputs for each readout key for each training example
    # This likely results in duplicate examples, as the training examples for each readout key are
    # the same or at least similar.
    outputs_model: list[torch.Tensor] = []
    readout_keys_list = model.readout.readout_keys()
    time_generation_start = time.time()
    for batch_no, (_, data) in enumerate(LongCycler(joint_dataloaders["train"])):
        all_activations_list = []
        for readout_key in readout_keys_list:
            with torch.no_grad():
                activations = model.forward(data.inputs.to(device), readout_key)
                all_activations_list.append(activations)
        all_activations = torch.cat(all_activations_list, dim=-1)
        # Put each example of the batch individually into the list by using extend instead of append
        outputs_model.extend(all_activations.cpu())
        if (batch_no+1) % 20 == 0:
            print(f"Generated {batch_no+1} batches")
    print(f"Generated {len(outputs_model)} examples in {time.time()-time_generation_start:.1f}s")
    outputs_model_single_tensor = torch.stack(outputs_model)
    # I/O is more efficient on a single tensor
    return outputs_model_single_tensor


def main(
    data_folder: str,
    save_folder: str,
    device: str,
    datasets: str,
    sparsity_factor: float,
    hidden_dim: int,
) -> None:
    dataset_names_list = datasets.split("_")
    for name in dataset_names_list:
        if name not in {"natural", "chirp", "mb"}:
            raise ValueError(f"Unsupported dataset name {name}")

    outputs_model_path = f"{save_folder}/../outputs_model.pkl"
    print(outputs_model_path)
    if os.path.exists(outputs_model_path):
        with open(outputs_model_path, "rb") as fr:
            outputs_model = torch.load(fr)
        print(f"Loaded model outputs from {outputs_model_path}")
    else:
        outputs_model = generate_neuron_activations(data_folder, dataset_names_list, device)
        with open(outputs_model_path, "wb") as fw:
            # it's more efficient to save a single tensor instead of a list of tensors
            torch.save(outputs_model, fw)
        print(f"Saved model outputs to {outputs_model_path}")

    # How to treat activations across different session?
    # - Train independent autoencoders?
    # - Same autoencoder but with zero weights for neurons not in that session?
    # - Just sample all data, or is input data the same (ignore outputs, feed input through all data_keys)
    sparsity_mse_loss = SparsityMSELoss(sparsity_factor=sparsity_factor)
    num_model_neurons = outputs_model[0].shape[-1]
    sparse_autoencoder = Autoencoder(num_model_neurons, hidden_dim, sparsity_mse_loss)
    sparse_autoencoder.to(device)
    activations_dataset = ActivationsDataset(outputs_model)
    # when num_workers > 0 the docker container needs more shared memory
    train_loader = torch.utils.data.DataLoader(activations_dataset, batch_size=30, num_workers=0)

    lightning_folder = f"{save_folder}_{hidden_dim}_neurons"
    csv_logger = CSVLogger(lightning_folder)
    trainer = lightning.Trainer(max_epochs=10, default_root_dir=lightning_folder, logger=csv_logger)
    trainer.fit(model=sparse_autoencoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
