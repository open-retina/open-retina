#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from openretina.hoefling_2024.configs import model_config, trainer_config
from openretina.hoefling_2024.data_io import natmov_dataloaders_v2
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
from openretina.maheswaranathan_2023_data_io import CLIP_LENGTH, load_all_sessions
from openretina.training import standard_early_stop_trainer as trainer

import wandb

wandb.login()


sweep_configuration = {
    "method": "bayes",
    "name": "Maheswaranathan 2023 sweep",
    "metric": {"goal": "maximize", "name": "val_corr"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "lr_init": {"max": 0.1, "min": 0.0001},
        "lr_decay_steps": {"values": [1, 2, 3, 4, 5]},
        "train_chunk_size": {"values": [50, 60, 90]},
        "nonlinearity": {"values": ["ELU", "ReLU", "GELU", "Softplus"]},
        "conv_type": {"values": ["separable", "custom_separable", "full"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="hoefling_2022_bayesopt")


def main():
    run = wandb.init()

    # note that we define values from `wandb.config`
    # instead of defining hard values
    trainer_config["lr_init"] = wandb.config.lr_init
    trainer_config["lr_decay_steps"] = wandb.config.lr_decay_steps
    model_config["nonlinearity"] = wandb.config.nonlinearity
    model_config["conv_type"] = wandb.config.conv_type

    # Load models and data
    base_folder = "/Data/"
    data_path = os.path.join(base_folder, "baccus_data/neural_code_data/ganglion_cell_data/")
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
    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)

    test_score, val_score, output, model_state = trainer(
        model=model,
        dataloaders=dataloaders,
        seed=1000,
        **trainer_config,
        wandb_logger=run,
    )

    # Plotting
    val_field = "1_ventral1_20210929"
    val_sample = next(iter(dataloaders["validation"][val_field]))
    input_samples = val_sample.inputs
    targets = val_sample.targets

    with torch.no_grad():
        reconstructions = model(input_samples.to("cuda:0"), val_field)
    reconstructions = reconstructions.cpu().numpy().squeeze()
    targets = targets.cpu().numpy().squeeze()

    # Plot the reconstruction
    neuron = 1
    fig, axes = plt.subplots(3, 5, figsize=(20, 5), sharey="row", sharex="col")
    for trace_chunk in range(targets.shape[0]):
        ax_idx_1 = trace_chunk // 5
        ax_idx_2 = trace_chunk % 5
        ax = axes[ax_idx_1, ax_idx_2]
        ax.plot(targets[trace_chunk, 30:, neuron], label="target")
        ax.plot(reconstructions[trace_chunk, :, neuron], label="prediction")

        # Set x and y labels for only outer subplots
        if ax_idx_1 == 2:  # Bottom row
            ax.set_xlabel("Frames")
        if ax_idx_2 == 0:  # Leftmost column
            ax.set_ylabel("Firing rate")

        # Only turn on x-axis labels for the bottom row
        if ax_idx_1 == 2:
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)

        # Only turn on y-axis labels for the leftmost column
        if ax_idx_2 == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

        # Place the legend outside of the subplots
        axes[0, 0].legend()

    sns.despine()
    plt.tight_layout()

    wandb.log({"reconstruction": fig})


if __name__ == "__main__":
    wandb.agent(sweep_id, function=main, count=50)
