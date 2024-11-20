#!/usr/bin/env python3

import argparse
import os
from functools import partial
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.insilico.stimulus_optimization.objective import (
    AbstractObjective,
    ContrastiveNeuronObjective,
    SingleNeuronObjective,
    SliceMeanReducer,
)
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.legacy.hoefling_configs import STIMULUS_RANGE_CONSTRAINTS
from openretina.models.sparse_autoencoder import Autoencoder, AutoencoderWithModel
from openretina.utils.nnfabrik_model_loading import Center, load_ensemble_retina_model_from_directory
from openretina.utils.plotting import plot_stimulus_composition


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to a model, if this is a directory loads an ensemble model, " "otherwise calls torch.load",
    )
    parser.add_argument("--autoencoder_path", required=True, type=str)
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--use_contrastive_objective", action="store_true")

    return parser.parse_args()


def load_model(path: str, device: str):
    if os.path.isdir(path):
        _, model = load_ensemble_retina_model_from_directory(path, device)
        print(f"Initialized ensemble model from {path}")
    else:
        model = torch.load(path, map_location=torch.device(device))

    center_readout = Center(target_mean=(0.0, 0.0))
    center_readout(model)
    return model


def main(
    model_path: str, autoencoder_path: str, save_folder: str, device: str, use_contrastive_objective: bool
) -> None:
    model = load_model(model_path, device)
    autoencoder = Autoencoder.load_from_checkpoint(autoencoder_path)  # type: ignore
    autoencoder_with_model = AutoencoderWithModel(model, autoencoder)
    if use_contrastive_objective:
        objective_class: Type[AbstractObjective] = ContrastiveNeuronObjective
    else:
        objective_class = SingleNeuronObjective

    # from controversial stimuli: (2, 50, 18, 16): (channels, time, height, width)
    stimulus_shape = (1, 2, 50, 18, 16)

    response_reducer = SliceMeanReducer(axis=0, start=10, length=10)
    stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately(
        min_max_values=(
            (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
            (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
        ),
        norm=STIMULUS_RANGE_CONSTRAINTS["norm"],
    )
    stimulus_regularizing_loss = RangeRegularizationLoss(
        min_max_values=(
            (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
            (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
        ),
        max_norm=STIMULUS_RANGE_CONSTRAINTS["norm"],
        factor=0.1,
    )

    for neuron_id in range(autoencoder.hidden_dim()):
        print(f"Generating MEI for {neuron_id=}")
        objective = objective_class(
            autoencoder_with_model,
            neuron_idx=neuron_id,  # type: ignore
            data_key=None,
            response_reducer=response_reducer,
        )

        stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)

        stimulus.data = stimulus_postprocessor.process(stimulus.data)
        optimizer_init_fn = partial(torch.optim.SGD, lr=100.0)

        # Throws: RuntimeError: Expected all tensors to be on the same device,
        # but found at least two devices, cuda:0 and cpu!
        # reason probably: not all model parameters are on gpu(?)
        optimize_stimulus(
            stimulus,
            optimizer_init_fn,
            objective,
            OptimizationStopper(max_iterations=10),
            stimulus_regularization_loss=stimulus_regularizing_loss,
            stimulus_postprocessor=stimulus_postprocessor,
        )
        stimulus_np = stimulus[0].cpu().numpy()
        fig, axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
        plot_stimulus_composition(
            stimulus=stimulus_np,
            temporal_trace_ax=axes[0, 0],  # type: ignore
            freq_ax=axes[0, 1],  # type: ignore
            spatial_ax=axes[1, 0],  # type: ignore
            highlight_x_list=[(40, 49)],
        )

        mei_str = "cei" if use_contrastive_objective else "mei"
        img_path = f"{save_folder}/{mei_str}_{neuron_id}_slice_mean.pdf"
        np_path = f"{save_folder}/{mei_str}_{neuron_id}_slice_mean.npy"
        with open(np_path, "wb") as fwb:
            np.save(fwb, stimulus_np)
        fig.savefig(img_path, bbox_inches="tight", facecolor="w", dpi=300)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
