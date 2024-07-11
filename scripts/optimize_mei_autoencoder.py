#!/usr/bin/env python3

from typing import Type
import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from openretina.hoefling_2024.constants import STIMULUS_RANGE_CONSTRAINTS
from openretina.optimization.objective import (AbstractObjective, SingleNeuronObjective,
                                               ContrastiveNeuronObjective, MeanReducer)
from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.optimization_stopper import OptimizationStopper
from openretina.optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.plotting import plot_stimulus_composition
from openretina.hoefling_2024.nnfabrik_model_loading import load_ensemble_retina_model_from_directory, Center
from openretina.models.autoencoder import Autoencoder, AutoencoderWithModel


ENSEMBLE_MODEL_PATH = ("/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/"
                       "models/nonlinear/9d574ab9fcb85e8251639080c8d402b7")


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--autoencoder_path", required=True, type=str)
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--use_contrastive_objective", action="store_true")

    return parser.parse_args()


def load_model(path: str = ENSEMBLE_MODEL_PATH, device: str = "cuda"):
    center_readout = Center(target_mean=(0.0, 0.0))
    data_info, ensemble_model = load_ensemble_retina_model_from_directory(
        path, device, center_readout=center_readout)
    print(f"Initialized ensemble model from {path}")
    return ensemble_model


def main(autoencoder_path: str, save_folder: str, device: str, use_contrastive_objective: bool) -> None:
    device = "cuda"
    model = load_model(device=device)
    autoencoder = Autoencoder.load_from_checkpoint(autoencoder_path)
    autoencoder_with_model = AutoencoderWithModel(model, autoencoder)
    if use_contrastive_objective:
        objective_class: Type[AbstractObjective] = ContrastiveNeuronObjective
    else:
        objective_class = SingleNeuronObjective

    # from controversial stimuli: (2, 50, 18, 16): (channels, time, height, width)
    stimulus_shape = (1, 2, 50, 18, 16)

    mean_response_reducer = MeanReducer()
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
        objective = objective_class(autoencoder_with_model, neuron_idx=neuron_id,  # type: ignore
                                    data_key=None, response_reducer=mean_response_reducer)

        stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)

        stimulus.data = stimulus_postprocessor.process(stimulus.data)
        optimizer_init_fn = partial(torch.optim.SGD, lr=10.0)

        # Throws: RuntimeError: Expected all tensors to be on the same device,
        # but found at least two devices, cuda:0 and cpu!
        # reason probably: not all model parameters are on gpu(?)
        optimize_stimulus(
            stimulus,
            optimizer_init_fn,
            objective,
            OptimizationStopper(max_iterations=100),
            stimulus_regularization_loss=stimulus_regularizing_loss,
            stimulus_postprocessor=stimulus_postprocessor,
        )
        stimulus_np = stimulus[0].cpu().numpy()
        fig, axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
        plot_stimulus_composition(
            stimulus=stimulus_np,
            temporal_trace_ax=axes[0, 0],
            freq_ax=axes[0, 1],
            spatial_ax=axes[1, 0],
            highlight_x_list=[(40, 49)],
        )

        mei_str = "cei" if use_contrastive_objective else "mei"
        img_path = f"{save_folder}/{mei_str}_{neuron_id}.pdf"
        np_path = f"{save_folder}/{mei_str}_{neuron_id}.npy"
        with open(np_path, "wb") as fwb:
            np.save(fwb, stimulus_np)
        fig.savefig(img_path, bbox_inches="tight", facecolor="w", dpi=300)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
