#!/usr/bin/env python3

import os
from functools import partial
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from openretina.hoefling_2024.configs import STIMULUS_RANGE_CONSTRAINTS
from openretina.optimization.objective import SingleNeuronObjective, MeanReducer
from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.optimization_stopper import OptimizationStopper
from openretina.optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.plotting import plot_stimulus_composition


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize MEIs for all output neurons")

    parser.add_argument("model_path", type=str, help="Path to the pt file of the model")
    parser.add_argument("save_folder", type=str, help="Output folder", default=".")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main(model_path: str, save_folder: str, device: str) -> None:

    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    print(f"Init model from {model_path=}")
    os.makedirs(save_folder, exist_ok=True)

    # from controversial stimuli: (2, 50, 18, 16): (channels, time, height, width)
    stimulus_shape = (1, 2, 50, 18, 16)

    mean_response_reducer = MeanReducer()
    for session_id in model.readout.keys():
        for neuron_id in range(model.readout[session_id].outdims):
            print(f"Generating MEI for {session_id=} {neuron_id=}")
            objective = SingleNeuronObjective(model, neuron_idx=neuron_id,
                                              data_key=session_id, response_reducer=mean_response_reducer)
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately(
                min_max_values=(
                    (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
                    (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
                ),
                norm=STIMULUS_RANGE_CONSTRAINTS["norm"],
            )
            stimulus.data = stimulus_postprocessor.process(stimulus.data)
            optimizer_init_fn = partial(torch.optim.SGD, lr=10.0)
            stimulus_regularizing_loss = RangeRegularizationLoss(
                min_max_values=(
                    (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
                    (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
                ),
                max_norm=STIMULUS_RANGE_CONSTRAINTS["norm"],
                factor=0.1,
            )
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
            fig_axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
            axes: np.ndarray = fig_axes[1]  # type: ignore
            plot_stimulus_composition(
                stimulus=stimulus_np,
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(40, 49)],
            )
            img_path = f"{save_folder}/mei_{session_id}_{neuron_id}.pdf"
            fig_axes[0].savefig(img_path, bbox_inches="tight", facecolor="w", dpi=300)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
