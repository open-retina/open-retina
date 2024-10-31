#!/usr/bin/env python3

import argparse
import os
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from openretina.hoefling_2024.configs import STIMULUS_RANGE_CONSTRAINTS
from openretina.optimization.objective import InnerNeuronVisualizationObjective, SliceMeanReducer, SingleNeuronObjective
from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.optimization_stopper import OptimizationStopper
from openretina.optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.plotting import plot_stimulus_composition, save_stimulus_to_mp4_video
from openretina.hoefling_2024.nnfabrik_model_loading import load_ensemble_retina_model_from_directory, Center
from openretina.models.core_readout import CoreReadout


DEFAULT_BASE_PATH = "/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/"
DEFAULT_ENSEMBLE_MODEL_PATH = os.path.join(DEFAULT_BASE_PATH, "models/nonlinear/9d574ab9fcb85e8251639080c8d402b7")


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model_path", default=DEFAULT_ENSEMBLE_MODEL_PATH)
    parser.add_argument(
        "--model_id",
        type=int,
        default=-1,
        help="If >= 0 load the ensemble model with that model_id, " "else use torch.load to load the model",
    )
    parser.add_argument("--core_readout_lightning", action="store_true")
    parser.add_argument(
        "--stimulus_shape",
        nargs="+",
        type=int,
        default=[2, 50, 72, 64],
        help="Stimulus shape: [color_channels, time_dim, height, width",
    )

    return parser.parse_args()


def load_model(
    path: str,
    device: str = "cuda",
    model_id: int = 0,
    do_center_readout: bool = False,
    core_readout_lightning: bool = True,
):
    if core_readout_lightning:
        model = CoreReadout.load_from_checkpoint(path).to(device)  # type: ignore
        print(f"Initialized lightning model from {path} to {device=}")
    elif model_id < 0:
        model = torch.load(path, map_location=torch.device(device))
        print(f"Initialized model from {path}")
    else:
        _, ensemble_model = load_ensemble_retina_model_from_directory(path, device)
        print(f"Initialized ensemble model from {path}")
        model = ensemble_model.members[model_id]

    if do_center_readout and not core_readout_lightning:
        center_readout = Center(target_mean=(0.0, 0.0))
        center_readout(model)
    return model


def main(
    model_path: str,
    save_folder: str,
    device: str,
    model_id: int,
    core_readout_lightning: bool,
    stimulus_shape: tuple[int, ...],
) -> None:
    if len(stimulus_shape) != 4:
        raise ValueError(f"Invalid stimulus shape, needs to contain 4 integers, but was {stimulus_shape=}")
    stimulus_shape = (1,) + tuple(stimulus_shape)

    model = load_model(
        model_path,
        device=device,
        model_id=model_id,
        do_center_readout=True,
        core_readout_lightning=core_readout_lightning,
    )
    model.eval()

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
    optimizer_init_fn = partial(torch.optim.SGD, lr=100.0)

    try:
        model_readout_keys = model.readout_keys()
    except AttributeError:
        model_readout_keys = model.readout.readout_keys()
    data_key = model_readout_keys[0]
    inner_neuron_objective = InnerNeuronVisualizationObjective(model, data_key, response_reducer)
    # only select output of each layer (ignore submodules like ..._layer0_norm or ..._layer0_pool)
    layer_names_array = [x for x in inner_neuron_objective.features_dict.keys() if "layer" in x and x[-1].isdigit()]
    print(f"Generating MEIs for the following layers: {layer_names_array}")
    for layer_name in layer_names_array:
        output_shape = inner_neuron_objective.get_output_shape_for_layer(layer_name, stimulus_shape)
        if output_shape is None:
            print(f"Could not determine output shape for {layer_name=}, skipping layer.")
            continue
        num_channels, num_timesteps = output_shape[1:3]
        # We maximize the last frames of the time dimension of the output of the layer
        response_reducer.start = num_timesteps - response_reducer.length
        print(f"Reset response reduce for layer {layer_name} to: {response_reducer}")
        for channel_id in range(num_channels):
            print(f"Optimizing {layer_name=} {channel_id=}")
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus.data = stimulus_postprocessor.process(stimulus.data)
            inner_neuron_objective.set_layer_channel(layer_name, channel_id)

            try:
                optimize_stimulus(
                    stimulus,
                    optimizer_init_fn,
                    inner_neuron_objective,
                    OptimizationStopper(max_iterations=10),
                    stimulus_regularization_loss=stimulus_regularizing_loss,
                    stimulus_postprocessor=stimulus_postprocessor,
                )
            except Exception as e:
                print(f"Skipping {layer_name=} {channel_id=} because of Exception: {e}")
                continue
            stimulus_np = stimulus[0].detach().cpu().numpy()
            fig_axes_tuple = plt.subplots(2, 2, figsize=(7 * 3, 12))
            axes: np.ndarray[Any, plt.Axes] = fig_axes_tuple[1]  # type: ignore

            plot_stimulus_composition(
                stimulus=stimulus_np,
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(40, 49)],
            )
            output_folder = f"{save_folder}/{layer_name}"
            os.makedirs(output_folder, exist_ok=True)
            fig_path = f"{output_folder}/{channel_id}.jpg"
            fig_axes_tuple[0].savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
            print(f"Saved figure at {fig_path=}")
            fig_axes_tuple[0].clf()
            plt.close()
            save_stimulus_to_mp4_video(stimulus_np, f"{output_folder}/{channel_id}.mp4")
            del stimulus_np

    response_reducer = SliceMeanReducer(axis=0, start=10, length=10)
    print(f"Reset response reducer for optimizing output neurons: {response_reducer}")
    for session_key in model_readout_keys:
        output_folder = f"{save_folder}/output_neurons/{session_key}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Optimizing output neurons for {session_key} in folder {output_folder}")

        for neuron_id in range(model.readout[session_key].outdims):
            objective = SingleNeuronObjective(
                model, neuron_idx=neuron_id, data_key=session_key, response_reducer=response_reducer
            )
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus.data = stimulus_postprocessor.process(stimulus.data)

            try:
                optimize_stimulus(
                    stimulus,
                    optimizer_init_fn,
                    objective,
                    OptimizationStopper(max_iterations=10),
                    stimulus_regularization_loss=stimulus_regularizing_loss,
                    stimulus_postprocessor=stimulus_postprocessor,
                )
            except Exception as e:
                print(f"Skipping neuron {neuron_id} in session {session_key} because of exception {e}")
                continue
            stimulus_np = stimulus[0].cpu().numpy()
            fig_axes_tuple = plt.subplots(2, 2, figsize=(7 * 3, 12))
            axes: np.ndarray[Any, plt.Axes] = fig_axes_tuple[1]  # type: ignore

            plot_stimulus_composition(
                stimulus=stimulus_np,
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(40, 49)],
            )
            fig_path = f"{output_folder}/{neuron_id}.jpg"
            fig_axes_tuple[0].savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig_axes_tuple[0].clf()
            save_stimulus_to_mp4_video(stimulus_np, f"{output_folder}/{neuron_id}.mp4")
            plt.close()
            del stimulus_np

    # Reload model without centered readouts
    if core_readout_lightning:
        model.save_weight_visualizations(save_folder)
    else:
        model = load_model(
            model_path,
            device=device,
            model_id=model_id,
            do_center_readout=False,
            core_readout_lightning=core_readout_lightning,
        )
        model.to(device).eval()
        for session_key in model_readout_keys:
            folder_path = f"{save_folder}/weights_readout/{session_key}"
            os.makedirs(folder_path, exist_ok=True)
            model.readout[session_key].save_weight_visualizations(folder_path)
            print(f"Plotted visualizations for {session_key}")

        # Visualize weights
        for i, layer in enumerate(model.core.features):
            output_dir = f"{save_folder}/weights_layer_{i}"
            os.makedirs(output_dir, exist_ok=True)
            layer.conv.save_weight_visualizations(output_dir)
            print(f"Saved weight visualization at path {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
