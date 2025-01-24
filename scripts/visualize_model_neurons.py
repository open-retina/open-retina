#!/usr/bin/env python3

import argparse
import os
from functools import partial
from pathlib import PurePath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.insilico.stimulus_optimization.objective import (
    InnerNeuronVisualizationObjective,
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
from openretina.models.core_readout import CoreReadout, GRUCoreReadout
from openretina.utils.nnfabrik_model_loading import Center, load_ensemble_retina_model_from_directory
from openretina.utils.plotting import plot_stimulus_composition, save_stimulus_to_mp4_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize most exciting stimuli for each neuron in the model "
        "and visualize the weights of the model"
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--save_folder",
        type=str,
        required=True,
        help="Path were to save outputs",
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=-1,
        help="If >= 0 load the ensemble model with that model_id, else use torch.load to load the model",
    )
    parser.add_argument("--is_hoefling_ensemble_model", action="store_true")
    parser.add_argument(
        "--time_steps_stimulus",
        type=int,
        default=50,
        help="The time steps used in the stimulus to optimize",
    )

    return parser.parse_args()


def load_model(
    path: str,
    device: str = "cuda",
    model_id: int = 0,
    do_center_readout: bool = False,
    is_hoefling_ensemble_model: bool = False,
):
    map_location = torch.device(device)
    if not is_hoefling_ensemble_model:
        # check if the filename contains gru to decide whether to load from the gru name
        if "gru" in PurePath(path).name.lower():
            print(f"Initializing lightning GRU model from {path} to {device=}")
            model: CoreReadout = GRUCoreReadout.load_from_checkpoint(path, map_location=map_location)  # type: ignore
        else:
            print(f"Initializing lightning base model from {path} to {device=}")
            model = CoreReadout.load_from_checkpoint(path, map_location=map_location)  # type: ignore
    elif model_id < 0:
        model = torch.load(path, map_location=map_location)
        print(f"Initialized model using torch.load() from {path}")
    else:
        _, ensemble_model = load_ensemble_retina_model_from_directory(path, device)
        print(f"Initialized ensemble model from {path}")
        model = ensemble_model.members[model_id]

    if do_center_readout and is_hoefling_ensemble_model:
        center_readout = Center(target_mean=(0.0, 0.0))
        center_readout(model)
    return model


def get_min_max_values_and_norm(num_channels: int) -> tuple[list[tuple], float | None]:
    if num_channels == 2:
        min_max_values = [
            (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
            (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
        ]
        norm = float(STIMULUS_RANGE_CONSTRAINTS["norm"])
        return min_max_values, norm
    else:
        return [(None, None)], None


def main(
    model_path: str,
    save_folder: str,
    device: str,
    model_id: int,
    is_hoefling_ensemble_model: bool,
    time_steps_stimulus: int,
) -> None:
    model = load_model(
        model_path,
        device=device,
        model_id=model_id,
        do_center_readout=True,
        is_hoefling_ensemble_model=is_hoefling_ensemble_model,
    )
    model.eval()

    stimulus_shape = model.stimulus_shape(time_steps=time_steps_stimulus, num_batches=1)
    response_reducer = SliceMeanReducer(axis=0, start=10, length=10)
    min_max_values, norm = get_min_max_values_and_norm(stimulus_shape[1])
    stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately(
        min_max_values=min_max_values,
        norm=norm,
    )
    stimulus_regularizing_loss = RangeRegularizationLoss(
        min_max_values=min_max_values,
        max_norm=norm,
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
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)  # type: ignore
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

            highlight_stim_start = stimulus_shape[2] - num_timesteps + response_reducer.start
            highlight_stim_end = highlight_stim_start + response_reducer.length - 1
            plot_stimulus_composition(
                stimulus=stimulus_np,
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(highlight_stim_start, highlight_stim_end)],
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
    if not is_hoefling_ensemble_model:
        model.save_weight_visualizations(save_folder)
    else:
        model = load_model(
            model_path,
            device=device,
            model_id=model_id,
            do_center_readout=False,
            is_hoefling_ensemble_model=is_hoefling_ensemble_model,
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
