import argparse
import os
from functools import partial
from pathlib import PurePath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.insilico.stimulus_optimization.objective import (
    IncreaseObjective,
    InnerNeuronVisualizationObjective,
    SliceMeanReducer,
)
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.legacy.hoefling_configs import STIMULUS_RANGE_CONSTRAINTS
from openretina.models.core_readout import load_core_readout_model
from openretina.utils.nnfabrik_model_loading import Center, load_ensemble_model_from_remote
from openretina.utils.plotting import plot_stimulus_composition, save_stimulus_to_mp4_video


def add_parser_arguments(parser: argparse.ArgumentParser):
    parser.description = (
        "Visualize most exciting stimuli for each neuron in the model and visualize the weights of the model."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", type=str)
    group.add_argument("--is-hoefling-ensemble-model", action="store_true")

    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Path were to save outputs",
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=-1,
        help="If >= 0 load the ensemble model with that model_id, else use torch.load to load the model",
    )
    parser.add_argument(
        "--time-steps-stimulus",
        type=int,
        default=50,
        help="The time steps used in the stimulus to optimize",
    )
    parser.add_argument(
        "--image-file-format", type=str, default="jpg", help="File format to save the visualization plots in"
    )


def load_model(
    path: str | None,
    device: str = "cuda",
    model_id: int = 0,
    do_center_readout: bool = False,
    is_hoefling_ensemble_model: bool = False,
):
    if path is not None:
        # check if the filename contains gru to decide whether to load from the gru name
        is_gru_model = "gru" in PurePath(path).name.lower()
        model = load_core_readout_model(path, device, is_gru_model)
        print(f"Initializing lightning model from {path} to {device=} ({is_gru_model=}")
    elif is_hoefling_ensemble_model:
        center_readout = Center(target_mean=(0.0, 0.0)) if do_center_readout else None
        _, ensemble_model = load_ensemble_model_from_remote(device=device, center_readout=center_readout)
        print("Initialized ensemble model from remote.")
        model = ensemble_model.members[model_id]
    else:
        raise ValueError(
            f"Either path must be set or is_hoefling_ensemble_model must be True "
            f"({path=}, {is_hoefling_ensemble_model=})"
        )

    return model


def _get_min_max_values_and_norm(num_channels: int) -> tuple[list[tuple], float | None]:
    if num_channels == 2:
        min_max_values = [
            (STIMULUS_RANGE_CONSTRAINTS["x_min_green"], STIMULUS_RANGE_CONSTRAINTS["x_max_green"]),
            (STIMULUS_RANGE_CONSTRAINTS["x_min_uv"], STIMULUS_RANGE_CONSTRAINTS["x_max_uv"]),
        ]
        norm = float(STIMULUS_RANGE_CONSTRAINTS["norm"])
        return min_max_values, norm
    else:
        return [(None, None)], None


def visualize_model_neurons(
    model_path: str,
    save_folder: str,
    device: str,
    model_id: int,
    is_hoefling_ensemble_model: bool,
    image_file_format: str,
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
    min_max_values, norm = _get_min_max_values_and_norm(stimulus_shape[1])
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
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus.data = stimulus_postprocessor.process(stimulus.data * 0.1)

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
            fig_axes_tuple = plt.subplots(2, 2, figsize=(7 * 3, 12))
            axes: np.ndarray[Any, plt.Axes] = fig_axes_tuple[1]  # type: ignore

            highlight_stim_start = stimulus_shape[2] - num_timesteps + response_reducer.start
            highlight_stim_end = highlight_stim_start + response_reducer.length - 1
            plot_stimulus_composition(
                stimulus=stimulus[0],
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(highlight_stim_start, highlight_stim_end)],
            )
            output_folder = f"{save_folder}/{layer_name}"
            os.makedirs(output_folder, exist_ok=True)
            fig_path = f"{output_folder}/{channel_id}.{image_file_format}"
            fig_axes_tuple[0].savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
            print(f"Saved figure at {fig_path=}")
            fig_axes_tuple[0].clf()
            plt.close()
            save_stimulus_to_mp4_video(stimulus[0], f"{output_folder}/{channel_id}.mp4")

    response_reducer = SliceMeanReducer(axis=0, start=10, length=10)
    print(f"Reset response reducer for optimizing output neurons: {response_reducer}")
    for session_key in model_readout_keys:
        output_folder = f"{save_folder}/output_neurons/{session_key}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Optimizing output neurons for {session_key} in folder {output_folder}")

        for neuron_id in range(model.readout[session_key].outdims):
            objective = IncreaseObjective(
                model, neuron_indices=neuron_id, data_key=session_key, response_reducer=response_reducer
            )
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus.data = stimulus_postprocessor.process(stimulus.data * 0.1)

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
            fig_axes_tuple = plt.subplots(2, 2, figsize=(7 * 3, 12))
            axes: np.ndarray[Any, plt.Axes] = fig_axes_tuple[1]  # type: ignore

            plot_stimulus_composition(
                stimulus=stimulus[0],
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(40, 49)],
            )
            fig_path = f"{output_folder}/{neuron_id}.{image_file_format}"
            fig_axes_tuple[0].savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig_axes_tuple[0].clf()
            save_stimulus_to_mp4_video(stimulus[0], f"{output_folder}/{neuron_id}.mp4")
            plt.close()

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
            model.readout[session_key].save_weight_visualizations(folder_path, image_file_format)
            print(f"Plotted visualizations for {session_key}")

        # Visualize weights
        for i, layer in enumerate(model.core.features):
            output_dir = f"{save_folder}/weights_layer_{i}"
            os.makedirs(output_dir, exist_ok=True)
            layer.conv.save_weight_visualizations(output_dir, image_file_format)
            print(f"Saved weight visualization at path {output_dir}")
