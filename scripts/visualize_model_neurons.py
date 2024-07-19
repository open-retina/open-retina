#!/usr/bin/env python3

import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from openretina.hoefling_2024.constants import STIMULUS_RANGE_CONSTRAINTS
from openretina.optimization.objective import (InnerNeuronVisualizationObjective, SliceMeanReducer)
from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.optimization_stopper import OptimizationStopper
from openretina.optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.plotting import plot_stimulus_composition
from openretina.hoefling_2024.nnfabrik_model_loading import load_ensemble_retina_model_from_directory, Center


BASE_PATH = "/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/"
# BASE_PATH = "/home/tzenkel/GitRepos/rgc-natstim-model/data/"
ENSEMBLE_MODEL_PATH = BASE_PATH + "models/nonlinear/9d574ab9fcb85e8251639080c8d402b7"


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model_id", type=int, default=1)

    return parser.parse_args()


def load_model(path: str = ENSEMBLE_MODEL_PATH, device: str = "cuda", model_id: int = 0):
    center_readout = Center(target_mean=(0.0, 0.0))
    data_info, ensemble_model = load_ensemble_retina_model_from_directory(
        path, device, center_readout=center_readout)
    print(f"Initialized ensemble model from {path}")
    return ensemble_model.members[model_id]


def main(
        save_folder: str,
        device: str,
        model_id: int,
        neuron_identifier: str = "core_features_layer0_conv:1",
        data_key: str = "2_ventral2_20201016"
) -> None:
    model = load_model(device=device, model_id=model_id)
    model.to(device).eval()

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
    optimizer_init_fn = partial(torch.optim.SGD, lr=100.0)

    layer_str, channel_str = neuron_identifier.split(':')
    layer_str = layer_str.strip()
    n_channel = int(channel_str.strip())
    inner_neuron_objective = InnerNeuronVisualizationObjective(model, data_key, layer_str, n_channel, response_reducer)
    layer_names_array = [x for x in inner_neuron_objective.features_dict.keys()
                         if "readout" not in x and "regularizer" not in x and x != "core_features"]
    print(layer_names_array)
    for layer_name in layer_names_array:
        for channel_id in range(16):
            print(f"Optimizing {layer_name=} {channel_id=}")
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus.data = stimulus_postprocessor.process(stimulus.data)
            inner_neuron_objective.set_layer_channel(layer_name, channel_id)
            if "layer0" in layer_name:
                response_reducer._start = 20
            elif "layer1" in layer_name or layer_name == "core":
                response_reducer._start = 10
            else:
                raise ValueError(layer_name)

            optimize_stimulus(
                stimulus,
                optimizer_init_fn,
                inner_neuron_objective,
                OptimizationStopper(max_iterations=10),
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
            output_folder = f"{save_folder}/{layer_name}"
            os.makedirs(output_folder, exist_ok=True)
            fig_path = f"{output_folder}/{channel_id}.jpg"
            fig.savefig(fig_path, bbox_inches="tight", facecolor="w", dpi=300)
            print(f"Saved figure at {fig_path=}")

    for session_key in model.readout.keys():
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
