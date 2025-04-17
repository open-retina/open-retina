import argparse
import os
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.data_io.hoefling_2024.constants import STIMULUS_RANGE_CONSTRAINTS
from openretina.insilico.stimulus_optimization.objective import (
    IncreaseObjective,
    SliceMeanReducer,
)
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    StimulusPostprocessor,
    TemporalGaussianLowPassFilterProcessor,
)
from openretina.models.core_readout import load_core_readout_from_remote
from openretina.utils.nnfabrik_model_loading import Center
from openretina.utils.plotting import plot_stimulus_composition, save_stimulus_to_mp4_video


def add_parser_arguments(parser: argparse.ArgumentParser):
    parser.description = "Create group Meis for all cell types in the model"

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", type=str)
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
        "--time-steps-stimulus",
        type=int,
        default=50,
        help="The time steps used in the stimulus to optimize",
    )
    parser.add_argument("seed", type=int, default=42, help="Seed for generating the initial stimulus")
    parser.add_argument(
        "--use-smoothing",
        action="store_true",
        help="Whether to smooth the stimulus during optimization",
    )
    parser.add_argument(
        "--mdi",
        action="store_true",
        help="Whether to generate most depressing input stimuli instead of most exciting ones",
    )
    parser.add_argument(
        "--center-scale-and-bias",
        action="store_true",
        help="Whether to set scale to 1 and bias to 0 for every neuron in the model",
    )
    parser.add_argument(
        "--image-file-format", type=str, default="jpg", help="File format to save the visualization plots in"
    )


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


def visualize_group_meis(
    model_path: str,
    save_folder: str,
    device: str,
    image_file_format: str,
    time_steps_stimulus: int,
    use_smoothing: bool,
    mdi: bool,
    center_scale_and_bias: bool,
    seed: int,
) -> None:
    model = load_core_readout_from_remote(model_path, device=device)

    # center all readout to 0.0 to create shared group MEIs
    center_readout = Center(target_mean=(0.0, 0.0), mean_key="mask_mean")
    center_readout(model)

    if center_scale_and_bias:
        center_scale = Center(1.0, mean_key="scale_param")
        center_scale(model)
        center_bias = Center(0.0, mean_key="bias_param")
        center_bias(model)

    model.eval()
    group_assignments = model.get_group_assignments()
    group_to_neuron_ids = {
        group: [i for i, g in enumerate(group_assignments) if g == group] for group in set(group_assignments)
    }

    stimulus_shape = model.stimulus_shape(time_steps=time_steps_stimulus, num_batches=1)
    min_max_values, norm = _get_min_max_values_and_norm(stimulus_shape[1])
    stimulus_clipper = ChangeNormJointlyClipRangeSeparately(
        min_max_values=min_max_values,
        norm=norm,
    )

    os.makedirs(save_folder, exist_ok=True)
    for group, neuron_indices in group_to_neuron_ids.items():
        os.makedirs(save_folder, exist_ok=True)
        print(f"Optimizing group {'mdi' if mdi else 'mei'} for {group=}")
        stimulus_postprocessor_list: list[StimulusPostprocessor] = [stimulus_clipper]
        if use_smoothing:
            stimulus_lowpass_filter = TemporalGaussianLowPassFilterProcessor(sigma=0.5, kernel_size=5, device=device)
            stimulus_postprocessor_list.append(stimulus_lowpass_filter)

        # Create stimulus and clip it to the expected range, make sure to always use the same initial stimulus
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
        stimulus.data = stimulus_clipper.process(stimulus.data * 0.1)

        objective = IncreaseObjective(
            model,
            neuron_indices=neuron_indices,
            data_key=None,
            response_reducer=SliceMeanReducer(axis=0, start=10, length=10),
            invert=mdi,
        )
        optimize_stimulus(
            stimulus,
            optimizer_init_fn=partial(torch.optim.SGD, lr=100.0),
            objective_object=objective,
            optimization_stopper=OptimizationStopper(max_iterations=10),
            stimulus_postprocessor=stimulus_postprocessor_list,
        )

        fig_axes_tuple = plt.subplots(2, 2, figsize=(7 * 3, 12))
        axes: np.ndarray[Any, plt.Axes] = fig_axes_tuple[1]  # type: ignore

        plot_stimulus_composition(
            stimulus=stimulus[0],
            temporal_trace_ax=axes[0, 0],
            spatial_ax=axes[0, 1],
            freq_ax=axes[1, 1],
            highlight_x_list=[(40, 49)],
        )
        axes[0, 0].axvline(1.0, ls="--", color="dimgray")  # type:ignore

        # plot traces
        neuron_traces = model.forward(stimulus)[0, :, neuron_indices].detach().cpu().numpy()
        neuron_traces_avg = np.average(neuron_traces, axis=-1)
        ax: plt.Axes = axes[1, 0]  # type: ignore
        ax.set_title(f"Neuron Responses (n={len(neuron_indices)})")
        time_axis = np.arange(30, 30 + neuron_traces.shape[0]) / 30.0
        for trace in np.swapaxes(neuron_traces, 0, 1):
            ax.plot(time_axis, trace, alpha=0.4)
        ax.plot(time_axis, neuron_traces_avg, linewidth=4, color="black")

        ax.fill_betweenx(ax.get_ylim(), time_axis[-10], time_axis[-1], color="k", alpha=0.1)
        ax.axvline(1.0, ls="--", color="dimgray")

        input_name = "mdi" if mdi else "mei"
        path_prefix = f"{save_folder}/group_{input_name}_{group:02}"
        fig_axes_tuple[0].savefig(f"{path_prefix}.{image_file_format}", bbox_inches="tight", facecolor="w", dpi=300)
        fig_axes_tuple[0].clf()
        save_stimulus_to_mp4_video(stimulus[0], f"{path_prefix}.mp4")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    visualize_group_meis(**vars(args))
