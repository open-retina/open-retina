import datetime
import os
from functools import partial
from typing import Any, Dict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from jaxtyping import Float
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib as mpl 
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, FixedLocator
import seaborn as sns

from openretina.data_io.hoefling_2024.constants import FRAME_RATE_MODEL
from openretina.legacy.hoefling_configs import MEAN_STD_DICT_74x64, pre_normalisation_values_18x16
from openretina.utils.video_analysis import calculate_fft, decompose_kernel, weighted_main_frequency

# Longer animations
matplotlib.rcParams["animation.embed_limit"] = 2**128


def undo_video_normalization(
    video: Float[torch.Tensor, "channels time height width"], values_dict: dict = pre_normalisation_values_18x16
) -> Float[torch.Tensor, "channels time height width"]:
    """
    Undo the normalization of the video.
    """
    video = video.clone()
    video[0] = video[0] * values_dict["channel_0_std"] + values_dict["channel_0_mean"]
    video[1] = video[1] * values_dict["channel_1_std"] + values_dict["channel_1_mean"]

    return video.type(torch.int)


def save_stimulus_to_mp4_video(
    stimulus: np.ndarray,
    filepath: str,
    fps: int = 5,
    start_at_frame: int = 0,
    apply_undo_video_normalization: bool = False,
) -> None:
    assert len(stimulus.shape) == 4
    assert stimulus.shape[0] == 2  # color channels

    assert filepath.endswith(".mp4")
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video = cv2.VideoWriter(filepath, fourcc, fps, (stimulus.shape[3], stimulus.shape[2]))

    # Normalize to uint8
    if apply_undo_video_normalization:
        stimulus[0] = stimulus[0] * MEAN_STD_DICT_74x64["channel_0_mean"] + MEAN_STD_DICT_74x64["channel_0_std"]
        stimulus[1] = stimulus[1] * MEAN_STD_DICT_74x64["channel_1_mean"] + MEAN_STD_DICT_74x64["channel_1_std"]
        # Clip to the range of uint8, otherwise there'll be an overflow (-1 will get converted to 255)
        stimulus_uint8 = stimulus.clip(0.0, 255.0).astype(np.uint8)
    else:
        stimulus_norm = stimulus - stimulus.min()
        stimulus_norm = 255 * (stimulus_norm / stimulus_norm.max())
        stimulus_uint8 = stimulus_norm.astype(np.uint8)

    for i in range(start_at_frame, stimulus_uint8.shape[1]):
        # Create an empty 3D array and assign the data from the 4D array
        frame = np.zeros((stimulus_uint8.shape[2], stimulus_uint8.shape[3], 3), dtype=np.uint8)
        frame[:, :, 1] = stimulus_uint8[0, i, :, :]  # Green channel
        frame[:, :, 2] = stimulus_uint8[1, i, :, :]  # Red channel
        video.write(frame)

    video.release()


def update_video(video, ax, frame):
    """
    Updates the video frame in the given axis.

    To be used in animations.
    """
    ax.clear()

    if video.shape[0] == 1:
        current_frame = video[0, frame].numpy() / video[0, frame].max()

    elif video.shape[0] == 2:
        # Extract the two channels
        green_channel = video[0, frame].numpy()
        purple_channel = video[1, frame].numpy()

        # Create an empty RGB image
        current_frame = np.zeros((*green_channel.shape, 3))

        # Assign green channel to the green color in RGB
        current_frame[:, :, 1] = green_channel

        # Assign purple channel to the blue and red color in RGB (to create purple)
        current_frame[:, :, 0] = purple_channel  # Red
        current_frame[:, :, 2] = purple_channel  # Blue

    else:
        raise NotImplementedError("Only 1 or 2 channels are supported")

    # Display the composite image
    ax.imshow(current_frame, cmap="gray" if video.shape[0] == 1 else None)
    ax.axis("off")


def play_stimulus(video: Float[torch.Tensor, "channels time height width"], normalise: bool = True) -> HTML:
    if normalise:
        min_val = torch.min(video)
        max_val = torch.max(video)
        video = (video - min_val) / (max_val - min_val)

    fig, ax = plt.subplots()

    update = partial(update_video, video, ax)

    ani = animation.FuncAnimation(fig, update, frames=len(video[0]), interval=50)
    return HTML(ani.to_jshtml())


def play_sample_batch(
    video: Float[torch.Tensor, "channels time height width"],
    responses: Float[torch.Tensor, "time n_neurons"],
    neuron_idx: int | None = 0,
):
    assert video.shape[1] == responses.shape[0], "Movie length and response length must match"

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 2]})

    def update_trace(frame):
        ax[1].clear()
        ax[1].plot(responses[:frame, neuron_idx])
        ax[1].set_ylim(0, responses[:, neuron_idx].max() * 1.1)
        ax[1].set_xlim(0, len(responses))
        ax[1].set_title(f"Neuron {neuron_idx}")
        ax[1].set_xlabel("Time (frames from start)")
        ax[1].set_ylabel("Response")

    def update_all(frame):
        update_video(video, ax[0], frame)
        update_trace(frame)
        return ax

    ani = animation.FuncAnimation(fig, update_all, frames=len(video[0]), interval=50)
    return HTML(ani.to_jshtml())


def plot_stimulus_composition(
    stimulus: np.ndarray,
    temporal_trace_ax,
    freq_ax: Any | None,
    spatial_ax,
    lowpass_cutoff: float = 10.0,
    highlight_x_list: list[tuple[int, int]] | None = None,
) -> None:
    color_array = ["darkgreen", "darkviolet"]
    color_channel_names_array = ("Green", "UV")

    assert len(stimulus.shape) == 4
    num_color_channels, dim_t, dim_y, dim_x = stimulus.shape

    time_steps = stimulus.shape[1]
    stimulus_time = np.linspace(0, time_steps / FRAME_RATE_MODEL, time_steps)
    weighted_main_freqs = [0.0, 0.0]
    temporal_traces_max = 0.0
    temp_green, spat_green, _ = decompose_kernel(stimulus[0])
    temp_uv, spat_uv, _ = decompose_kernel(stimulus[1])
    temporal_kernels = [temp_green, temp_uv]

    # Spatial structure
    spatial_ax.set_title(f"Spatial Component {color_channel_names_array}")
    padding = np.ones((spat_green.shape[0], 8))
    spat = np.concatenate([spat_green, padding, spat_uv], axis=1)

    abs_max = np.max([abs(spat.max()), abs(spat.min())])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    spatial_ax.imshow(spat, cmap="RdBu_r", norm=norm)
    # In the low res model the stimulus shape was 18x16 (50 um pixels), for the high-res it is 72x64 (12.5um pixels)
    scale_bar_with = 4 if stimulus.shape[-1] > 20 else 1
    scale_bar = Rectangle(xy=(6, 15), width=scale_bar_with, height=1, color="k", transform=spatial_ax.transData)
    spatial_ax.annotate(
        text="50 µm",
        xy=(6, 14),
    )
    spatial_ax.add_patch(scale_bar)
    spatial_ax.axis("off")

    for color_idx in range(num_color_channels):
        temp = temporal_kernels[color_idx]
        temporal_traces_max = max(temporal_traces_max, np.abs(temp).max())

        # Temporal structure
        temporal_trace_ax.plot(stimulus_time, temp, color=color_array[color_idx])

        if freq_ax is not None:
            fft_freqs, fft_weights = calculate_fft(temp, FRAME_RATE_MODEL, lowpass_cutoff)
            weighted_main_freqs[color_idx] = weighted_main_frequency(fft_freqs, fft_weights)
            freq_ax.plot(fft_freqs, fft_weights, color=color_array[color_idx])

    temporal_trace_ax.set_ylim(-temporal_traces_max, +temporal_traces_max + 1)
    temporal_trace_ax.set_title("Temporal Trace of the Stimulus")
    temporal_trace_ax.set_xlabel("Time [s]")

    if freq_ax is not None:
        freq_ax.set_xlim(0.0, lowpass_cutoff + 1)
        freq_ax.set_xlabel("Frequency [Hz]")
        freq_ax.set_title(
            f"Weighted Frequency: {weighted_main_freqs[0]:.1f}/{weighted_main_freqs[1]:.1f} Hz"
            f" ({color_channel_names_array[0]}/{color_channel_names_array[1]})"
        )

    if highlight_x_list is not None:
        for x_0_idx, x_1_idx in highlight_x_list:
            x_0 = stimulus_time[x_0_idx]
            x_1 = stimulus_time[x_1_idx]
            temporal_trace_ax.fill_betweenx(temporal_trace_ax.get_ylim(), x_0, x_1, color="k", alpha=0.1)


def polar_plot_of_direction_of_motion_responses(
    direction_in_degree: list[int],
    peak_response_per_directions: list[float],
) -> None:
    # Convert directions to radians
    directions_with_peak_response = sorted(
        [(d, v) for d, v in zip(direction_in_degree, peak_response_per_directions, strict=True)]
    )

    # Add the first direction and data point to the end to close the plot
    directions_with_peak_response.append(directions_with_peak_response[0])

    # Create the polar plot
    plt.figure(figsize=(6, 6))
    sorted_directions = [x[0] for x in directions_with_peak_response]
    sorted_data = [x[1] for x in directions_with_peak_response]
    plt.polar(np.deg2rad(sorted_directions), sorted_data, marker="o")

    # Set the direction of the zero point to the top
    plt.gca().set_theta_zero_location("N")  # type: ignore

    # Set the direction of rotation to clockwise
    plt.gca().set_theta_direction(-1)  # type: ignore

    # Set the labels for the directions
    plt.gca().set_xticklabels([f"{x}°" for x in sorted_directions])


class ColorMapper:
    def __init__(self, cmap_name: str, vmin: float = 0, vmax: float = 1):
        self.cmap_name = cmap_name
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(self.cmap_name)
        self.norm = plt.Normalize(vmin=self.vmin, vmax=self.vmax)

    def get_color(self, number):
        color = self.cmap(self.norm(number))
        return color


def plot_vector_field_resp_iso(x: np.ndarray, y: np.ndarray, gradient_dict: np.ndarray, 
                               resp_dict: np.ndarray, normalize_response: bool = False,
                               rc_dict: dict[str, Any] = {},
                               cmap: str = "hsv") -> None:
    """
    Plots a vector field response with isoresponse lines.

    Parameters:
    x (array-like): The x-coordinates of the grid points.
    y (array-like): The y-coordinates of the grid points.
    gradient_dict (ndarray): A dictionary containing response gradients at grid points.
    resp_dict (ndarray): A dictionary containing responses at grid points.
    normalize_response (bool, optional): If True, normalize the response data. Default is False.
    colour_norm (bool, optional): If True, apply color normalization. Default is False.
    rc_dict (dict, optional): A dictionary of rc settings to use in the plot. Default is an empty dictionary.
    cmap (str, optional): The colormap to use for the plot. Default is "hsv".

    Returns:
    None
    """

    Z = resp_dict.transpose()
    if normalize_response:
        Z=Z/Z.max() * 100
    gradient_grid = gradient_dict[:, 1:-1, 1:-1]
    X, Y = np.meshgrid(x, x)

    # Define levels for isoresponse lines
    levels = np.linspace(Z.min(), Z.max(), 25)
    # cm = ColorMapper("cool", vmin=gradient_norm_grid.min(),
    #                  vmax=gradient_norm_grid.max())

    with mpl.rc_context(rc_dict):
        fig = plt.figure()

        # Create a contour plot with isoresponse lines

        plt.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=200)  # Change cmap to the desired colormap
        cont_lines = plt.contour(X,Y,Z, levels=levels, cmap='jet_r',zorder=300)
        plt.gca().clabel(cont_lines, inline=True, fmt='%1.0f',
                         levels = cont_lines.levels[::2], colors="k", fontsize=5, zorder=400)
        ax = plt.gca()
        ax.set_aspect("equal")
        
        for i, contrast_green in enumerate(x[1:-1]):
            for j, contrast_uv in enumerate(y[1:-1]):
                unit_vec = gradient_grid[:, i, j]/np.linalg.norm(gradient_grid[:, i, j]) * .1
                ax.arrow(contrast_green, contrast_uv, unit_vec[0], unit_vec[1],
                            fill=True, linewidth=.5,
                            head_width=.03, color="grey", zorder=300)
        ax.set_xlabel("Green contrast")
        ax.set_ylabel("UV contrast")
        ax.xaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        sns.despine()
    return fig


def save_figure(filename: str, folder: str, fig=None) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{date}_{filename}"
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(folder, filename))
    plt.close(fig)


def legend_without_duplicate_labels(ax=None) -> None:
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique = [(handle, label) for i, (handle, label) in enumerate(zip(handles, labels)) if label not in labels[:i]]
    ax.legend(*zip(*unique))
