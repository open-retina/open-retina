import datetime
import os
from functools import partial
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from jaxtyping import Float
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from .hoefling_2024.configs import pre_normalisation_values
from .hoefling_2024.constants import FRAME_RATE_MODEL
from .video_analysis import calculate_fft, decompose_kernel, weighted_main_frequency

# Longer animations
matplotlib.rcParams["animation.embed_limit"] = 2**128


def undo_video_normalization(
    video: Float[torch.Tensor, "channels time height width"], values_dict: dict = pre_normalisation_values
) -> Float[torch.Tensor, "channels time height width"]:
    """
    Undo the normalization of the video.
    """
    video = video.clone()
    video[0] = video[0] * values_dict["channel_0_std"] + values_dict["channel_0_mean"]
    video[1] = video[1] * values_dict["channel_1_std"] + values_dict["channel_1_mean"]

    return video.type(torch.int)


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
    neuron_idx: Optional[int] = 0,
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
    stimulus: np.array,
    temporal_trace_ax,
    freq_ax: Optional[Any],
    spatial_ax,
    lowpass_cutoff: float = 10.0,
    highlight_x_list: Optional[List[Tuple[int, int]]] = None,
):
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
    padding = np.zeros((spat_green.shape[0], 8))
    spat = np.concatenate([spat_green, padding, spat_uv], axis=1)

    abs_max = np.max([abs(spat.max()), abs(spat.min())])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    spatial_ax.imshow(spat, cmap="RdBu_r", norm=norm)
    scale_bar = Rectangle(xy=(6, 15), width=3, height=1, color="k", transform=spatial_ax.transData)
    spatial_ax.annotate(
        text="150 µm",
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


def save_figure(filename: str, folder: str, fig=None):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{date}_{filename}"
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(folder, filename))
    plt.close(fig)
