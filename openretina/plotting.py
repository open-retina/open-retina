from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from jaxtyping import Float
from matplotlib import animation

from .hoefling_2022_configs import pre_normalisation_values


def undo_video_normalization(
    video: Float[torch.Tensor, "channels time height width"], values_dict: dict = pre_normalisation_values
) -> Float[torch.Tensor, "channels time height width"]:
    """
    Undo the normalization of the video.
    """
    video[0] = video[0] * values_dict["channel_0_std"] + values_dict["channel_0_mean"]
    video[1] = video[1] * values_dict["channel_1_std"] + values_dict["channel_1_mean"]

    if isinstance(video, torch.Tensor):
        return video.type(torch.int32)
    else:
        return video.astype(np.int32)


def update_video(video, ax, frame):
    """
    Updates the video frame in the given axis.

    To be used in animations.
    """
    ax.clear()

    # Extract the two channels
    green_channel = video[0, frame].numpy()
    purple_channel = video[1, frame].numpy()

    # Normalize the channels
    green_normalized = green_channel / green_channel.max()
    purple_normalized = purple_channel / purple_channel.max()

    # Create an empty RGB image
    rgb_image = np.zeros((*green_channel.shape, 3))

    # Assign green channel to the green color in RGB
    rgb_image[:, :, 1] = green_normalized

    # Assign purple channel to the blue and red color in RGB (to create purple)
    rgb_image[:, :, 0] = purple_normalized  # Red
    rgb_image[:, :, 2] = purple_normalized  # Blue

    # Display the composite image
    ax.imshow(rgb_image)
    ax.axis("off")


def play_stimulus(video: Float[torch.Tensor, "channels time height width"]) -> HTML:
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

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

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
