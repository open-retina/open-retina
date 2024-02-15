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

        # Normalize the channels
        green_normalized = green_channel / green_channel.max()
        purple_normalized = purple_channel / purple_channel.max()

        # Create an empty RGB image
        current_frame = np.zeros((*green_channel.shape, 3))

        # Assign green channel to the green color in RGB
        current_frame[:, :, 1] = green_normalized

        # Assign purple channel to the blue and red color in RGB (to create purple)
        current_frame[:, :, 0] = purple_normalized  # Red
        current_frame[:, :, 2] = purple_normalized  # Blue

    else:
        raise NotImplementedError("Only 1 or 2 channels are supported")

    # Display the composite image
    ax.imshow(current_frame, cmap="gray" if video.shape[0] == 1 else None)
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
