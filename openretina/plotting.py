import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from jaxtyping import Float
from matplotlib import animation


def play_stimulus(video: Float[torch.Tensor, "channels time height width"]) -> HTML:
    fig, ax = plt.subplots()

    def update(frame):
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

    ani = animation.FuncAnimation(fig, update, frames=len(video[0]), interval=50)
    return HTML(ani.to_jshtml())
