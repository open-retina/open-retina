import datetime
import os
import tempfile
from functools import partial
from typing import Any, Literal

import cv2
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange, repeat
from IPython.display import HTML, Video, display
from jaxtyping import Float, Int
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from moviepy import ImageSequenceClip
from scipy.ndimage import center_of_mass
from tqdm.auto import tqdm

from openretina.data_io.hoefling_2024.constants import FRAME_RATE_MODEL
from openretina.legacy.hoefling_configs import MEAN_STD_DICT_74x64, pre_normalisation_values_18x16
from openretina.utils.constants import BADEN_TYPE_BOUNDARIES
from openretina.utils.video_analysis import calculate_fft, decompose_kernel, weighted_main_frequency

# Longer animations
matplotlib.rcParams["animation.embed_limit"] = 2**128

# Editable text in PDFs
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


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
    color_channels = stimulus.shape[0]

    assert filepath.endswith(".mp4")
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video = cv2.VideoWriter(filepath, fourcc, fps, (stimulus.shape[3], stimulus.shape[2]))

    # Normalize to uint8
    if apply_undo_video_normalization:
        assert color_channels == 2, "Normalization is only supported for 2 color channels, but {color_channels=}"
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
        if color_channels == 1:
            for c_id in range(3):
                frame[:, :, c_id] = stimulus_uint8[0, i, :, :]
        elif color_channels == 2:
            frame[:, :, 1] = stimulus_uint8[0, i, :, :]  # Green channel
            frame[:, :, 2] = stimulus_uint8[1, i, :, :]  # Red channel
        else:
            frame[:, :, 0] = stimulus_uint8[0, i, :, :]  # Red
            frame[:, :, 1] = stimulus_uint8[1, i, :, :]  # Green
            frame[:, :, 2] = stimulus_uint8[2, i, :, :]  # Blue
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
    assert len(stimulus.shape) == 4
    num_color_channels, time_steps, dim_y, dim_x = stimulus.shape

    # guess color channels
    color_array_map = {
        1: ["black"],
        2: ["darkgreen", "darkviolet"],
        3: ["red", "green", "blue"],
    }
    color_channel_names_map = {
        1: ("Grey",),
        2: ("Green", "UV"),
        3: ("Red", "Green", "Blue"),
    }
    color_array = color_array_map[num_color_channels]
    color_channel_names_array = color_channel_names_map[num_color_channels]

    stimulus_time = np.linspace(0, time_steps / FRAME_RATE_MODEL, time_steps)
    weighted_main_freqs = [0.0] * num_color_channels
    temporal_traces_max = 0.0

    temporal_kernels = []
    spatial_kernels_with_padding = []
    for color_idx in range(num_color_channels):
        temporal, spatial, _ = decompose_kernel(stimulus[color_idx])
        temporal_kernels.append(temporal)
        spatial_kernels_with_padding.append(spatial)

        if color_idx < (num_color_channels - 1):
            padding = np.ones((spatial.shape[0], 8))
            spatial_kernels_with_padding.append(padding)

    # Spatial structure
    spatial_ax.set_title(f"Spatial Component ({'/'.join(color_channel_names_array)})")
    # Create spatial kernel with interleave

    spat = np.concatenate(spatial_kernels_with_padding, axis=1)
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
        frequencies_str = "/".join([f"{x:.1f}" for x in weighted_main_freqs])
        freq_ax.set_title(f"Weighted Frequency: {frequencies_str} Hz ({'/'.join(color_channel_names_array)})")

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


def plot_vector_field_resp_iso(
    x: np.ndarray,
    y: np.ndarray,
    gradient_dict: np.ndarray,
    resp_dict: np.ndarray,
    normalize_response: bool = False,
    rc_dict: dict[str, Any] = {},
    cmap: str = "hsv",
) -> plt.Figure:
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
        Z = Z / Z.max() * 100
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
        cont_lines = plt.contour(X, Y, Z, levels=levels, cmap="jet_r", zorder=300)
        plt.gca().clabel(
            cont_lines,
            inline=True,
            fmt="%1.0f",
            levels=list(cont_lines.levels)[::2],
            colors="k",
            fontsize=5,
            zorder=400,
        )
        ax = plt.gca()
        ax.set_aspect("equal")

        for i, contrast_green in enumerate(x[1:-1]):
            for j, contrast_uv in enumerate(y[1:-1]):
                unit_vec = gradient_grid[:, i, j] / np.linalg.norm(gradient_grid[:, i, j]) * 0.1
                ax.arrow(
                    contrast_green,
                    contrast_uv,
                    unit_vec[0],
                    unit_vec[1],
                    fill=True,
                    linewidth=0.5,
                    head_width=0.03,
                    color="grey",
                    zorder=300,
                )
        ax.set_xlabel("Green contrast")
        ax.set_ylabel("UV contrast")
        ax.xaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
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


def numpy_to_mp4_video(
    video_array: Float[np.ndarray, "time height width 3"],
    save_path: str | os.PathLike | None = None,
    fps: int = 30,
    display_video=True,
    display_width: int = 640,
    display_height: int = 360,
) -> None:
    """
    Convert a NumPy array to mp4 video and optionally display it in a Jupyter Notebook.

    Parameters:
        video_array (np.ndarray): An array of video frames of shape (time, height, width, channels),
                                    where channels must be exactly 3.
        fps (int): Frames per second for the video.
        save_path (str): Path to save the video to. If None, the video is written to a temporary file and
                        just displayed in the notebook.
        display_video (bool): Whether to display the video in the notebook.
        display_width (int): Width of the video iframe in the notebook.
        dispaly_height (int): Height of the video iframe in the notebook.
    """
    assert video_array.ndim == 4, "video_array must have 4 dimensions"
    assert display_video or save_path is not None, "Either display_video or save_path must be provided"

    video_array = prepare_video_for_display(video_array)

    if save_path is not None:
        assert str(save_path).endswith(".mp4"), "save_path must end with '.mp4'"
        file_path = save_path
    else:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            file_path = temp_file.name

    try:
        # Convert NumPy array to a moviepy ImageSequenceClip
        clip = ImageSequenceClip(list(video_array), fps=fps)

        # Write the video to the file
        clip.write_videofile(file_path, codec="libx264", audio=False, logger=None)

        # Display the video with specified iframe dimensions
        if display_video:
            display(Video(file_path, embed=True, width=display_width, height=display_height))
    finally:
        # Ensure the file is deleted after use if it was temporary
        if os.path.exists(file_path) and save_path is None:
            os.remove(file_path)


def stitch_videos(
    video1: Float[np.ndarray, "time height width channels"],
    video2: Float[np.ndarray, "time height width channels"],
    separator_width: int = 50,
    resize_method: Literal["linear", "nearest", "cubic"] = "linear",
) -> np.ndarray:
    """
    Stitch two video arrays side by side with a white band in between.

    Parameters:
        video1 (np.ndarray): Video array of shape (time, height, width, channels).
        video2 (np.ndarray): Video array of shape (time, height, width, channels).
        separator_width (int): Width of the white band separating the videos.
        resize_method (str): Interpolation method for resizing ("linear", "nearest", etc.).

    Returns:
        np.ndarray: Stitched video array of shape (time, max_height, width1 + width2 + band_width, channels).
    """
    assert (
        video1.shape[0] == video2.shape[0]
    ), f"Videos must have the same number of frames (time). Got {video1.shape[0]} and {video2.shape[0]}."
    assert (
        video1.shape[3] == video2.shape[3] == 3
    ), f"Videos must have 3 color channels (RGB). Got {video1.shape[3]} and {video2.shape[3]}."
    time, _, _, channels = video1.shape

    max_height = max(video1.shape[1], video2.shape[1])
    bigger_index = 0 if video1.shape == video2.shape else 1 if video1.shape[1] > video2.shape[1] else 2

    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }.get(resize_method, cv2.INTER_LINEAR)

    def resize_video(video, target_height):
        """Resize video frames to the target height while maintaining aspect ratio."""
        scale = target_height / video.shape[1]
        target_width = int(video.shape[2] * scale)
        return np.stack(
            [cv2.resize(frame, (target_width, target_height), interpolation=interpolation) for frame in video], axis=0
        )

    # Resize the videos if necessary
    video1_resized = resize_video(video1, max_height) if bigger_index == 2 else video1
    video2_resized = resize_video(video2, max_height) if bigger_index == 1 else video2

    # Create a white band of the desired width
    band = 255 * np.ones((time, max_height, separator_width, channels), dtype=np.uint8)

    # Concatenate the videos with the band in between
    stitched_video = np.concatenate((video1_resized, band, video2_resized), axis=2)
    return stitched_video


def create_roi_animation(
    roi_mask: Int[np.ndarray, "roi_height roi_width"],
    activity: Float[np.ndarray, "n_neurons time"],
    roi_ids: Int[np.ndarray, " n_neurons"] | None = None,
    cell_types: Int[np.ndarray, " n_neurons"] | None = None,
    min_cell_type: int | None = 1,
    max_cell_type: int | None = 46,
    type_boundaries: list[int] | None = None,
    max_activity: int = 10,
    visualize_ids: bool = False,
    figsize: tuple[int, int] = (8, 8),
) -> Float[np.ndarray, "time fig_height fig_width 3"]:
    """
    Create an animation of neuronal activity and return a numpy tensor of video frames.

    ---
    Parameters:
    roi_mask : np.ndarray
        2D array where background is 1 and neurons are -n (n being neuron index)
    activity : np.ndarray
        n x time array of neuronal activity
    roi_ids : np.ndarray
        1D array indicating the neuron ID for each row in activity (default None)
    cell_types : np.ndarray
        1D array indicating the cell type for each neuron (default None)
    type_boundaries : list[int]
        List of integers indicating boundaries between broad cell types
    max_activity : float
        Maximum activity value for scaling (default 10)
    visualize_ids : bool
        If True, displays neuron IDs on their ROIs
    figsize : tuple
        Figure size in inches (width, height), default (8, 8)

    ---
    Returns:
    np.ndarray
        Numpy array with shape (time, height, width, channels) containing video frames.
    """

    # Get number of neurons and normalize activity
    n_neurons = np.min(roi_mask) * -1
    activity_normalized = np.clip(activity / max_activity, 0, 1)

    # Pre-compute color mapping if cell types are provided
    color_map = {}
    if cell_types is not None:
        max_type = np.max(cell_types) if max_cell_type is None else max_cell_type
        min_type = np.min(cell_types) if min_cell_type is None else min_cell_type

        # Create broad type ranges
        if type_boundaries is None:
            type_boundaries = []
        type_boundaries = sorted([min_type] + type_boundaries + [max_type])

        # For each broad type range, assign a base saturation
        saturations = {}
        for i in range(len(type_boundaries) - 1):
            start, end = type_boundaries[i], type_boundaries[i + 1]
            types_in_range = np.where((cell_types >= start) & (cell_types <= end))[0]
            base_saturation = 0.6 + (i * (0.3 / len(type_boundaries)))  # Vary saturation slightly for each broad type
            for type_idx in types_in_range:
                saturations[cell_types[type_idx]] = base_saturation

        # Assign hues based on cell type numbers
        for cell_type in np.unique(cell_types):
            # Map cell type to hue (0-1)
            hue = (cell_type - min_type) / (max_type - min_type)
            saturation = saturations[cell_type]
            color_map[cell_type] = (hue, saturation)

    # Pre-compute centroids for each ROI if we're visualizing IDs
    centroids = {}
    if visualize_ids:
        for i in range(n_neurons):
            if roi_ids is not None and (i + 1) not in roi_ids:
                continue
            neuron_mask = roi_mask == -(i + 1)
            if np.any(neuron_mask):
                centroids[i] = center_of_mass(neuron_mask)

    def render_frame(frame_idx):
        """Render a single frame using matplotlib"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        plt.close()

        # Create frame image (now in RGB)
        frame_image = np.zeros((*roi_mask.shape, 3))

        # Update each neuron's color based on type and brightness based on activity
        if roi_ids is not None:
            for i, roi_id in enumerate(roi_ids):
                neuron_mask = roi_mask == -roi_id
                activity_val = activity_normalized[i, frame_idx]

                if cell_types is not None:
                    # Get pre-computed hue and saturation for this cell type
                    hue, saturation = color_map[cell_types[i]]
                    # Convert HSV to RGB
                    rgb = mcolors.hsv_to_rgb([hue, saturation, activity_val])
                    frame_image[neuron_mask] = rgb
                else:
                    # If no cell types provided, use grayscale
                    frame_image[neuron_mask] = [activity_val] * 3
        else:
            for i in range(n_neurons):
                neuron_mask = roi_mask == -(i + 1)
                activity_val = activity_normalized[i, frame_idx]

                if cell_types is not None:
                    hue, saturation = color_map[cell_types[i]]
                    rgb = mcolors.hsv_to_rgb([hue, saturation, activity_val])
                    frame_image[neuron_mask] = rgb
                else:
                    frame_image[neuron_mask] = [activity_val] * 3

        # Display frame
        ax.imshow(frame_image)
        ax.axis("off")

        # Add neuron ID labels if requested
        if visualize_ids:
            for i in range(n_neurons):
                if i in centroids:
                    y, x = centroids[i]
                    ax.text(
                        x,
                        y,
                        str(i + 1),
                        color="white",
                        fontsize=8,
                        ha="center",
                        va="center",
                        path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
                    )

        # Redraw canvas to reflect updates
        fig.canvas.draw()

        # Convert matplotlib figure to RGB numpy array
        buffer = fig.canvas.buffer_rgba()
        data = np.asarray(buffer)
        # Convert RGBA to RGB
        data = data[:, :, :3]
        return data

    # Generate all frames
    n_frames = activity.shape[1]
    video_frames = []
    for frame_idx in tqdm(range(n_frames), desc="Generating frames for ROI animation"):
        frame = render_frame(frame_idx)
        video_frames.append(frame)

    # Stack frames into a single numpy array
    video_tensor = np.stack(video_frames, axis=0)
    return video_tensor


def prepare_video_for_display(
    video_array: Float[np.ndarray, "time height width channels"] | Float[np.ndarray, "channels time height width"],
) -> Float[np.ndarray, "time height width 3"]:
    """
    Prepare a video array for display by converting it to uint8 and clipping values to [0, 255].

    Parameters:
    -----------
    video_array : np.ndarray
        Video array of shape (time, height, width, channels) or (channels, time, height, width).

    Returns:
    --------
    np.ndarray
        Video array of shape (time, height, width, channels) with values clipped to [0, 255] and converted to uint8,
        ready for display.
    """
    if np.argmin(video_array.shape) == 0:
        video_array = rearrange(video_array, "channels time height width -> time height width channels")

    if video_array.shape[-1] == 1:
        # Black and white video
        video_array = repeat(video_array, "time height width channel -> time height width (n channel)", n=3)
    if video_array.shape[-1] == 2:
        # Green and UV video
        video_array = np.concatenate((video_array[..., 1][..., None], video_array), axis=-1)

    if video_array.dtype == np.float32 or video_array.dtype == np.float64:
        video_min = video_array.min()
        video_max = video_array.max()
        video_array_normalized = (video_array - video_min) / (video_max - video_min) * 255
        video_array = video_array_normalized.astype(np.uint8)
    return video_array


def display_video(
    video_array: Float[np.ndarray, "time height width channels"] | None,
    video_save_path: str | os.PathLike | None = None,
    fps: int = 30,
    display_width: int = 640,
    display_height: int = 360,
):
    """
    Display a video array or a saved video file.
    This function can either display a video from a numpy array or from a saved video file.
    If a path to an existing video is provided, it will be displayed directly.
    Otherwise, the function will create and display a video from the provided numpy array.

    Args:
        video_array (np.ndarray | None): A 4D numpy array with dimensions (time, height, width, channels)
            representing the video to display. Can be None if video_save_path points to an existing video.
        video_save_path (str | os.PathLike | None): Path to a saved video file. If the file exists,
            it will be displayed directly. If None, video_array must be provided.
        fps (int, optional): Frames per second for video playback. Defaults to 30.
        display_width (int, optional): Width of the displayed video in pixels. Defaults to 640.
        display_height (int, optional): Height of the displayed video in pixels. Defaults to 360.

    Raises:
        AssertionError: If neither a valid video_save_path nor a video_array is provided.

    Examples:
        >>> # Display from numpy array
        >>> video = np.random.rand(100, 480, 640, 3)
        >>> display_video(video, None)
        >>> # Display from saved file
        >>> display_video(None, "path/to/video.mp4")
    """
    if video_save_path is not None and os.path.exists(video_save_path):
        # If save_path already exist, simply display that.
        display(Video(video_save_path, embed=True, width=display_width, height=display_height))
    else:
        assert (
            video_array is not None
        ), "Video array to display must be provided without an already existing saved rendering."
        numpy_to_mp4_video(
            video_array=video_array,
            save_path=video_save_path,
            fps=fps,
            display_video=True,
            display_height=display_height,
            display_width=display_width,
        )


def extract_baden_group(value):
    """
    Given a cell type value, return the Baden group it belongs to,
    according to the classification in the Baden et al. 2016 paper.
    """
    if value <= BADEN_TYPE_BOUNDARIES[0]:
        return "OFF"
    elif value > BADEN_TYPE_BOUNDARIES[0] and value <= BADEN_TYPE_BOUNDARIES[1]:
        return "ON-OFF"
    elif value > BADEN_TYPE_BOUNDARIES[1] and value <= BADEN_TYPE_BOUNDARIES[2]:
        return "fast ON"
    elif value > BADEN_TYPE_BOUNDARIES[2] and value <= BADEN_TYPE_BOUNDARIES[3]:
        return "slow ON"
    elif value > BADEN_TYPE_BOUNDARIES[3] and value <= BADEN_TYPE_BOUNDARIES[4]:
        return "uncertain RGC"
    else:
        return "AC"
