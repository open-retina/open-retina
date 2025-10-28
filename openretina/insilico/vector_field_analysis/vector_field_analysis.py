import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from git import Optional
from PIL import Image
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

from openretina.models.core_readout import BaseCoreReadout

"""
LSTA (Local Spike-Triggered Average) visualization toolkit.

Example usage:
    >>> # Load your model (ensure it has the required attributes)
    >>> model = load_your_model_function()
    >>> # Prepare dataset
    >>> movies, n_empty = prepare_movies_dataset(
    ...     model, session_id="example", image_library=your_image_array
    ... )
    >>>
    >>> # Compute LSTA
    >>> lsta_lib, resp_lib = compute_lsta_library(
    ...     model, movies, session_id="example", cell_id=0
    ... )
    >>>
    >>> # Visualize
    >>> PC1, PC2, ev = get_pc_from_pca(model, 0, lsta_lib)
    >>> fig = plot_clean_vectorfield(lsta_lib, 0, PC1, PC2, images, coords, ev)
    >>> plt.show()
"""

LOGGER = logging.getLogger(__name__)


def load_and_preprocess_images(image_dir: str, target_h: int, target_w: int, n_channels: int) -> np.ndarray:
    """
    Loads PNG images from a directory, downsamples, center-crops, and repeats channels as needed.
    Parameters
    ----------
        image_dir (str): Directory containing PNG images.
        target_h (int): Target height for cropping.
        target_w (int): Target width for cropping.
        n_channels (int): Number of channels to repeat.
    Returns
    -------
        np.ndarray: Array of shape (num_images, n_channels, target_h, target_w).
    Raises
    -------
        ValueError: If no PNG images are found in the directory.
    Notes
    -------
        - Images are downsampled to fit within target dimensions while maintaining aspect ratio.
        - Center-cropping is applied after downsampling.
        - Single-channel images are repeated across channels if n_channels > 1.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
    images = np.array([np.array(Image.open(os.path.join(image_dir, f))) for f in image_files])
    # Downsample and crop using array operations
    downsample_factors = np.minimum(images.shape[1] / target_h, images.shape[2] / target_w).astype(int)
    # Ensure downsample_factors is at least 1
    downsample_factors[downsample_factors < 1] = 1

    # Downsample
    ds_images = np.array([img[::factor, ::factor] for img, factor in zip(images, downsample_factors)])

    # Center crop
    h, w = ds_images.shape[1:3]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    cropped_images = ds_images[:, start_h : start_h + target_h, start_w : start_w + target_w]

    compressed_images = cropped_images.astype(np.float32)
    # Add channel dimension
    compressed_images = compressed_images[:, np.newaxis]
    # Repeat channels if needed
    if n_channels > 1:
        compressed_images = np.repeat(compressed_images, n_channels, axis=1)
    return compressed_images


def get_model_temporal_padding(
    model: torch.nn.Module, n_channels: int, target_h: int, target_w: int, device: str
) -> int:
    """
    Determines the number of empty frames needed for temporal padding by the model.
    Parameters
    ----------
        model: Neural model object.
        n_channels (int): Number of input channels.
        target_h (int): Input height.
        target_w (int): Input width.
        device (str): Device for torch tensor.
    Returns
    -------
        n_empty_frames (int): Number of initial empty frames for temporal padding.
    Raises
    -------
        ValueError: If the model output has more temporal frames than the input test movie.
    """
    test_movie_temporal_length: int = 100
    empty_movie: torch.Tensor = torch.zeros(
        1, n_channels, test_movie_temporal_length, target_h, target_w, dtype=torch.float32, device=device
    )
    n_empty_frames: int = test_movie_temporal_length - model(empty_movie).shape[1]
    return n_empty_frames


def normalize_movies_array(movies: np.ndarray, model: BaseCoreReadout, session_id: str, n_channels: int) -> np.ndarray:
    """
    Normalizes movies using model parameters.
    Parameters:
    -----------
        movies (np.ndarray): Movies array to normalize.
        model: Neural model object.
        session_id (str): Session identifier.
        n_channels (int): Number of channels.
    Returns
    -------
        np.ndarray: Normalized movies array.
    Raises
    -------
        KeyError: If session_id not found in model's normalization dictionary.
    Notes
    -------
        - For single-channel models, uses session-specific normalization.
        - For multi-channel models, uses default normalization.
    """
    if session_id in model.data_info["movie_norm_dict"]:
        movie_norm_dict_key = session_id
    else:
        movie_norm_dict_key = "default"

    movie_norm_dict: dict[str, float] = model.data_info["movie_norm_dict"][movie_norm_dict_key]
    stim_mean = movie_norm_dict["norm_mean"]
    stim_std = movie_norm_dict["norm_std"]
    for channel in range(n_channels):
        movies[:, channel, :, :, :] = (movies[:, channel, :, :, :] - stim_mean) / stim_std
    return movies


def prepare_movies_dataset(
    model: BaseCoreReadout,
    session_id: str,
    n_image_frames: int = 16,
    normalize_movies: bool = True,
    image_library: Optional[np.ndarray] = None,
    image_dir: Optional[str] = None,
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    """
    Prepares a dataset of movie stimuli for input into a neural model.
    This function delegates image loading, preprocessing, normalization, and temporal padding to helper functions.
    Parameters
    ----------
        model: Neural model object with `data_info` attribute.
        session_id (str): Identifier for the session.
        n_image_frames (int, optional): Number of frames per movie for each image.
        normalize_movies (bool, optional): Whether to normalize the movies.
        image_library (np.ndarray, optional): Preprocessed image library.
        image_dir (str, optional): Directory containing image files (.png).
        device (str, optional): Device for torch tensors.
    Returns
    -------
        movies (np.ndarray): Array of shape (num_images, n_channels, n_frames, target_h, target_w).
        n_empty_frames (int): Number of initial empty frames for temporal padding.
    Raises
    -------
        ValueError: If both `image_library` and `image_dir` are provided.
    """
    n_channels = model.data_info["input_shape"][0]
    target_h, target_w = model.data_info["input_shape"][1:3]

    if image_library is not None and image_dir is not None:
        raise ValueError("Provide either image_library or image_dir, not both.")
    if image_dir is not None:
        LOGGER.info(f"Loading images from {image_dir}...")
        compressed_images = load_and_preprocess_images(image_dir, target_h, target_w, n_channels)
    elif image_library is not None:
        LOGGER.info("Using provided image library...")
        compressed_images = image_library
    else:
        raise ValueError("Provide either image_library or image_dir.")

    # number of grey frames = size of equivalent temporal filter of the full model + 10 for border effects
    n_empty_frames = get_model_temporal_padding(model, n_channels, target_h, target_w, device) + 10
    movies = np.repeat(compressed_images[:, :, np.newaxis, :, :], n_empty_frames + n_image_frames, axis=2)

    if normalize_movies:
        movies = normalize_movies_array(movies, model, session_id, n_channels)

    # Set initial empty frames to mean grey
    movies[:, :, :n_empty_frames, :, :] = movies.mean()
    return movies, n_empty_frames


def compute_lsta_library(
    model: torch.nn.Module,
    movies: np.ndarray,
    session_id: str,
    cell_id: int,
    batch_size: int = 64,
    integration_window: tuple[int, int] = (5, 15),
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Local Spike-Triggered Average (LSTA) library and response library for a given model,
      set of movies, and cell_id.

    For each batch of input movies, this function:
        - Runs the model to obtain outputs for all cells and time points.
        - Selects the output for a specific cell over a specified integration window (time range).
        - Sums the selected outputs and computes the gradient of this sum with respect to the input movies.
        - The resulting gradients (LSTA maps) are averaged over the integration window for each movie.
        - Collects both the LSTA maps and the raw model outputs for all movies.

    Parameters
    ----------
        model (torch.nn.Module): The neural network model to evaluate.
        movies (np.ndarray or torch.Tensor): Array of input movie stimuli with shape (num_samples, channels, frames,
          height, width).
        session_id (str): Identifier for the session/data key used by the model.
        cell_id (int): Index of the cell for which to compute LSTA.
        batch_size (int, optional): Number of samples per batch. Default is 64.
        integration_window (tuple, optional): Tuple (start, end) specifying the time window (frame indices) over which
          to sum outputs. Default is (5, 10).
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.

    Returns
    -------
        lsta_library (np.ndarray): Array of LSTA maps averaged over the integration window, shape
         (num_samples, channels, height, width).
        response_library (np.ndarray): Array of model outputs for all batches, shape (num_samples, frames, num_cells).

    Raises
    ------
        IndexError: If cell_id is out of bounds for the model output.

    Notes
    -----
        - The LSTA map for each movie is computed as the gradient of the summed output for the specified cell and time
          window,
            with respect to the input movie frames.
        - The returned lsta_library is averaged over the integration window
          (i.e., mean gradient across selected frames).
        - The response_library contains the raw model outputs for all movies, all frames, and all cells.
        - Default integration_window is not always optimal;
          adjust based on model architecture and expected response timing.
    """
    model.eval()
    all_lstas = []
    all_outputs = []

    for i in range(0, len(movies), batch_size):
        batch_movies = torch.tensor(movies[i : i + batch_size], dtype=torch.float32, device=device, requires_grad=True)

        outputs = model(batch_movies, data_key=session_id)
        num_cells = outputs.shape[-1]
        if not (0 <= cell_id < num_cells):
            raise IndexError(f"cell_id {cell_id} is out of bounds (number of cells: {num_cells})")

        chosen_cell_outputs = outputs[:, integration_window[0] : integration_window[1], cell_id].sum()
        chosen_cell_outputs.backward()

        assert batch_movies.grad is not None
        batch_lstas = batch_movies.grad.detach()
        all_lstas.append(batch_lstas)
        all_outputs.append(outputs.detach())

        # Clear gradients
        del batch_movies
        torch.cuda.empty_cache()

    lstas = torch.cat(all_lstas, dim=0)
    lsta_library = lstas.mean(dim=2)  # Average over the integration window (frames)
    response_library = torch.cat(all_outputs, dim=0)
    return lsta_library.cpu().numpy(), response_library.cpu().numpy()


def get_pc_from_pca(
    model, channel: int, lsta_library: np.ndarray, plot: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the first two principal components (PC1 and PC2) from a PCA analysis on a selected channel of the
      input data.
    Parameters
    ----------
    model : object
        Model object containing data information, specifically the input shape in `model.data_info["input_shape"]`.
    channel : int
        Index of the channel to select from `lsta_library` for PCA analysis.
    lsta_library : np.ndarray
        Input data array of shape (samples, channels, height, width).
    plot : bool, optional
        If True, plots the first two principal components as images using matplotlib.
    Returns
    -------
    PC1 : np.ndarray
        The first principal component as a flattened array.
    PC2 : np.ndarray
        The second principal component as a flattened array.
    explained_variance : np.ndarray
        Array containing the explained variance ratio for the first two principal components.
    Notes
    -----
    - The function reshapes the selected channel data to (samples, height * width) before applying PCA.
    - If `plot` is True, displays the principal components as images with color mapping.
    """

    # Select channel and reshape
    lsta_reshaped = lsta_library[:, channel, :, :].reshape(lsta_library.shape[0], -1)

    pca = PCA(n_components=2)
    pca.fit(lsta_reshaped)

    explained_variance = pca.explained_variance_ratio_
    PC1, PC2 = pca.components_

    if plot:
        PC_max = max(np.abs(PC1).max(), np.abs(PC2).max())
        plt.figure(figsize=(10, 5))
        for component in range(2):
            plt.subplot(1, 2, component + 1)
            plt.imshow(
                pca.components_[component].reshape(model.data_info["input_shape"][1:3]),
                cmap="bwr",
                vmin=-PC_max,
                vmax=PC_max,
            )
            plt.title(f"PCA {component} ({explained_variance[component]:.2f} e.v.)")
            plt.axis("off")

    return PC1, PC2, explained_variance


def get_images_coordinate(images: np.ndarray, PC1: np.ndarray, PC2: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Projects a set of images onto two principal component vectors and optionally plots their coordinates.
    Parameters
    ----------
        images (np.ndarray): Array of images with shape (n_samples, height, width).
        PC1 (np.ndarray): First principal component vector with shape (height * width,).
        PC2 (np.ndarray): Second principal component vector with shape (height * width,).
        plot (bool, optional): If True, plots the projected coordinates. Default is False.
    Returns
    -------
        np.ndarray: Array of shape (n_samples, 2) containing the coordinates of each image projected onto PC1 and PC2.
    Note
    -----
        The function reshapes each image to a 1D vector before projection.
    """
    flatten_images = images.reshape(images.shape[0], -1)
    # Vectorized dot product: (N, features) @ (2, features).T -> (N, 2)
    PC_stack = np.stack([PC1, PC2], axis=0)  # Shape: (2, features)
    images_coordinate = flatten_images @ PC_stack.T  # Shape: (N, 2)

    if plot:
        pt_x = images_coordinate[:, 0]
        pt_y = images_coordinate[:, 1]
        plt.figure()
        plt.scatter(pt_x, pt_y)

    return images_coordinate


def plot_pc_insets(
    fig: plt.Figure,
    PC1: np.ndarray,
    PC2: np.ndarray,
    x_size: int,
    y_size: int,
    explained_variance: np.ndarray | None = None,
) -> None:
    """
    Helper function to plot PC1 and PC2 as inset images on a matplotlib figure.
    Parameters
    ----------
        fig (matplotlib.figure.Figure): The figure to add insets to.
        PC1 (np.ndarray): First principal component, flattened.
        PC2 (np.ndarray): Second principal component, flattened.
        x_size (int): Height of the reshaped PC images.
        y_size (int): Width of the reshaped PC images.
        explained_variance (list or np.ndarray, optional): Explained variance ratios for PC1 and PC2.
    Returns
    -------
        None
    Notes
    -----
        - The insets are positioned at the top-right and bottom-left corners of the figure.
        - Color mapping is set to 'bwr' with symmetric limits based on the maximum absolute values of PC1 and PC2.
    """
    PC_max = max(np.abs(PC1).max(), np.abs(PC2).max())

    ax_img1 = fig.add_axes((0.825, 0.425, 0.15, 0.15), anchor="C", zorder=1)
    ax_img1.imshow(PC1.reshape(x_size, y_size), cmap="bwr", vmin=-PC_max, vmax=PC_max)
    ax_img1.axis("off")
    title1 = "PC1"
    if explained_variance is not None:
        title1 += f" ({explained_variance[0]:.1%})"
    ax_img1.set_title(title1, size=20)

    ax_img2 = fig.add_axes((0.425, 0.825, 0.15, 0.15), anchor="C", zorder=1)
    ax_img2.imshow(PC2.reshape(x_size, y_size), cmap="bwr", vmin=-PC_max, vmax=PC_max)
    ax_img2.axis("off")
    title2 = "PC2"
    if explained_variance is not None:
        title2 += f" ({explained_variance[1]:.1%})"
    ax_img2.set_title(title2, size=20)


def plot_untreated_vectorfield(
    lsta_library: np.ndarray, channel: int, PC1: np.ndarray, PC2: np.ndarray, images_coordinate: np.ndarray
) -> plt.Figure:
    """
    Plots a vector field visualization using principal components from an LSTA library.
    This function extracts the specified channel from the LSTA library, projects each LSTA onto two principal components
      (PC1 and PC2),
    and visualizes the resulting vector field at given image coordinates using matplotlib's quiver plot. Additionally,
      it displays
    the PC1 and PC2 components as inset images.
    This function is primarily for visualization in notebooks.
    Returns figure for saving or further customization.
    Parameters
    ----------
    lsta_library : np.ndarray
        A 4D numpy array containing the LSTA library data with shape (n_samples, n_channels, x_size, y_size).
    channel : int
        The index of the channel to extract from the LSTA library for analysis.
    PC1 : np.ndarray
        The first principal component vector used for projection.
    PC2 : np.ndarray
        The second principal component vector used for projection.
    images_coordinate : np.ndarray
        A 2D numpy array of shape (n_samples, 2) containing the (x, y) coordinates for each LSTA sample.
    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the vector field plot with PC1 and PC2 inset images. Call plt.show()
         to display,
        or fig.savefig() to save.
    Notes
    -----
    - The function uses matplotlib's quiver plot to visualize the vector field.
    - PC1 and PC2 are displayed as inset images for reference.
    - The axes are turned off for a cleaner visualization.
    """
    lsta_library = lsta_library[:, channel, :, :]
    arrowheads = np.array([[np.dot(PC1, lsta.flatten()), np.dot(PC2, lsta.flatten())] for lsta in lsta_library])
    fig, ax = plt.subplots(figsize=(20, 15))
    window_size = int(max(images_coordinate[:, 0].max(), images_coordinate[:, 1].max()) * 1.1)
    ax.quiver(
        images_coordinate[: len(lsta_library), 0],
        images_coordinate[: len(lsta_library), 1],
        arrowheads[:, 0],
        arrowheads[:, 1],
        width=0.002,
        scale_units="xy",
        angles="xy",
        scale=arrowheads.max(),
        alpha=0.5,
    )
    ax.set_xlim((-window_size, window_size))
    ax.set_ylim((-window_size, window_size))
    ax.axis("off")

    x_size = lsta_library.shape[-2]
    y_size = lsta_library.shape[-1]
    plot_pc_insets(fig, PC1, PC2, x_size, y_size)
    return plt.gcf()


def plot_clean_vectorfield(
    lsta_library: np.ndarray,
    channel: int,
    PC1: np.ndarray,
    PC2: np.ndarray,
    images: list[Any] | np.ndarray,
    images_coordinate: np.ndarray,
    explained_variance: np.ndarray,
    x_bins: int = 31,
    y_bins: int = 31,
    responses: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plots a cleaned vector field representation of binned image and LSTA data projected onto principal components.
    This function bins images and their corresponding LSTA (Local Spike-Triggered Average) responses based on spatial
    coordinates,
    projects the binned data onto two principal components (PC1 and PC2),
    and visualizes the resulting vector field using quiver plots.
    Insets showing the PC1 and PC2 components are also added to the figure.
        This function is primarily for visualization in notebooks.
    Returns figure for saving or further customization.
    Insets showing the PC1 and PC2 components are also added to the figure for reference.
    Parameters
    ----------
    lsta_library : np.ndarray
        Array of LSTA responses with shape (n_samples, n_channels, x_size, y_size).
    channel : int
        Index of the channel to select from lsta_library.
    PC1 : np.ndarray
        First principal component vector for projection (flattened).
    PC2 : np.ndarray
        Second principal component vector for projection (flattened).
    images : np.ndarray
        Array of images corresponding to LSTA responses, shape (n_samples, x_size, y_size).
    images_coordinate : np.ndarray
        Array of spatial coordinates for each image, shape (n_samples, 2).
    explained_variance : np.ndarray
        Array containing explained variance for each principal component.
    x_bins : int, optional
        Number of bins along the x-axis for spatial binning (default is 31).
    y_bins : int, optional
        Number of bins along the y-axis for spatial binning (default is 31).
    responses : np.ndarray, optional
        Array of response values for each image, shape (n_samples,). If provided, will overlay
        response magnitudes as colored markers at each location.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the vector field plot and PC insets. Call plt.show() to display,
        or fig.savefig() to save.
    Raises
    ------
    ValueError
        If no images are found in the coordinate bins (e.g., due to bin size or coordinate range).
    Notes
    -----
    - This function is primarily intended for visualization in Jupyter notebooks.
    - The vector field arrows represent the projection of binned images and LSTA responses onto the first two
    principal components.
    - Insets display the spatial structure of PC1 and PC2 for interpretability.
    - If responses are provided, they will be averaged within bins and displayed as colored markers.
    """
    lsta_library = lsta_library[:, channel, :, :]
    x_size = lsta_library.shape[-2]
    y_size = lsta_library.shape[-1]

    # Bin edges for PC1 and PC2 coordinates
    x_edges = np.linspace(images_coordinate[:, 0].min(), images_coordinate[:, 0].max(), x_bins + 1)
    y_edges = np.linspace(images_coordinate[:, 1].min(), images_coordinate[:, 1].max(), y_bins + 1)

    # Digitize coordinates to bins
    x_bin_idx = np.digitize(images_coordinate[:, 0], x_edges) - 1
    y_bin_idx = np.digitize(images_coordinate[:, 1], y_edges) - 1

    # Mask for valid bins
    valid_mask = (x_bin_idx >= 0) & (x_bin_idx < x_bins) & (y_bin_idx >= 0) & (y_bin_idx < y_bins)

    # Prepare lists for binned images and lstas
    binned_imgs_list = []
    binned_lstas_list = []
    bin_coords_list = []
    binned_responses_list = []

    # For each bin, average images and lstas assigned to it
    for xi in range(x_bins):
        for yi in range(y_bins):
            bin_mask = valid_mask & (x_bin_idx == xi) & (y_bin_idx == yi)
            if np.any(bin_mask):
                binned_imgs_list.append(images[bin_mask].mean(axis=0))
                binned_lstas_list.append(lsta_library[bin_mask].mean(axis=0))
                # Use bin center as coordinate
                bin_coords_list.append([0.5 * (x_edges[xi] + x_edges[xi + 1]), 0.5 * (y_edges[yi] + y_edges[yi + 1])])
                # Average responses within bin if provided
                if responses is not None:
                    binned_responses_list.append(responses[bin_mask].mean())

    binned_imgs = np.array(binned_imgs_list)
    binned_lstas = np.array(binned_lstas_list)
    images_coordinate = np.array(bin_coords_list)
    if responses is not None:
        binned_responses = np.array(binned_responses_list)
    # Check if we have any binned data

    if len(binned_imgs) == 0:
        raise ValueError("No images found in coordinate bins. Try adjusting bin size or coordinate range.")

    flatten_binned_imgs = binned_imgs.reshape(binned_imgs.shape[0], -1)
    flatten_binned_lstas = binned_lstas.reshape(binned_lstas.shape[0], -1)

    binned_arrowtails = np.array([[np.dot(PC1, img), np.dot(PC2, img)] for img in flatten_binned_imgs])
    binned_arrowheads = np.array([[np.dot(PC1, lsta), np.dot(PC2, lsta)] for lsta in flatten_binned_lstas])

    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Calculate plot limits
    xlim = max(np.abs(binned_arrowtails[:, 0]).max(), np.abs(images_coordinate[:, 0]).max()) * 1.1
    ylim = max(np.abs(binned_arrowtails[:, 1]).max(), np.abs(images_coordinate[:, 1]).max()) * 1.1
    plot_limit = max(xlim, ylim)
    
    # Overlay response magnitudes as density plot if provided
    if responses is not None:
        # Create a grid for interpolation
        grid_resolution = 100
        xi = np.linspace(-plot_limit, plot_limit, grid_resolution)
        yi = np.linspace(-plot_limit, plot_limit, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate the response values onto the grid
        zi = griddata(
            binned_arrowtails,
            binned_responses,
            (xi_grid, yi_grid),
            method='linear',
            fill_value=np.nan
        )
        
        # Create the density plot using pcolormesh
        density = ax.pcolormesh(xi, yi, zi, cmap='viridis', alpha=0.4, shading='gouraud', zorder=0)
        
        # Add colorbar
        cbar = plt.colorbar(density, ax=ax)
        cbar.set_label('Response magnitude', size=14)
    
    ax.quiver(
        binned_arrowtails[:, 0],
        binned_arrowtails[:, 1],
        binned_arrowheads[:, 0],
        binned_arrowheads[:, 1],
        color="black",
        width=0.002,
        scale_units="xy",
        angles="xy",
        scale=binned_arrowheads.max(),
        zorder=2
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add arrowheads to axes using matplotlib arrow function
    ax.arrow(
        -plot_limit * 0.75, 0, 1.5 * plot_limit, 0, head_width=plot_limit * 0.02, head_length=plot_limit * 0.02, fc="k", ec="k", linewidth=1
    )
    ax.arrow(
        0, -plot_limit * 0.75, 0, 1.5 * plot_limit, head_width=plot_limit * 0.02, head_length=plot_limit * 0.02, fc="k", ec="k", linewidth=1
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim((-plot_limit, plot_limit))
    ax.set_ylim((-plot_limit, plot_limit))

    plot_pc_insets(fig, PC1, PC2, x_size, y_size, explained_variance)
    return fig
