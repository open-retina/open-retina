"""
Minimal stimuli loading utilities to train a model on the data used in Karamanlis et al. 2024

Paper: https://doi.org/10.1038/s41586-024-08212-3
Data: https://doi.org/10.12751/g-node.ejk8kx

This version uses numpy instead of PyTorch for all operations.
"""

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from einops import rearrange
from tqdm.auto import tqdm

from ..base import MoviesTrainTestSplit, normalize_train_test_movies


def resize_numpy(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Simple numpy-based image resizing using nearest neighbor interpolation.
    For more sophisticated resizing, consider using scipy.ndimage.zoom or PIL.
    """
    from scipy.ndimage import zoom
    
    if len(image.shape) == 2:
        # Single image
        h, w = image.shape
        target_h, target_w = target_size
        zoom_factors = (target_h / h, target_w / w)
        return zoom(image, zoom_factors, order=1)  # Linear interpolation
    elif len(image.shape) == 3:
        # Multiple images or channels
        h, w = image.shape[:2]
        target_h, target_w = target_size
        zoom_factors = (target_h / h, target_w / w, 1)
        return zoom(image, zoom_factors, order=1)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def load_stimuli_for_session(
    session_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"],
    downsampled_size: tuple[int, int],
    normalize_stimuli: bool,
) -> MoviesTrainTestSplit | None:
    """
    Load stimuli for a single session.

    Args:
        session_path (str): Path to the session directory.
        stim_type (str): The stimulus type to filter files.
        downsampled_size (tuple[int, int]): Size to downsample the stimuli.
        normalize_stimuli (bool): Whether to normalize the stimuli.

    Returns:
        MoviesTrainTestSplit | None: Loaded stimuli for the session or None if no relevant file found.
    """
    # Note: This is a simplified version. The original implementation
    # would need to be adapted based on the specific data format and
    # requirements for loading Karamanlis 2024 data.
    
    # For now, return None to indicate this needs to be implemented
    # based on the specific data format
    warnings.warn(
        "load_stimuli_for_session for Karamanlis 2024 data needs to be implemented "
        "based on the specific data format and requirements.",
        UserWarning
    )
    return None


def get_ranges(x_fix, y_fix, Nx, Ny, Nxim, Nyim, x_screen, y_screen):
    """
    Calculate ranges for image cropping and positioning.
    
    This is a helper function for return_fix_movie_numpy.
    """
    # Calculate the ranges for cropping the image and positioning on screen
    x_start = max(0, x_fix - Nxim // 2)
    x_end = min(Nx, x_fix + Nxim // 2)
    y_start = max(0, y_fix - Nyim // 2)
    y_end = min(Ny, y_fix + Nyim // 2)
    
    # Calculate corresponding ranges in the original image
    img_x_start = max(0, Nxim // 2 - x_fix)
    img_x_end = img_x_start + (x_end - x_start)
    img_y_start = max(0, Nyim // 2 - y_fix)
    img_y_end = img_y_start + (y_end - y_start)
    
    return x_start, x_end, y_start, y_end, np.arange(img_x_start, img_x_end), np.arange(img_y_start, img_y_end)


def return_fix_movie_numpy(
    screensize: tuple[int, int], im_ensemble: np.ndarray, list_fixations: np.ndarray
) -> np.ndarray:
    """
    Generates a movie of fixations using an image ensemble.
    Numpy version of the original PyTorch implementation.

    Python port from: https://github.com/dimokaramanlis/subunit_grid_model/blob/main/code/returnFixMovie.m

    Parameters:
    screensize : tuple
        (Ny, Nx) size of the monitor, e.g., (600, 800).
    im_ensemble : ndarray
        Ensemble of images with dimensions (Nyim, Nxim, Nimages).
    list_fixations : ndarray
        Array of shape (3, Nframes) containing fixations (image index, x, y).

    Returns:
    blockstimulus : ndarray
        Movie of fixations with shape (Ny, Nx, Nframes), where the images
        presented at each frame are shifted based on gaze data.
    """
    Ny, Nx = screensize
    Nyim, Nxim, Nimages = im_ensemble.shape
    Nframes = list_fixations.shape[1]

    blockstimulus = np.zeros((Ny, Nx, Nframes), dtype=np.float32)
    im_ensemble = im_ensemble.astype(np.float32)
    list_fixations = list_fixations.astype(np.int64)

    for ifix in range(Nframes):
        xmin, xmax, ymin, ymax, x_or, y_or = get_ranges(
            list_fixations[1, ifix],
            list_fixations[2, ifix],
            Nx,
            Ny,
            Nxim,
            Nyim,
            np.arange(1, Nx + 1),
            np.arange(1, Ny + 1),
        )
        
        # Use numpy advanced indexing instead of torch indexing
        img_idx = int(list_fixations[0, ifix])
        if img_idx < Nimages:
            # Extract the relevant portion of the image
            img_patch = im_ensemble[np.ix_(y_or, x_or, [img_idx])][:, :, 0]
            blockstimulus[ymin:ymax+1, xmin:xmax+1, ifix] = img_patch

    return blockstimulus


def load_karamanlis_movies(
    train_images: np.ndarray,
    test_images: np.ndarray,
    train_fixations: list[np.ndarray],
    frozen_fixations: np.ndarray,
    screensize: tuple[int, int] = (600, 800),
    normalize_stimuli: bool = True,
) -> MoviesTrainTestSplit:
    """
    Load and process Karamanlis 2024 movie data using numpy operations.
    
    Args:
        train_images: Training images array
        test_images: Test images array
        train_fixations: List of fixation arrays for training trials
        frozen_fixations: Frozen fixations for test
        screensize: Screen size tuple (height, width)
        normalize_stimuli: Whether to normalize the stimuli
        
    Returns:
        MoviesTrainTestSplit object containing processed movies
    """
    # Generate test video using frozen fixations
    test_video = return_fix_movie_numpy(screensize, rearrange(test_images, "n x y -> y x n"), frozen_fixations.T)
    
    # Generate training videos for each trial
    train_videos = []
    for trial in train_fixations:
        train_snippet = return_fix_movie_numpy(screensize, rearrange(train_images, "n x y -> y x n"), trial.T)
        train_videos.append(train_snippet)
    
    train_video = np.concatenate(train_videos, axis=2)  # Concatenate along time dimension
    
    # Rearrange to (channels, time, height, width) format
    train_video = rearrange(train_video, "h w t -> 1 t h w")
    test_video = rearrange(test_video, "h w t -> 1 t h w")
    
    if normalize_stimuli:
        train_video, test_video, norm_stats = normalize_train_test_movies(train_video, test_video)
        return MoviesTrainTestSplit(
            train=train_video,
            test=test_video,
            norm_mean=norm_stats["norm_mean"],
            norm_std=norm_stats["norm_std"]
        )
    else:
        return MoviesTrainTestSplit(train=train_video, test=test_video)


def load_stimuli_all_sessions(
    data_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"],
    downsampled_size: tuple[int, int] = (36, 64),
    normalize_stimuli: bool = True,
) -> dict[str, MoviesTrainTestSplit]:
    """
    Load stimuli for all sessions in the data path.

    Args:
        data_path (str): Path to the data directory.
        stim_type (str): The stimulus type to filter files.
        downsampled_size (tuple[int, int]): Size to downsample the stimuli.
        normalize_stimuli (bool): Whether to normalize the stimuli.

    Returns:
        dict[str, MoviesTrainTestSplit]: Dictionary mapping session names to stimuli.
    """
    stimuli_all_sessions = {}
    
    # This would need to be implemented based on the specific directory structure
    # and data format of the Karamanlis 2024 dataset
    warnings.warn(
        "load_stimuli_all_sessions for Karamanlis 2024 data needs to be implemented "
        "based on the specific data format and directory structure.",
        UserWarning
    )
    
    return stimuli_all_sessions