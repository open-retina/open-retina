"""
Minimal stimuli loading utilities to train a model on the data used in Karamanlis et al. 2024

Paper: https://doi.org/10.1038/s41586-024-08212-3
Data: https://doi.org/10.12751/g-node.ejk8kx
"""

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import Resize
from tqdm.auto import tqdm

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies
from openretina.utils.file_utils import get_local_file_path, unzip_and_cleanup
from openretina.utils.h5_handling import load_dataset_from_h5


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
    if str(session_path).endswith(".zip"):
        session_path = unzip_and_cleanup(Path(session_path))

    mat_file = None
    npz_file = None
    downsample = Resize(downsampled_size)

    for recording_file in os.listdir(session_path):
        full_path = os.path.join(session_path, recording_file)

        # First check for a `.npz` file, which is a saved version of the preprocessing of the stimuli
        if recording_file.endswith(
            f"{stim_type}_data_extracted_stimuli_{downsampled_size[0]}_{downsampled_size[1]}.npz"
        ):
            npz_file = full_path
            break  # Prefer `.npz` file

        # Fall back to the original `.mat` file if no `.npz` file is found (e.g., for the first run of the script)
        elif recording_file.endswith(f"{stim_type}_data.mat"):
            mat_file = full_path

    if npz_file:
        tqdm.write(f"Loading stimuli from cached {npz_file}")
        video_data = np.load(npz_file)
        train_video = video_data["train_data"]
        test_video = video_data["test_data"]

    elif mat_file:
        tqdm.write(f"Loading stimuli from {mat_file}")
        train_images = load_dataset_from_h5(mat_file, "runningImages")
        test_images = load_dataset_from_h5(mat_file, "frozenImages")

        frozen_fixations = load_dataset_from_h5(mat_file, "frozenfixations").astype(int)
        running_fixations = load_dataset_from_h5(mat_file, "runningfixations").astype(int)

        test_video = return_fix_movie_torch((600, 800), rearrange(test_images, "n x y -> y x n"), frozen_fixations.T)
        test_video = downsample(rearrange(test_video, "h w n -> 1 n h w")).cpu().numpy()

        train_videos = []
        for trial in tqdm(running_fixations, desc=f"Composing training video for {Path(mat_file).parent.name}"):
            train_snippet = return_fix_movie_torch((600, 800), rearrange(train_images, "n x y -> y x n"), trial.T)
            train_videos.append(downsample(rearrange(train_snippet, "h w n -> 1 n h w")))
        train_video = torch.cat(train_videos, dim=1).cpu().numpy()

        np.savez_compressed(
            f"{mat_file[:-4]}_extracted_stimuli_{downsampled_size[0]}_{downsampled_size[1]}.npz",
            train_data=train_video,
            test_data=test_video,
        )
    else:
        warnings.warn(
            f"No file with postfix {stim_type}_data.mat (or preprocessed version) found in folder {session_path}",
        )
        return None

    if normalize_stimuli:
        train_video, test_video, norm_dict = normalize_train_test_movies(train_video, test_video)
    else:
        norm_dict = {"norm_mean": None, "norm_std": None}

    return MoviesTrainTestSplit(
        train=train_video,
        test=test_video,
        stim_id=stim_type,
        random_sequences=None,
        norm_mean=norm_dict["norm_mean"],
        norm_std=norm_dict["norm_std"],
    )


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"] = "fixationmovie",
    normalize_stimuli: bool = True,
    specie: Literal["mouse", "marmoset"] = "mouse",
    downsampled_size: tuple[int, int] = (60, 80),
) -> dict[str, MoviesTrainTestSplit]:
    """
    Load stimuli for all sessions.

    Args:
        base_data_path (str | os.PathLike): Base directory containing session data.
                                            Can also be the path to the "sessions" folder in the huggingface mirror.
        "https://huggingface.co/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024/sessions"
        stim_type (str): The stimulus type to filter files.
        normalize_stimuli (bool): Whether to normalize the stimuli.
        specie (str): Animal species (e.g., "mouse", "marmoset").
        downsampled_size (tuple[int, int]): Size to downsample the stimuli.

    Returns:
        dict[str, MoviesTrainTestSplit]: Dictionary mapping session names to stimulus data.
    """
    base_data_path = get_local_file_path(str(base_data_path))

    stimuli_all_sessions = {}
    exp_sessions = [
        path
        for path in os.listdir(base_data_path)
        if (os.path.isdir(os.path.join(base_data_path, path)) or path.endswith(".zip")) and specie in path
    ]

    assert len(exp_sessions) > 0, (
        f"No data directories found in {base_data_path} for animal {specie}."
        "Please check the path and the animal species provided, and that you unrared the data files."
    )

    for session in tqdm(exp_sessions, desc="Processing sessions"):
        session_path = os.path.join(base_data_path, session)
        stimuli = load_stimuli_for_session(session_path, stim_type, downsampled_size, normalize_stimuli)
        if stimuli:
            stimuli_all_sessions[session.split("/")[-1]] = stimuli

    return stimuli_all_sessions


def return_fix_movie_torch(
    screensize: tuple[int, int], im_ensemble: np.ndarray | torch.Tensor, list_fixations: np.ndarray
) -> torch.Tensor:
    """
    Generates a movie of fixations using an image ensemble.
    Faster version using PyTorch, which supports fancier indexing.

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

    blockstimulus = torch.zeros((Ny, Nx, Nframes), dtype=torch.float32)
    im_ensemble = torch.tensor(im_ensemble, dtype=torch.float32)
    list_fixations_tensor = torch.tensor(list_fixations, dtype=torch.int64)

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
        x_or = torch.tensor(x_or, dtype=torch.int64).unsqueeze(0)  # type: ignore
        y_or = torch.tensor(y_or, dtype=torch.int64).unsqueeze(1)  # type: ignore
        blockstimulus[ymin : ymax + 1, xmin : xmax + 1, ifix] = im_ensemble[
            y_or, x_or, list_fixations_tensor[0, ifix] - 1
        ]

    return blockstimulus


def return_fix_movie(screensize: tuple[int, int], im_ensemble: np.ndarray, list_fixations: np.ndarray) -> np.ndarray:
    """
    Generates a movie of fixations using an image ensemble.

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
        x_or = x_or.astype(int)
        y_or = y_or.astype(int)
        blockstimulus[ymin : ymax + 1, xmin : xmax + 1, ifix] = im_ensemble[np.ix_(y_or, x_or)][
            :, :, list_fixations[0, ifix] - 1
        ]

    return blockstimulus


def get_ranges(
    tr_x: int, tr_y: int, Nxs: int, Nys: int, Nx: int, Ny: int, rx: np.ndarray, ry: np.ndarray
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    """
    Compute valid ranges and indices based on translations.

    Parameters:
    tr_x, tr_y : int
        Translation values for x and y.
    Nxs, Nys : int
        Screen dimensions (Nx, Ny).
    Nx, Ny : int
        Image dimensions.
    rx, ry : ndarray
        Ranges for x and y.

    Returns:
    xmin, xmax, ymin, ymax : int
        Indices for cropping (screen coordinates).
    x_or, y_or : ndarray
        Translated indices for image coordinates.
    """
    ymin = max((Nys // 2) - tr_y, 0)  # Minimum y index for the screen
    ymax = min(Ny + (Nys // 2) - tr_y, Nys)  # Maximum y index for the screen

    xmin = max((Nxs // 2) - tr_x, 0)  # Minimum x index for the screen
    xmax = min(Nx + (Nxs // 2) - tr_x, Nxs)  # Maximum x index for the screen

    # Filter valid indices within these bounds
    rx_use = (rx >= xmin) & (rx <= xmax)
    ry_use = (ry >= ymin) & (ry <= ymax)

    # Translate these to the original image coordinates
    x_or = tr_x - (Nxs // 2) + rx[rx_use] - 1
    y_or = tr_y - (Nys // 2) + ry[ry_use] - 1

    # Find the bounds again for the block on the screen
    xmin = np.where(rx_use)[0][0]
    xmax = np.where(rx_use)[0][-1]
    ymin = np.where(ry_use)[0][0]
    ymax = np.where(ry_use)[0][-1]

    return xmin, xmax, ymin, ymax, x_or, y_or
