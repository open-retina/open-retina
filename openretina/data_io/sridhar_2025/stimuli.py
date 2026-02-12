import os
import pickle

import numpy as np
from jaxtyping import Float
from tqdm import tqdm

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path


def load_frames(
    img_dir_name: str | os.PathLike, frame_file: str, full_img_w: int, full_img_h: int
) -> Float[np.ndarray, "frames width height"]:
    """
    loads all stimulus frames of the movie into memory
    """
    img_dir_name = get_local_file_path(str(img_dir_name))
    print("Loading all frames from:", img_dir_name, "into memory")
    images = os.listdir(img_dir_name)
    images = [frame for frame in images if frame_file in frame]
    all_frames = np.zeros((len(images), full_img_w, full_img_h), dtype=np.float16)
    i = 0
    for img_file in tqdm(sorted(images)):
        img = np.load(f"{img_dir_name}/{img_file}")

        all_frames[i] = img / 255
        i += 1
    return all_frames


def process_fixations(fixations, flip_imgs: bool = False, select_flip: int | None = None) -> list[dict[str, int]]:
    if not flip_imgs:
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0]),
            }
            for f in fixations
        ]
    else:
        print("FLIPPING THE OTHER WAY AROUND!")
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0] == "0"),
            }
            for f in fixations
        ]
    if select_flip is not None:
        fixations = [x for x in fixations if x["flip"] == select_flip]
    return fixations


def build_placeholder_movies(
    session_ids,
    *,
    channels: int,
    height: int,
    width: int,
    time_bins: int = 1,
    test_time_bins: int | None = None,
    stim_id_prefix: str = "sridhar_2025",
    norm_mean: float = 0.0,
    norm_std: float = 1.0,
) -> dict[str, MoviesTrainTestSplit]:
    """
    Create lightweight MoviesTrainTestSplit placeholders that encode spatial dimensions of the stimuli.
    Will not be used directly by the dataloader and model, but rather for `data_info` computation.

    Note: For accurate frame counts, use `build_movies_from_responses` instead, which reads the
    response files to determine actual training/test frame counts.
    """
    if test_time_bins is None:
        test_time_bins = time_bins

    movies = {}
    for session_id in session_ids:
        train = np.zeros((channels, time_bins, height, width), dtype=np.float32)
        test = np.zeros((channels, test_time_bins, height, width), dtype=np.float32)
        movies[session_id] = MoviesTrainTestSplit(
            train=train,
            test=test,
            stim_id=f"{stim_id_prefix}_{session_id}",
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    return movies


def build_movies_from_responses(
    base_path: str | os.PathLike,
    response_files: dict[str, str],
    *,
    channels: int,
    height: int,
    width: int,
    stim_id_prefix: str = "sridhar_2025",
    stimulus_seed: int = 0,
    norm_mean: float = 0.0,
    norm_std: float = 1.0,
) -> dict[str, MoviesTrainTestSplit]:
    """
    Create MoviesTrainTestSplit placeholders with accurate frame counts by reading response files.

    This function loads the response pickle files to determine the actual number of training and
    test frames per session, accounting for stimulus_seed filtering. The resulting placeholders
    have correct time dimensions for accurate dataset statistics reporting.

    Each session gets a unique stim_id because different sessions see different visual content
    due to different fixation patterns and trial assignments.

    Args:
        base_path: Base directory containing the response files.
        response_files: Dictionary mapping session IDs to response pickle file paths (relative to base_path).
        channels: Number of channels in the stimulus.
        height: Height of the stimulus frames.
        width: Width of the stimulus frames.
        stim_id_prefix: Prefix for the stimulus ID (each session will have "{prefix}_{session_id}").
        stimulus_seed: Random seed used for trial selection (affects frame counts for some sessions).
        norm_mean: Normalization mean for the stimulus.
        norm_std: Normalization std for the stimulus.

    Returns:
        Dictionary mapping session IDs to MoviesTrainTestSplit placeholders with accurate frame counts.
    """

    base_path = get_local_file_path(str(base_path))
    movies = {}

    for session_id, response_file in response_files.items():
        with open(os.path.join(base_path, response_file), "rb") as f:
            data = pickle.load(f)

        train_responses = data["train_responses"]
        test_responses = data["test_responses"]

        # Apply seed filtering logic (same as in responses.py load_responses)
        if "seeds" in data:
            seed_info = data["seeds"]
            if stimulus_seed in seed_info:
                trials = data["trial_separation"][stimulus_seed]
                train_responses = train_responses[:, :, trials]
            # Note: test responses are not filtered by seed in the dataloader

        # Get frame counts
        _, frames_per_trial, n_trials = train_responses.shape
        train_time_bins = frames_per_trial * n_trials
        test_time_bins = test_responses.shape[1]

        # Create placeholder arrays with correct dimensions
        train = np.zeros((channels, train_time_bins, height, width), dtype=np.float32)
        test = np.zeros((channels, test_time_bins, height, width), dtype=np.float32)

        movies[session_id] = MoviesTrainTestSplit(
            train=train,
            test=test,
            stim_id=f"{stim_id_prefix}_{session_id}",
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    return movies
