"""
Minimal stimulus loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1016/j.neuron.2023.06.007
Data: https://doi.org/10.25740/rk663dm5577
"""

import os
from typing import Any

from openretina.data_io.movie_dataloader import MoviesTrainTestSplit
from openretina.utils.h5_handling import load_dataset_from_h5

CLIP_LENGTH = 90  # in frames @ 30 fps


def load_all_sessions(
    base_data_path: str | os.PathLike,
    response_type: str = "firing_rate_20ms",
    stim_type: str = "naturalscene",
    fr_normalization: float = 1,
    normalize_stimuli: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load all ganglion cell data from sessions within subfolders in a given base data path.

    The base data path should point to the location of neural_code_data/ganglion_cell_data
    (See https://doi.org/10.25740/rk663dm5577 for dataset download)
    """
    responses_all_sessions = {}
    stimuli_all_sessions = {}
    for session in os.listdir(base_data_path):
        # List sessions in the base data path
        session_path = os.path.join(base_data_path, session)
        if not os.path.isdir(session_path):
            continue
        for recording_file in os.listdir(
            session_path,
        ):
            if str(recording_file).endswith(f"{stim_type}.h5"):
                recording_file = os.path.join(session_path, recording_file)

                print(f"Loading data from {recording_file}")

                # Load video stimuli
                train_video = load_dataset_from_h5(recording_file, "/train/stimulus")
                test_video = load_dataset_from_h5(recording_file, "/test/stimulus")

                # Add channel dimension
                train_video = train_video[None, ...]
                test_video = test_video[None, ...]

                if normalize_stimuli:
                    train_video = train_video - train_video.mean() / train_video.std()
                    test_video = test_video - test_video.mean() / test_video.std()

                stimuli_all_sessions["".join(session.split("/")[-1])] = MoviesTrainTestSplit(
                    train=train_video,
                    test=test_video,
                )

                train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")
                test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")

                assert (
                    train_session_data.shape[0] == test_session_data.shape[0]
                ), "Train and test responses should have the same number of neurons."

                assert (
                    train_session_data.shape[1] == train_video.shape[1]
                ), "The number of timepoints in the responses does not match the number of frames in the video."

                responses_all_sessions["".join(session.split("/")[-1])] = {
                    "responses_final": {
                        "train": train_session_data / fr_normalization,
                        "test": test_session_data / fr_normalization,
                    },
                    "stim_id": "salamander_natural",
                }

    return responses_all_sessions, stimuli_all_sessions
