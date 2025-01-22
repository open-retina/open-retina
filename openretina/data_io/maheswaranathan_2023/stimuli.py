"""
Minimal stimulus loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1016/j.neuron.2023.06.007
Data: https://doi.org/10.25740/rk663dm5577
"""

import os
from typing import Literal

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies
from openretina.utils.h5_handling import load_dataset_from_h5

CLIP_LENGTH = 90  # in frames @ 30 fps


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    stim_type: Literal["naturalscene", "whitenoise"] = "naturalscene",
    normalize_stimuli: bool = True,
) -> dict[str, MoviesTrainTestSplit]:
    """
    Load all stimuli from sessions within subfolders in a given base data path.

    The base data path should point to the location of neural_code_data/ganglion_cell_data
    (See https://doi.org/10.25740/rk663dm5577 for dataset download)
    """
    stimuli_all_sessions = {}
    for session in [x.name for x in os.scandir(os.fspath(base_data_path)) if x.is_dir()]:
        session_path = os.path.normpath(os.path.join(base_data_path, session))
        for recording_file in os.listdir(session_path):
            if str(recording_file).endswith(f"{stim_type}.h5"):
                recording_file = os.path.join(session_path, recording_file)

                print(f"Loading stimuli from {recording_file}")

                # Load video stimuli
                train_video = load_dataset_from_h5(recording_file, "/train/stimulus")
                test_video = load_dataset_from_h5(recording_file, "/test/stimulus")

                # Add channel dimension
                train_video = train_video[None, ...]
                test_video = test_video[None, ...]

                if normalize_stimuli:
                    train_video, test_video = normalize_train_test_movies(train_video, test_video)

                stimuli_all_sessions[str(session)] = MoviesTrainTestSplit(
                    train=train_video,
                    test=test_video,
                    stim_id=stim_type,
                )
    return stimuli_all_sessions
