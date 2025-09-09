"""
Minimal stimulus loading utilities to train a model on the data used in Goldin et al. 2022.

Paper: https://doi.org/10.1038/s41467-022-33242-8
Data: https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted]

OpenRetina provides a mirror of the dataset on huggingface:
### TODO: Update this link when the dataset is available
"""

import os
from typing import Literal

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_dataset_from_h5

CLIP_LENGTH = 1  # one frame of 300ms


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    stim_type: Literal["naturalscene"] = "naturalscene",
    specie: Literal["mouse", "axolotl"] = "mouse",
    normalize_stimuli: bool = True,
) -> dict[str, MoviesTrainTestSplit]:
    """
    Load all stimuli from sessions within subfolders in a given base data path.

    The base data path should point to the location of the `neural_code_data` folder.
    (See https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted] for dataset download).

    Alternatively, base_data_path can point directly to our huggingface mirror of the dataset, which will then
    be downloaded and extracted automatically to the openretina cache directory.
    TO DO: Update this link when the dataset is available
    """
    # Resolve data path
    base_data_path = get_local_file_path(str(base_data_path))
    full_data_path = base_data_path

    stimuli_all_sessions = {}
    for session in [x.name for x in os.scandir(os.fspath(full_data_path))]:
        if specie not in session:
            continue
        session_path = os.path.normpath(os.path.join(full_data_path, session))
        recording_file = session_path

        # Load image stimuli
        train_image = load_dataset_from_h5(recording_file, "/train/stimulus")
        test_image = load_dataset_from_h5(recording_file, "/test/stimulus")

        # Add channel dimension
        train_image = train_image[None, ...]
        test_image = test_image[None, ...]

        if normalize_stimuli:
            train_image, test_image, norm_dict = normalize_train_test_movies(train_image, test_image)
        else:
            norm_dict = {"norm_mean": None, "norm_std": None}

        stimuli_all_sessions[str(session).replace(".h5", "")] = MoviesTrainTestSplit(
            train=train_image,
            test=test_image,
            stim_id=stim_type,
            random_sequences=None,
            norm_mean=norm_dict["norm_mean"],
            norm_std=norm_dict["norm_std"],
        )
    print(stimuli_all_sessions.keys())
    return stimuli_all_sessions
