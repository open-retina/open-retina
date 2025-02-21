import os.path
from functools import lru_cache
from typing import Iterable

import h5py
import numpy as np

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies

_STIMULUS_FOLDER = "stimuli"
_RESPONSES_PREFIX = "responses_"


def load_stimuli(
        base_data_path: str,
        normalize_stimuli: bool,
        test_names: Iterable[str]
) -> dict[str, MoviesTrainTestSplit]:
    if not os.path.isdir(base_data_path):
        raise ValueError(f"{base_data_path=} is not a directory.")
    test_stimuli_names = sorted(test_names)

    name_to_stimulus = {}
    # first load all stimuli from stimuli folder
    stimulus_folder = os.path.join(base_data_path, _STIMULUS_FOLDER)
    if os.path.isdir(stimulus_folder):
        for file_name in [x for x in os.listdir(stimulus_folder) if x.endswith(".npz")]:
            name = file_name.removesuffix(".npz")
            stim = np.load(os.path.join(stimulus_folder, file_name))
            name_to_stimulus[name] = stim
    else:
        print(f"Did not find {stimulus_folder=}")


    # create standard names



    # build MoviesTrainTestSplit for each session
    for file_name in [x for x in os.listdir(base_data_path) if x.endswith(".h5") or x.endswith(".hdf5")]:
        with h5py.File(os.path.join(base_data_path, file_name), "r") as f:
            stimuli_with_responses = sorted(x.removeprefix(_RESPONSES_PREFIX) for x in f.keys() if x.startswith(_RESPONSES_PREFIX))
            test_stimuli = []















