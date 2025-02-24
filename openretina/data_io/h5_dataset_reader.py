import logging
import os.path
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies, ResponsesTrainTestSplit

LOGGER = logging.getLogger(__name__)

_STIMULUS_FOLDER = "stimuli"
_RESPONSES_PREFIX = "responses_"
_SESSION_SPECIFIC_STIMULUS_PREFIX = "stimulus_"
_SESSION_INFO_KEY = "session_info"



class TrainTestStimuliProcessor:
    """ Manages the processing of stimuli. Useful for caching results"""
    def __init__(
            self,
            test_stimuli: Iterable[str],
            name_to_stimulus: dict[str, np.ndarray],
            train_stimuli: Iterable[str] | None = None,
            train_mean: float | None = None,
            train_var: float | None = None,
    ):
        self._test_stimuli = set(test_stimuli)
        self._name_to_stimulus = name_to_stimulus
        # self._train_stimuli = set(train_stimuli)

    def process(self, stimulus_names: Iterable[str], session_specific_stimuli: dict[str, np.ndarray]) -> MoviesTrainTestSplit:
        # Todo: maybe add caching in the future to reduce memory consumption
        train_stimulus_array, test_stimulus_array = [], []

        # Todo implement normalization
        for name in stimulus_names:
            stimulus = session_specific_stimuli.get(name, self._name_to_stimulus[name])
            if name in self._test_stimuli:
                test_stimulus_array.append(stimulus)
            else:
                train_stimulus_array.append(stimulus)
        # concatenate stimuli over time dimension
        train_stimuli = np.concatenate(train_stimulus_array, axis=1)
        test_stimuli = np.concatenate(test_stimulus_array, axis=1)

        return MoviesTrainTestSplit(train_stimuli, test_stimuli)


def load_stimuli(
        base_data_path: str,
        test_names: Iterable[str],
        normalize_stimuli: bool,
        stimulus_size: list[int],
) -> dict[str, MoviesTrainTestSplit]:
    if not os.path.isdir(base_data_path):
        raise ValueError(f"{base_data_path=} is not a directory.")
    test_stimuli_names = sorted(test_names)

    name_to_stimulus = {}
    # first load all stimuli from stimuli folder
    stimulus_folder = os.path.join(base_data_path, _STIMULUS_FOLDER)
    if os.path.isdir(stimulus_folder):
        for file_name in [x for x in os.listdir(stimulus_folder) if x.endswith(".npy")]:
            name = file_name.removesuffix(".npy")
            stim = np.load(os.path.join(stimulus_folder, file_name))
            name_to_stimulus[name] = stim
    else:
        LOGGER.warning(f"Did not find {stimulus_folder=}")

    result = {}
    stimulus_processor = TrainTestStimuliProcessor(test_stimuli_names, name_to_stimulus)
    # build MoviesTrainTestSplit for each session
    for file_name in [x for x in os.listdir(base_data_path) if x.endswith(".h5") or x.endswith(".hdf5")]:
        with h5py.File(os.path.join(base_data_path, file_name), "r") as f:
            stimuli_with_responses = sorted(x.removeprefix(_RESPONSES_PREFIX) for x in f.keys() if x.startswith(_RESPONSES_PREFIX))
            if len(stimuli_with_responses) == 0:
                LOGGER.warning(f"No responses found for {file_name}: {list(f.keys())}")
                continue

            session_specific_stimuli = {
                x.removeprefix(_SESSION_SPECIFIC_STIMULUS_PREFIX): f[x] for x in f.keys()
                if x.startswith(_SESSION_SPECIFIC_STIMULUS_PREFIX)
            }
            train_test_split = stimulus_processor.process(stimuli_with_responses, session_specific_stimuli)
        session_name = Path(file_name).stem
        result[session_name] = train_test_split
    return result


def load_responses(base_data_path: str, test_names: Iterable[str]) -> dict[str, ResponsesTrainTestSplit]:
    result = {}
    test_names_set = set(test_names)
    # build MoviesTrainTestSplit for each session
    for file_name in [x for x in os.listdir(base_data_path) if x.endswith(".h5") or x.endswith(".hdf5")]:
        with h5py.File(os.path.join(base_data_path, file_name), "r") as f:
            stimuli_with_responses = sorted(
                x for x in f.keys() if x.startswith(_RESPONSES_PREFIX))
            train_responses = np.concatenate([f[x] for x in stimuli_with_responses if x.removeprefix(_RESPONSES_PREFIX) not in test_names_set], axis=-1)
            test_responses = np.concatenate([f[x] for x in stimuli_with_responses if x.removeprefix(_RESPONSES_PREFIX) in test_names_set], axis=-1)
            if train_responses.shape[0] != test_responses.shape[0]:
                raise ValueError(f"Train responses and test responses have a different number of neurons: "
                                 f"{train_responses.shape[0]=} {test_responses.shape[0]=}")
            # potentially improve: not sure if that fails for strings or other datatypes
            session_kwargs = {k: np.array(v) for k, v in f.get(_SESSION_INFO_KEY, {}).items()}

        session_name = Path(file_name).stem
        result[session_name] = ResponsesTrainTestSplit(train_responses, test_responses, session_kwargs=session_kwargs)
    return result
