import logging
import os.path
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from jaxtyping import Float

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit

LOGGER = logging.getLogger(__name__)

_STIMULUS_FOLDER = "stimuli"
_RESPONSES_PREFIX = "responses_"
_SESSION_SPECIFIC_STIMULUS_PREFIX = "stimulus_"
_SESSION_INFO_KEY = "session_info"


class TrainTestStimuliProcessor:
    """Manages the processing of stimuli. Useful for caching results"""

    def __init__(
        self,
        test_stimuli: Iterable[str],
        name_to_stimulus: dict[str, np.ndarray],
        norm_mean: float | None,
        norm_std: float | None,
        train_stimuli: Iterable[str] | None = None,
    ):
        self._test_stimuli = set(test_stimuli)
        self._name_to_stimulus = name_to_stimulus
        self._norm_mean = 0.0 if norm_mean is None else norm_mean
        self._norm_std = 1.0 if norm_std is None else norm_std
        self._train_stimuli = set(train_stimuli) if train_stimuli is not None else None

    def _normalize_stimulus(self, stimulus: np.ndarray) -> np.ndarray:
        # Using pytorch is faster than numpy
        stimulus_tensor = torch.tensor(stimulus, dtype=torch.float32)
        normalized_tensor = (stimulus_tensor - self._norm_mean) / self._norm_std
        return normalized_tensor.detach().numpy()

    def process(
        self, stimulus_names: Iterable[str], session_specific_stimuli: dict[str, np.ndarray]
    ) -> MoviesTrainTestSplit:
        # Todo: maybe add caching in the future to reduce memory consumption
        train_stimulus_array, test_stimulus_dict = [], {}

        for name in stimulus_names:
            stimulus = session_specific_stimuli.get(name, self._name_to_stimulus[name])
            normalized_stimulus = self._normalize_stimulus(stimulus)
            if name in self._test_stimuli:
                test_stimulus_dict[name] = normalized_stimulus
            else:
                train_stimulus_array.append(normalized_stimulus)
        # concatenate stimuli over time dimension
        train_stimuli = np.concatenate(train_stimulus_array, axis=1)

        split = MoviesTrainTestSplit(
            train_stimuli,
            test_stimulus_dict,
            norm_mean=self._norm_mean,
            norm_std=self._norm_std,
        )
        return split


def _check_stimulus_size(
    stimulus: Float[np.ndarray, "channels time height width"],
    stimulus_size: tuple[int, int, int],
    stimulus_name: str,
) -> bool:
    """Expecting Float[np.ndarray, "channels time height width"] with height, width == stimulus_size[-2:]
    and channels == stimulus_size[0]
    """
    if len(stimulus.shape) != 4:
        raise ValueError(
            f"Expected four dimensional stimulus (channels, time, height, width), "
            f" but stimulus {stimulus_name} has {stimulus.shape=}"
        )
    elif (stimulus.shape[0] != stimulus_size[0]) or (stimulus.shape[-2:] != stimulus_size[-2:]):
        raise ValueError(
            f"The spatial dimensions of stimulus {stimulus_name} did not match the "
            f"provided stimulus size: {stimulus.shape=}, provided_size: {stimulus_size=}"
        )
    return True


def load_stimuli(
    base_data_path: str,
    test_names: Iterable[str],
    stimulus_size: list[int],
    norm_mean: float | None,
    norm_std: float | None,
) -> dict[str, MoviesTrainTestSplit]:
    if not os.path.isdir(base_data_path):
        raise ValueError(f"{base_data_path=} is not a directory.")
    if len(stimulus_size) != 3:
        raise ValueError(f"Stimulus size should contain the channels, height and width, but was {stimulus_size=}")
    stimulus_size_tuple = (stimulus_size[0], stimulus_size[1], stimulus_size[2])

    test_stimuli_names = sorted(test_names)

    name_to_stimulus = {}
    # first load all stimuli from stimuli folder
    stimulus_folder = os.path.join(base_data_path, _STIMULUS_FOLDER)
    if os.path.isdir(stimulus_folder):
        for file_name in [x for x in os.listdir(stimulus_folder) if x.endswith(".npy")]:
            name = file_name.removesuffix(".npy")
            stim_path = os.path.join(stimulus_folder, file_name)
            stim = np.load(stim_path)
            _check_stimulus_size(stim, stimulus_size_tuple, stim_path)
            name_to_stimulus[name] = stim
    else:
        LOGGER.warning(f"Did not find {stimulus_folder=}")

    result = {}
    stimulus_processor = TrainTestStimuliProcessor(test_stimuli_names, name_to_stimulus, norm_mean, norm_std)
    # build MoviesTrainTestSplit for each session
    for file_name in [x for x in os.listdir(base_data_path) if x.endswith(".h5") or x.endswith(".hdf5")]:
        h5_path = os.path.join(base_data_path, file_name)
        with h5py.File(h5_path, "r") as f:
            stimuli_with_responses = sorted(
                x.removeprefix(_RESPONSES_PREFIX) for x in f.keys() if x.startswith(_RESPONSES_PREFIX)
            )
            if len(stimuli_with_responses) == 0:
                LOGGER.warning(f"No responses found for {file_name}: {list(f.keys())}")
                continue

            session_specific_stimuli = {
                x.removeprefix(_SESSION_SPECIFIC_STIMULUS_PREFIX): f[x]
                for x in f.keys()
                if (
                    x.startswith(_SESSION_SPECIFIC_STIMULUS_PREFIX)
                    and _check_stimulus_size(f[x], stimulus_size_tuple, f"{h5_path}/{x}")
                )
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
            stimuli_with_responses = sorted(x for x in f.keys() if x.startswith(_RESPONSES_PREFIX))
            train_responses = np.concatenate(
                [f[x] for x in stimuli_with_responses if x.removeprefix(_RESPONSES_PREFIX) not in test_names_set],
                axis=-1,
            )
            test_responses_dict = {
                x.removeprefix(_RESPONSES_PREFIX): np.array(f[x])
                for x in stimuli_with_responses
                if x.removeprefix(_RESPONSES_PREFIX) in test_names_set
            }

            test_responses_neurons = {x.shape[0] for x in test_responses_dict.values()}
            if len(test_responses_neurons | {train_responses.shape[0]}) > 1:
                raise ValueError(
                    f"Train responses and test responses have a different number of neurons: "
                    f"{train_responses.shape[0]=} {test_responses_neurons=}"
                )
            # potentially improve: not sure if that fails for strings or other datatypes
            session_kwargs = {k: np.array(v) for k, v in f.get(_SESSION_INFO_KEY, {}).items()}

        session_name = Path(file_name).stem
        result[session_name] = ResponsesTrainTestSplit(
            train_responses, test_responses_dict, session_kwargs=session_kwargs
        )
    return result
