import os
import pickle
from typing import Any, Optional

import numpy as np

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path


def average_repeated_stimuli_responses(repeated_responses: np.ndarray) -> np.ndarray:
    # shape num_of_cells x number of repeated images x number of repetitions of repeated stimuli
    return np.mean(repeated_responses, axis=-1)


def load_responses(
    base_path: str | os.PathLike,
    files: dict[str, str],
    stimulus_seed: int = 0,
    excluded_cells: Optional[dict[Any, list[int]]] = None,
    cell_index: Optional[int] = None,
) -> dict[str, dict[str, np.ndarray]]:
    base_path = get_local_file_path(str(base_path))
    responses = {}

    for session_id, file in files.items():
        with open(os.path.join(base_path, file), "rb") as pkl:
            neural_data = pickle.load(pkl)

        test_responses_raw = neural_data["test_responses"]
        train_responses = neural_data["train_responses"]
        if cell_index is not None:
            train_responses = train_responses[cell_index : cell_index + 1, :, :]
            test_responses_raw = test_responses_raw[cell_index : cell_index + 1, :, :]
        elif excluded_cells is not None:
            train_responses = np.delete(train_responses, excluded_cells[session_id], axis=0)
            test_responses_raw = np.delete(test_responses_raw, excluded_cells[session_id], axis=0)

        if "seeds" in neural_data.keys():
            seed_info: list[int] = neural_data["seeds"]
            if stimulus_seed in seed_info:
                trials_assigned_to_seed: list[int] = neural_data["trial_separation"][stimulus_seed]
                train_responses = train_responses[:, :, trials_assigned_to_seed]
                test_responses_raw = test_responses_raw[:, :, trials_assigned_to_seed]
            elif session_id == "01":
                # only uses the first ten trials (seed 2022) for evaluation in retina '01',
                # the responses do not match between seeds 2022 and 2023
                test_responses_raw = test_responses_raw[:, :, :10]

        # Preserve per-trial traces for evaluation while still exposing the averaged responses
        test_responses_by_trial: np.ndarray | None
        if test_responses_raw.ndim == 3:
            test_responses_by_trial = test_responses_raw
            test_responses = average_repeated_stimuli_responses(test_responses_raw)
        else:
            test_responses_by_trial = None
            test_responses = test_responses_raw

        responses[session_id] = {
            "train_responses": train_responses,
            "test_responses": test_responses,
            "test_responses_by_trial": test_responses_by_trial,
        }
    return responses


def response_splits_from_pickles(
    base_path: str | os.PathLike,
    files: dict[str, str],
    stimulus_seed: int = 0,
    excluded_cells: Optional[dict[Any, list[int]]] = None,
    cell_index: Optional[int] = None,
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Convert Sridhar pickled responses into ``ResponsesTrainTestSplit`` objects compatible with the unified pipeline.
    The output will not be used directly by the dataloader and model, but rather for `data_info` computation.
    """
    raw_responses = load_responses(
        base_path,
        files=files,
        stimulus_seed=stimulus_seed,
        excluded_cells=excluded_cells,
        cell_index=cell_index,
    )

    splits: dict[str, ResponsesTrainTestSplit] = {}
    for session_id, tensors in raw_responses.items():
        train_responses = np.asarray(tensors["train_responses"], dtype=np.float32)
        test_responses = np.asarray(tensors["test_responses"], dtype=np.float32)
        test_by_trial = tensors.get("test_responses_by_trial")

        n_neurons, frames_per_trial, n_trials = train_responses.shape
        train_matrix = train_responses.reshape(n_neurons, frames_per_trial * n_trials)

        test_by_trial_formatted = None
        if test_by_trial is not None and test_by_trial.ndim == 3:
            # Expecting shape (neurons, time, trials); re-order to (trials, neurons, time)
            test_by_trial_formatted = np.transpose(test_by_trial, (2, 0, 1))
        if test_responses.ndim == 3:
            test_responses = np.mean(test_responses, axis=-1)

        splits[session_id] = ResponsesTrainTestSplit(
            train=train_matrix,
            test=test_responses,
            test_by_trial=test_by_trial_formatted,
            stim_id=f"sridhar_{session_id}",
            session_kwargs={
                "stimulus_seed": stimulus_seed,
                "frames_per_trial": frames_per_trial,
                "num_trials": n_trials,
            },
        )
    return splits
