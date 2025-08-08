import os
import pickle
from typing import Any, Optional

import numpy as np


def average_repeated_stimuli_responses(repeated_responses: np.ndarray):
    # shape num_of_cells x number of repeated images x number of repetitions of repeated stimuli
    return np.mean(repeated_responses, axis=-1)


def load_responses(base_path, files, stimulus_seed=0,
                   excluded_cells: Optional[dict[Any, list[int]]] = None,
                   cell_index: Optional[int] = None):
    responses = {}

    for session_id, file in files.items():
        with open(os.path.join(base_path, file), "rb") as pkl:
            neural_data = pickle.load(pkl)

        test_responses = neural_data["test_responses"]
        train_responses = neural_data["train_responses"]
        if cell_index is not None:
            train_responses = train_responses[cell_index : cell_index + 1, :, :]
            test_responses = test_responses[cell_index : cell_index + 1, :, :]
        elif excluded_cells is not None:
            train_responses = np.delete(train_responses, excluded_cells[session_id], axis=0)
            test_responses = np.delete(test_responses, excluded_cells[session_id], axis=0)

        if "seeds" in neural_data.keys():
            seed_info = neural_data["seeds"]
            if stimulus_seed in seed_info:
                trials_assigned_to_seed = neural_data["trial_separation"][stimulus_seed]
                train_responses = train_responses[:, :, trials_assigned_to_seed]
                test_responses = test_responses[:, :, trials_assigned_to_seed]
            elif session_id == "01":
                # only uses the first ten trials (seed 2022) for evaluation in retina '01',
                # the responses do not match between seeds 2022 and 2023
                test_responses = test_responses[:, :, :10]

            test_responses = average_repeated_stimuli_responses(test_responses)
        responses[session_id] = {"train_responses": train_responses, "test_responses": test_responses}
    return responses
