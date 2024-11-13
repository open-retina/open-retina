"""
Minimal data loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1016/j.neuron.2023.06.007
Data: https://doi.org/10.25740/rk663dm5577
"""

import os
from typing import Any, List, Optional

import numpy as np
import torch
from jaxtyping import Float

from ..utils.h5_handling import load_dataset_from_h5


class NeuronDataBaccus:
    def __init__(
        self,
        responses_final: Float[np.ndarray, " n_neurons n_timepoints"] | dict,
        val_clip_idx: List[int],
        num_clips: int,
        clip_length: int,
        key: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to store neuron data. Added for backwards compatibility with Hoefling et al., 2022.

        Args:
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]) or
                dictionary with train and test responses of similar structure: The responses of neurons.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
        """
        self.neural_responses = responses_final

        self.num_neurons = (
            self.neural_responses["train"].shape[0]
            if isinstance(self.neural_responses, dict)
            else self.neural_responses.shape[0]
        )

        self.key = key
        self.roi_coords = ()
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.val_clip_idx = val_clip_idx

    # this has to become a regular method in the future!
    @property
    def response_dict(self):
        # Transpose the responses to have the shape (n_timepoints, n_neurons)
        self.responses_test = self.neural_responses["test"].T
        self.responses_train_and_val = self.neural_responses["train"].T
        self.test_responses_by_trial = []

        self.compute_validation_responses()

        return {
            "train": torch.tensor(self.responses_train).to(torch.float),
            "validation": torch.tensor(self.responses_val).to(torch.float),
            "test": {
                "avg": torch.tensor(self.responses_test).to(torch.float),
                "by_trial": torch.tensor(self.test_responses_by_trial),
            },
        }

    def compute_validation_responses(self) -> None:
        movie_ordering = np.arange(self.num_clips)

        # Initialise validation responses
        base_movie_sorting = np.argsort(movie_ordering)

        validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)
        self.responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

        # Compute validation responses and remove sections from training responses

        for i, ind1 in enumerate(self.val_clip_idx):
            grab_index = base_movie_sorting[ind1]
            self.responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                :,
            ]
            validation_mask[
                (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                :,
            ] = False

        self.responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)


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

                stimuli_all_sessions["".join(session.split("/")[-1])] = {
                    "train": train_video,
                    "test": test_video,
                }

                train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")
                test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")

                responses_all_sessions["".join(session.split("/")[-1])] = {
                    "responses_final": {
                        "train": train_session_data[:, : train_video.shape[1]] / fr_normalization,
                        "test": test_session_data[:, : test_video.shape[1]] / fr_normalization,
                    },
                    "stim_id": "salamander_natural",
                }

    return responses_all_sessions, stimuli_all_sessions
