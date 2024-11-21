"""
Minimal responses loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1016/j.neuron.2023.06.007
Data: https://doi.org/10.25740/rk663dm5577
"""

from typing import List, Optional

import numpy as np
import torch
from jaxtyping import Float


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
        Initialize the NeuronDataBaccus object.
        Boilerplate class to store neuron data. Added for backwards compatibility with Hoefling et al., 2022.

        Args:
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]) or
                dictionary with train and test responses of similar structure: The responses of neurons.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
            clip_length (int): The length of each clip.
            key (dict, optional): Additional key information.
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

        # Transpose the responses to have the shape (n_timepoints, n_neurons)
        self.responses_test = self.neural_responses["test"].T
        self.responses_train_and_val = self.neural_responses["train"].T

        self.responses_train, self.responses_val = self.compute_validation_responses()
        self.test_responses_by_trial = np.array([])  # Added for compatibility with Hoefling et al., 2024

    def compute_validation_responses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute validation responses and updated train responses stripped from validation clips.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated train and validation responses.
        """
        movie_ordering = np.arange(self.num_clips)

        # Initialise validation responses
        base_movie_sorting = np.argsort(movie_ordering)

        validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)
        responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

        # Compute validation responses and remove sections from training responses

        for i, ind1 in enumerate(self.val_clip_idx):
            grab_index = base_movie_sorting[ind1]
            responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                :,
            ]
            validation_mask[
                (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                :,
            ] = False

        responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)
        return responses_train, responses_val

    @property
    def response_dict(self):
        """
        Create and return a dictionary of neural responses for train, validation, and test datasets.
        """
        return {
            "train": torch.tensor(self.responses_train, dtype=torch.float),
            "validation": torch.tensor(self.responses_val, dtype=torch.float),
            "test": {
                "avg": torch.tensor(self.responses_test, dtype=torch.float),
                "by_trial": torch.tensor(self.test_responses_by_trial, dtype=torch.float),
            },
        }