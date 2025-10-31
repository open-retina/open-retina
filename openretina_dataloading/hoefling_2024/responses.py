import warnings
from copy import deepcopy
from typing import Literal, Optional, no_type_check

import numpy as np
from jaxtyping import Float
from tqdm.auto import tqdm

from ..base import ResponsesTrainTestSplit
from .constants import CLIP_LENGTH, NUM_CLIPS, STIMULI_IDS


class NeuronDataSplitHoefling:
    def __init__(
        self,
        neural_responses: ResponsesTrainTestSplit,
        val_clip_idx: Optional[list[int]],
        num_clips: Optional[int],
        clip_length: Optional[int],
        roi_mask: Optional[Float[np.ndarray, "64 64"]] = None,
        roi_ids: Optional[Float[np.ndarray, " n_neurons"]] = None,
        scan_sequence_idx: Optional[int] = None,
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,
        eye: Optional[Literal["left", "right"]] = None,
        group_assignment: Optional[Float[np.ndarray, " n_neurons"]] = None,
        key: Optional[dict] = None,
        use_base_sequence: Optional[bool] = False,
        **kwargs,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to store neuron data train/test/validation splits before feeding into a dataloader.
        Customized for compatibility with the data format in Hoefling et al., 2024.

        Args:
            eye (str): The eye from which the neuron data is recorded.
            group_assignment (Float[np.ndarray, "n_neurons"]): The group assignment of neurons.
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses_final (Float[np.ndarray, "n_neurons n_timepoints"]): The responses of neurons.
            roi_coords (Float[np.ndarray, "n_neurons 2"]): The coordinates of regions of interest (ROIs).
            roi_ids (Float[np.ndarray, "n_neurons"]): The IDs of regions of interest (ROIs).
            scan_sequence_idx (int): The index of the scan sequence.
            stim_id (int): The ID of the stimulus. 5 is mouse natural scenes.
            random_sequences (Float[np.ndarray, "n_clips n_sequences"]): The random sequences of clips.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
            clip_length (int): The length of each clip.
            use_base_sequence (bool): Whether to re-order all training responses to use the same "base" sequence.
        """
        self.neural_responses = neural_responses
        self.val_clip_idx = val_clip_idx
        self.num_clips = num_clips if num_clips is not None else NUM_CLIPS
        self.clip_length = clip_length if clip_length is not None else CLIP_LENGTH
        self.roi_mask = roi_mask
        self.roi_ids = roi_ids
        self.scan_sequence_idx = scan_sequence_idx
        self.random_sequences = random_sequences
        self.eye = eye
        self.group_assignment = group_assignment
        self.key = key
        self.use_base_sequence = use_base_sequence
        self.stim_id = STIMULI_IDS[neural_responses.stim_id] if neural_responses.stim_id is not None else None

        self.responses_test = self.neural_responses.test_response
        self.test_responses_by_trial = (
            self.neural_responses.test_by_trial
            if self.neural_responses.test_by_trial is not None
            else None,
        )
        self.responses_train, self.responses_val = self.compute_validation_responses()

    @property
    def response_dict(self):
        """Return response dictionary with numpy arrays instead of torch tensors"""
        return {
            "train": self.responses_train.astype(np.float32),
            "validation": self.responses_val.astype(np.float32),
            "test": {
                "avg": self.responses_test.astype(np.float32),
                "by_trial": self.test_responses_by_trial,
            },
        }

    @no_type_check
    def compute_validation_responses(self) -> tuple[np.ndarray, np.ndarray]:
        if self.stim_id in ["mb", "chirp"]:
            # Chirp and moving bar
            responses_val = np.array(np.nan)
            responses_train = self.neural_responses.train
        else:
            # Natural scenes
            if self.val_clip_idx is None:
                raise ValueError("val_clip_idx must be provided for natural scenes")

            # Compute validation responses
            val_start_idx = np.array(self.val_clip_idx) * self.clip_length
            val_end_idx = val_start_idx + self.clip_length
            val_indices = np.concatenate([np.arange(start, end) for start, end in zip(val_start_idx, val_end_idx)])
            responses_val = self.neural_responses.train[:, val_indices]

            # Compute training responses (excluding validation clips)
            all_indices = np.arange(self.neural_responses.train.shape[1])
            train_indices = np.setdiff1d(all_indices, val_indices)
            responses_train = self.neural_responses.train[:, train_indices]

            if self.use_base_sequence:
                responses_train = self.reorder_responses_to_base_sequence(responses_train)

        return responses_train, responses_val

    def reorder_responses_to_base_sequence(self, responses_train: np.ndarray) -> np.ndarray:
        """Reorder training responses to use the same base sequence"""
        if self.random_sequences is None:
            raise ValueError("random_sequences must be provided to reorder responses")

        # Get the base sequence (first sequence)
        base_sequence = self.random_sequences[0, :]
        
        # Reorder responses for each clip
        reordered_responses = []
        for clip_idx in range(self.num_clips):
            if clip_idx in (self.val_clip_idx or []):
                continue  # Skip validation clips
                
            clip_start = clip_idx * self.clip_length
            clip_end = clip_start + self.clip_length
            clip_responses = responses_train[:, clip_start:clip_end]
            
            # Get the sequence for this clip
            clip_sequence = self.random_sequences[clip_idx, :]
            
            # Create mapping from clip sequence to base sequence
            reorder_indices = np.argsort(np.argsort(base_sequence))[np.argsort(clip_sequence)]
            
            # Reorder the responses
            reordered_clip = clip_responses[:, reorder_indices]
            reordered_responses.append(reordered_clip)
        
        return np.concatenate(reordered_responses, axis=1)


def load_hoefling_responses(
    data_path: str,
    stim_id: str,
    val_clip_idx: Optional[list[int]] = None,
    num_clips: Optional[int] = None,
    clip_length: Optional[int] = None,
    use_base_sequence: bool = False,
    **kwargs,
) -> dict[str, NeuronDataSplitHoefling]:
    """
    Load Hoefling 2024 responses data.
    
    Args:
        data_path: Path to the data directory
        stim_id: Stimulus ID
        val_clip_idx: Validation clip indices
        num_clips: Number of clips
        clip_length: Length of each clip
        use_base_sequence: Whether to reorder responses to base sequence
        **kwargs: Additional arguments
        
    Returns:
        Dictionary mapping session names to NeuronDataSplitHoefling objects
    """
    # This would be implemented based on the specific data loading logic
    # for the Hoefling 2024 dataset
    raise NotImplementedError("load_hoefling_responses needs to be implemented based on specific data format")