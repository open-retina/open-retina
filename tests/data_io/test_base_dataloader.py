"""Tests for the shared helpers in ``openretina.data_io.base_dataloader``."""

import numpy as np

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import NeuronDataSplit

CLIP_LENGTH = 10
NUM_CLIPS = 5
VAL_CLIP_IDC = [1, 3]
N_NEURONS = 3


def test_neuron_data_split_response_dicts_are_cached() -> None:
    """Each access rebuilds every tensor, and callers index one key per access inside a loop."""
    rng = np.random.default_rng(0)
    responses = ResponsesTrainTestSplit(
        train=rng.random((N_NEURONS, CLIP_LENGTH * NUM_CLIPS)).astype(np.float32),
        test_dict={
            "cond1": rng.random((N_NEURONS, CLIP_LENGTH)).astype(np.float32),
            "cond2": rng.random((N_NEURONS, CLIP_LENGTH)).astype(np.float32),
        },
        stim_id="synthetic",
    )
    neuron_data = NeuronDataSplit(
        responses=responses, val_clip_idx=VAL_CLIP_IDC, num_clips=NUM_CLIPS, clip_length=CLIP_LENGTH
    )

    assert neuron_data.response_dict is neuron_data.response_dict
    assert neuron_data.response_dict_test is neuron_data.response_dict_test
    # Same tensor object per key, not just an equal one -- that is what makes the loop O(N) not O(N^2).
    assert neuron_data.response_dict_test["cond1"]["avg"] is neuron_data.response_dict_test["cond1"]["avg"]
    assert neuron_data.response_dict["train"].shape[1] == N_NEURONS
