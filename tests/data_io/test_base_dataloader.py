"""Tests for ``openretina.data_io.base_dataloader``."""

import numpy as np
import torch

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import (
    MovieDataSet,
    NeuronDataSplit,
    generate_movie_splits,
    get_movie_dataloader,
)

CLIP_LENGTH = 10
NUM_CLIPS = 5
VAL_CLIP_IDC = [1, 3]
N_CHANNELS, HEIGHT, WIDTH = 2, 4, 6
N_NEURONS = 3


def _make_movie() -> np.ndarray:
    """A movie whose every frame is filled with its own frame index, so clip identity is checkable."""
    frames = np.arange(CLIP_LENGTH * NUM_CLIPS, dtype=np.float32)
    return np.broadcast_to(frames[None, :, None, None], (N_CHANNELS, CLIP_LENGTH * NUM_CLIPS, HEIGHT, WIDTH)).copy()


def test_generate_movie_splits_returns_expected_clips() -> None:
    movie = _make_movie()
    test_dict = {"cond1": np.ones((N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH), dtype=np.float32)}

    train_subset, val, test_tensors = generate_movie_splits(
        movie, test_dict, val_clip_idc=VAL_CLIP_IDC, num_clips=NUM_CLIPS, clip_length=CLIP_LENGTH
    )

    # Validation holds exactly the requested clips, in the requested order.
    expected_val_frames = [idx * CLIP_LENGTH + offset for idx in VAL_CLIP_IDC for offset in range(CLIP_LENGTH)]
    assert val.shape[1] == len(VAL_CLIP_IDC) * CLIP_LENGTH
    torch.testing.assert_close(val[0, :, 0, 0], torch.tensor(expected_val_frames, dtype=torch.float))

    # The train subset holds the complement, so together they cover the movie with no overlap.
    expected_train_frames = [
        idx * CLIP_LENGTH + offset
        for idx in range(NUM_CLIPS)
        if idx not in VAL_CLIP_IDC
        for offset in range(CLIP_LENGTH)
    ]
    torch.testing.assert_close(train_subset[0, :, 0, 0], torch.tensor(expected_train_frames, dtype=torch.float))
    assert train_subset.shape[1] + val.shape[1] == CLIP_LENGTH * NUM_CLIPS
    assert set(test_tensors) == {"cond1"}


def test_generate_movie_splits_outputs_do_not_alias_input() -> None:
    """The returned tensors must own their storage, as callers may free the source movie afterwards."""
    movie = _make_movie()
    test_array = np.ones((N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH), dtype=np.float32)
    original = movie.copy()

    train_subset, val, test_tensors = generate_movie_splits(
        movie, {"cond1": test_array}, val_clip_idc=VAL_CLIP_IDC, num_clips=NUM_CLIPS, clip_length=CLIP_LENGTH
    )

    for name, tensor, source in (
        ("train_subset", train_subset, movie),
        ("val", val, movie),
        ("test", test_tensors["cond1"], test_array),
    ):
        assert tensor.data_ptr() != source.__array_interface__["data"][0], f"{name} aliases its source array"

    # Mutating the outputs must not write back into the caller's arrays.
    train_subset.fill_(-1.0)
    val.fill_(-1.0)
    test_tensors["cond1"].fill_(-1.0)
    np.testing.assert_array_equal(movie, original)
    np.testing.assert_array_equal(test_array, np.ones_like(test_array))


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


def _make_responses() -> np.ndarray:
    """Responses aligned with ``_make_movie``: one column per neuron, frame index as the value."""
    frames = np.arange(CLIP_LENGTH * NUM_CLIPS, dtype=np.float32)
    return np.repeat(frames[:, None], N_NEURONS, axis=1)


class _MovieDataSetWithExtra(MovieDataSet):
    """Minimal subclass of the kind the ``dataset_cls`` hook exists for: one extra per-frame array."""

    def __init__(self, *args, extra: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = extra


def test_get_movie_dataloader_defaults_to_a_plain_movie_dataset() -> None:
    """The hook must be invisible to every existing caller."""
    loader = get_movie_dataloader(
        movie=torch.as_tensor(_make_movie()),
        responses=torch.as_tensor(_make_responses()),
        split="train",
        start_indices=[0, CLIP_LENGTH],
        scene_length=CLIP_LENGTH,
        chunk_size=CLIP_LENGTH,
        batch_size=1,
        drop_last=False,
    )

    assert type(loader.dataset) is MovieDataSet


def test_get_movie_dataloader_injects_the_dataset_class_and_extra_arrays() -> None:
    extra = np.arange(CLIP_LENGTH * NUM_CLIPS, dtype=np.float32)

    loader = get_movie_dataloader(
        movie=torch.as_tensor(_make_movie()),
        responses=torch.as_tensor(_make_responses()),
        split="train",
        start_indices=[0, CLIP_LENGTH],
        scene_length=CLIP_LENGTH,
        chunk_size=CLIP_LENGTH,
        batch_size=1,
        drop_last=False,
        dataset_cls=_MovieDataSetWithExtra,
        extra_arrays={"extra": extra},
    )

    assert isinstance(loader.dataset, _MovieDataSetWithExtra)
    np.testing.assert_array_equal(loader.dataset.extra, extra)
