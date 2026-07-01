"""qiu_2026 dataloaders that carry ``pupil_center`` alongside inputs/targets.

open-retina's shared ``DataPoint`` is a 2-field ``(inputs, targets)`` namedtuple; extending it would break
every existing dataset (see the integration plan §0.4). Instead this module defines a qiu-local
``QiuDataPoint`` and a ``QiuMovieDataSet`` that slices a per-frame pupil trace with the same chunking as the
movie/responses, injected into :func:`get_movie_dataloader` via its ``dataset_cls``/``extra_arrays`` hook.

``qiu_2026_dataloaders`` mirrors :func:`multiple_movies_dataloaders` but (a) splits the pupil trace
identically to the movie via :func:`generate_movie_splits`, (b) honors each session's *curated* validation
split (the ``validation_clip_indices`` stashed in ``session_kwargs``) so validation trials are carved out of
training rather than chosen at random, and (c) evaluates each test condition as one whole clip.
"""

import collections
from collections import namedtuple
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import (
    MovieDataSet,
    NeuronDataSplit,
    _compute_test_batch_size,
    generate_movie_splits,
    get_movie_dataloader,
)

# qiu-local batch object: three real tensors, so PyTorch default_collate handles it unchanged.
QiuDataPoint = namedtuple("QiuDataPoint", ["inputs", "targets", "pupil_center"])


class QiuMovieDataSet(MovieDataSet):
    """``MovieDataSet`` that additionally carries a per-frame ``pupil_center`` ``(2, T)`` trace.

    ``pupil_center`` is sliced on its time axis (axis 1) with the exact chunk the movie (axis 1) and
    responses (axis 0) are sliced with, and returned as the third field of a :class:`QiuDataPoint`.
    """

    def __init__(
        self,
        movies,
        responses,
        roi_ids,
        roi_coords,
        group_assignment,
        split,
        chunk_size,
        pupil_center,
    ):
        super().__init__(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        self.pupil_center = torch.as_tensor(pupil_center, dtype=torch.float)  # (2, T)

    def __getitem__(self, idx) -> QiuDataPoint:  # type: ignore[override]
        if isinstance(idx, slice):
            return QiuDataPoint(self.samples[0][:, idx, ...], self.samples[1][idx, ...], self.pupil_center[:, idx])
        start = int(idx)
        chunk = slice(start, start + self.chunk_size)
        return QiuDataPoint(
            self.samples[0][:, chunk, ...],
            self.samples[1][chunk, ...],
            self.pupil_center[:, chunk],
        )


def _split_pupil(
    pupil_train: np.ndarray,
    pupil_test_dict: dict[str, np.ndarray],
    val_clip_idx: list[int],
    num_clips: int,
    clip_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Split a ``(2, T)`` pupil trace exactly like the movie, by shaping it as a ``(2, T, 1, 1)`` fake movie."""
    fake_train = pupil_train[:, :, None, None]
    fake_test = {name: trace[:, :, None, None] for name, trace in pupil_test_dict.items()}
    train_subset, val, test = generate_movie_splits(
        fake_train, fake_test, val_clip_idc=val_clip_idx, num_clips=num_clips, clip_length=clip_length
    )
    return train_subset[:, :, 0, 0], val[:, :, 0, 0], {name: t[:, :, 0, 0] for name, t in test.items()}


def qiu_2026_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, Any],
    pupil_dictionary: dict[str, dict[str, Any]],
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    clip_length: int = 300,
    num_val_clips: int = 10,
    val_clip_indices: list[int] | None = None,
    allow_over_boundaries: bool = False,
) -> dict[str, dict[str, DataLoader]]:
    """Build ``dict[split][session] -> DataLoader`` yielding :class:`QiuDataPoint` batches.

    Validation-clip selection per session, in priority order: the explicit ``val_clip_indices`` argument, then
    the session's curated ``session_kwargs["validation_clip_indices"]``, then a random draw of
    ``num_val_clips``. The curated path is the intended one — it removes the dataset's designated validation
    trials from the training subset. Each test condition is evaluated as a single whole clip.
    """
    assert set(neuron_data_dictionary) == set(movies_dictionary) == set(pupil_dictionary), (
        "neuron_data_dictionary, movies_dictionary and pupil_dictionary must share the same session keys."
    )

    dataloaders: dict[str, Any] = collections.defaultdict(dict)
    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating qiu movie dataloaders"):
        num_clips = movies_dictionary[session_key].train.shape[1] // clip_length

        if val_clip_indices is not None:
            val_clip_idx = list(val_clip_indices)
        elif "validation_clip_indices" in session_data.session_kwargs:
            val_clip_idx = list(session_data.session_kwargs["validation_clip_indices"])
        else:
            rnd = np.random.RandomState(seed)
            val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

        movie_train_subset, movie_val, movie_test_dict = generate_movie_splits(
            movies_dictionary[session_key].train,
            movies_dictionary[session_key].test_dict,
            val_clip_idc=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )
        pupil_train_subset, pupil_val, pupil_test_dict = _split_pupil(
            pupil_dictionary[session_key]["train"],
            pupil_dictionary[session_key]["test_dict"],
            val_clip_idx,
            num_clips,
            clip_length,
        )
        neuron_data = NeuronDataSplit(
            responses=session_data, val_clip_idx=val_clip_idx, num_clips=num_clips, clip_length=clip_length
        )

        for fold, movie, pupil, chunk_size in [
            ("train", movie_train_subset, pupil_train_subset, train_chunk_size),
            ("validation", movie_val, pupil_val, clip_length),
        ]:
            dataloaders[fold][session_key] = get_movie_dataloader(
                movie=movie,
                responses=neuron_data.response_dict[fold],
                split=fold,
                chunk_size=chunk_size,
                batch_size=batch_size,
                scene_length=clip_length,
                allow_over_boundaries=allow_over_boundaries,
                dataset_cls=QiuMovieDataSet,
                extra_arrays={"pupil_center": pupil},
            )

        # Each test condition is a single whole clip (300 or 450 frames): evaluate it in one chunk.
        for name, movie in movie_test_dict.items():
            test_chunk_size = movie.shape[1]
            test_batch_size = _compute_test_batch_size(batch_size, train_chunk_size, test_chunk_size)
            dataloaders[name][session_key] = get_movie_dataloader(
                movie=movie,
                responses=neuron_data.response_dict_test[name],
                split="test",
                chunk_size=test_chunk_size,
                batch_size=test_batch_size,
                scene_length=clip_length,
                allow_over_boundaries=True,
                dataset_cls=QiuMovieDataSet,
                extra_arrays={"pupil_center": pupil_test_dict[name]},
            )

    return dataloaders
