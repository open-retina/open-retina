"""
Custom dataloaders for Palmer et al. 2024 salamander recordings.

Provides repeat-aware dataloading that leverages per-trial responses without
duplicating the stimulus movies in memory.
"""

import collections
from collections import namedtuple
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import (
    NeuronDataSplit,
    generate_movie_splits,
    get_movie_dataloader,
)
from openretina.data_io.palmer_2024.responses import _load_repeats
from openretina.data_io.palmer_2024.stimuli import _decode_movie_names, _resolve_h5_path

DataPoint = namedtuple("DataPoint", ["inputs", "targets"])


class PalmerAveragedDataset(Dataset):
    """Strided Palmer movie chunks paired with valid-repeat averaged responses."""

    def __init__(
        self,
        movie: torch.Tensor,
        averaged_responses: torch.Tensor,
        movie_boundaries: list[int],
        chunk_size: int,
        chunk_stride: int | None = None,
        excluded_movie_indices: list[int] | None = None,
    ):
        self.movie = movie
        self.averaged_responses = averaged_responses
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_size if chunk_stride is None else chunk_stride
        self.excluded_movie_indices = set(excluded_movie_indices or [])

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.chunk_stride <= 0:
            raise ValueError(f"chunk_stride must be positive, got {self.chunk_stride}.")
        if movie.shape[1] != averaged_responses.shape[0]:
            raise ValueError(
                f"Palmer movie and averaged response lengths differ: {movie.shape[1]} != {averaged_responses.shape[0]}."
            )

        self.indices: list[tuple[int, int]] = []
        included_movies = []
        included_responses = []
        for movie_idx, (start_time, end_time) in enumerate(zip(movie_boundaries[:-1], movie_boundaries[1:])):
            movie_len = end_time - start_time
            if movie_len < self.chunk_size:
                raise ValueError(
                    f"Palmer movie {movie_idx} has {movie_len} frames, shorter than chunk_size={self.chunk_size}."
                )
            if movie_idx in self.excluded_movie_indices:
                continue

            included_movies.append(movie[:, start_time:end_time])
            included_responses.append(averaged_responses[start_time:end_time])
            for chunk_start in range(start_time, end_time - self.chunk_size + 1, self.chunk_stride):
                self.indices.append((chunk_start, chunk_start + self.chunk_size))

        if not self.indices:
            raise ValueError("PalmerAveragedDataset contains no training chunks after applying the validation split.")

        self._movies = torch.cat(included_movies, dim=1)
        self._responses = torch.cat(included_responses, dim=0)
        self.mean_response = self._responses.mean(dim=0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> DataPoint:
        chunk_start, chunk_end = self.indices[idx]
        return DataPoint(
            inputs=self.movie[:, chunk_start:chunk_end],
            targets=self.averaged_responses[chunk_start:chunk_end],
        )

    @property
    def movies(self) -> torch.Tensor:
        return self._movies

    @property
    def responses(self) -> torch.Tensor:
        return self._responses


class PalmerRepeatDataset(Dataset):
    """
    Dataset for Palmer 2024 that indexes over (trial, chunk) pairs without duplicating stimuli.

    Each movie was shown to the retina multiple times (repeats), and we have trial-wise responses.
    This dataset returns chunks from the **single** movie array paired with responses from
    individual trials, without duplicating the stimulus for every trial.

    Args:
        movie: Training movie, shape [channels, time_total, height, width].
        train_repeats: Per-trial responses for training movies, shape [n_trials, n_neurons, time_total].
        movie_boundaries: Time boundaries for each training movie (cumulative), e.g. [0, 1200, 2400, ...].
        n_reps_per_movie: Number of valid (non-padded) repeats for each training movie.
        chunk_size: Number of frames per sample.
        chunk_stride: Distance between adjacent chunks. Defaults to ``chunk_size``.
        split: "train" or "validation".
    """

    def __init__(
        self,
        movie: torch.Tensor,
        train_repeats: torch.Tensor,
        movie_boundaries: list[int],
        n_reps_per_movie: list[int],
        chunk_size: int,
        chunk_stride: int | None = None,
        split: str = "train",
        excluded_movie_indices: list[int] | None = None,
    ):
        self.movie = movie  # [C, T_total, H, W]
        self.train_repeats = train_repeats  # [n_trials, n_neurons, T_total]
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_size if chunk_stride is None else chunk_stride
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.chunk_stride <= 0:
            raise ValueError(f"chunk_stride must be positive, got {self.chunk_stride}.")
        self.split = split
        self.movie_boundaries = movie_boundaries
        self.n_reps_per_movie = n_reps_per_movie
        self.excluded_movie_indices = set(excluded_movie_indices or [])

        # Build a flat list of (trial_idx, chunk_start, chunk_end) tuples
        self.indices: list[tuple[int, int, int]] = []
        n_movies = len(movie_boundaries) - 1
        trial_offset = 0
        included_movies = []
        averaged_responses = []
        response_sum = torch.zeros(train_repeats.shape[1], dtype=train_repeats.dtype)
        response_count = 0

        for movie_idx in range(n_movies):
            start_time = movie_boundaries[movie_idx]
            end_time = movie_boundaries[movie_idx + 1]
            movie_len = end_time - start_time
            if movie_len < self.chunk_size:
                raise ValueError(
                    f"Palmer movie {movie_idx} has {movie_len} frames, shorter than chunk_size={self.chunk_size}."
                )
            chunk_starts = range(start_time, end_time - self.chunk_size + 1, self.chunk_stride)

            n_valid_repeats = n_reps_per_movie[movie_idx]
            movie_responses = self.train_repeats[trial_offset : trial_offset + n_valid_repeats, :, start_time:end_time]
            if movie_idx not in self.excluded_movie_indices:
                included_movies.append(self.movie[:, start_time:end_time])
                averaged_responses.append(movie_responses.mean(dim=0).T)
                response_sum += movie_responses.sum(dim=(0, 2))
                response_count += n_valid_repeats * movie_len

                for local_trial in range(n_valid_repeats):
                    trial_idx = trial_offset + local_trial
                    for chunk_start in chunk_starts:
                        chunk_end = chunk_start + self.chunk_size
                        self.indices.append((trial_idx, chunk_start, chunk_end))

            trial_offset += n_valid_repeats

        if not self.indices:
            raise ValueError("PalmerRepeatDataset contains no training chunks after applying the validation split.")

        self._movies = torch.cat(included_movies, dim=1)
        self._responses = torch.cat(averaged_responses, dim=0)
        self.mean_response = response_sum / response_count

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> DataPoint:
        trial_idx, chunk_start, chunk_end = self.indices[idx]
        # Movie chunk: [C, chunk_size, H, W]
        movie_chunk = self.movie[:, chunk_start:chunk_end, :, :]
        # Response chunk: [chunk_size, n_neurons]
        resp_chunk = self.train_repeats[trial_idx, :, chunk_start:chunk_end].T
        return DataPoint(inputs=movie_chunk, targets=resp_chunk)

    @property
    def movies(self) -> torch.Tensor:
        """Return only training movies, excluding held-out validation movies."""
        return self._movies

    @property
    def responses(self) -> torch.Tensor:
        """Return valid-repeat averages for training movies (shape [T, n_neurons])."""
        return self._responses


def _load_train_repeats_with_variable_reps(
    h5_path: Path,
    movie_names: list[str],
    train_movies: list[str],
    movie_boundaries: list[int],
    response_key: Literal["binned", "firing_rate_60ms"],
    fr_normalization: float,
    exclude_initial_frames: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Load per-trial responses for training movies, accounting for variable repeats per movie.

    Returns:
        train_repeats: shape [total_valid_trials, n_neurons, time_total] - full concatenated timeline
        n_reps_per_movie: list of actual (non-padded) repeat counts per training movie
        trial_to_movie: list mapping trial_idx -> movie_idx
    """
    with h5py.File(h5_path, "r") as f:
        movie_lengths = np.diff(movie_boundaries)
        if len(set(movie_lengths.tolist())) != 1:
            raise ValueError(f"Palmer movies must have equal lengths, got {movie_lengths.tolist()}.")
        movie_len = int(movie_lengths[0])
        repeats_by_movie = _load_repeats(
            f,
            movie_names,
            train_movies,
            frame_start=exclude_initial_frames,
            frame_end=exclude_initial_frames + movie_len,
            response_key=response_key,
        )
        if not repeats_by_movie:
            raise ValueError("No per-trial responses were found in /test/repeats.")

        # Build full timeline for each trial
        time_total = movie_boundaries[-1]
        n_neurons = next(iter(repeats_by_movie.values())).shape[1]
        all_trials = []
        n_reps_per_movie = []
        trial_to_movie = []

        for movie_idx, movie_name in enumerate(train_movies):
            movie_repeats = repeats_by_movie[movie_name]
            n_valid_reps = movie_repeats.shape[0]
            n_reps_per_movie.append(n_valid_reps)
            movie_start = movie_boundaries[movie_idx]
            movie_end = movie_boundaries[movie_idx + 1]

            # For each trial of this movie, create full timeline
            for trial_in_movie in range(n_valid_reps):
                trial_to_movie.append(movie_idx)
                # Create full timeline: [neurons, time_total]
                full_timeline = np.zeros((n_neurons, time_total), dtype=np.float32)
                full_timeline[:, movie_start:movie_end] = movie_repeats[trial_in_movie]
                all_trials.append(full_timeline)

        train_repeats = np.stack(all_trials, axis=0) / fr_normalization

    return train_repeats, n_reps_per_movie, trial_to_movie


def repeats_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, MoviesTrainTestSplit],
    train_chunk_size: int = 60,
    train_chunk_stride: int | None = None,
    average_repeats: bool = False,
    batch_size: int = 8,
    seed: int = 42,
    clip_length: int = 1170,
    num_val_clips: int = 1,
    val_clip_indices: list[int] | None = None,
    base_data_path: str | Path | None = None,
    response_key: Literal["binned", "firing_rate_60ms"] = "firing_rate_60ms",
    fr_normalization: float = 1.0,
    exclude_initial_frames: int = 30,
) -> dict[str, dict[str, DataLoader]]:
    """
    Create dataloaders for Palmer 2024 that leverage per-trial repeats without duplicating stimuli.

    For training: uses repeat-averaged responses by default when ``average_repeats`` is enabled, otherwise indexes
    over individual (trial, chunk) pairs with ``PalmerRepeatDataset``.
    For validation and test: uses the standard multiple_movies_dataloaders approach.

    Args:
        neuron_data_dictionary: Session responses (must contain train_movies info in session_kwargs).
        movies_dictionary: Session movies.
        train_chunk_size: Chunk size for training samples.
        train_chunk_stride: Distance between adjacent training chunks. Defaults to ``train_chunk_size``. For the
            Palmer core-readout model, a stride of 30 pairs overlapping 60-frame inputs with the 30-frame model
            output, covering every predictable response frame exactly once per repeat.
        average_repeats: Train on the valid-repeat averaged response instead of individual repeat targets.
        batch_size: Batch size for dataloaders.
        seed: Random seed for validation split.
        clip_length: Length of each movie clip (used for validation splits).
        num_val_clips: Number of clips to reserve for validation.
        val_clip_indices: Optional explicit validation clip indices.
        base_data_path: Path to HDF5 file. Required only when loading individual repeat targets.
        response_key: Which response type to use from HDF5.
        fr_normalization: Scalar to divide firing rates.
        exclude_initial_frames: Number of bins already removed from the start of every loaded movie and response.

    Returns:
        Nested dict: {"train": {session: loader}, "validation": {session: loader}, test_names: ...}
    """
    assert set(neuron_data_dictionary.keys()) == set(movies_dictionary.keys()), (
        "neuron_data_dictionary and movies_dictionary keys must match."
    )

    h5_path = None
    if not average_repeats:
        if base_data_path is None:
            raise ValueError("base_data_path is required to load per-trial repeats from HDF5.")
        h5_path = _resolve_h5_path(base_data_path)

    dataloaders: dict[str, Any] = collections.defaultdict(dict)

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating Palmer dataloaders"):
        # Extract train_movies from session_kwargs
        train_movies = session_data.session_kwargs.get("train_movies")
        if train_movies is None:
            raise ValueError(f"session_kwargs must contain 'train_movies' for session {session_key}.")
        loaded_response_key = session_data.session_kwargs.get("response_key")
        if loaded_response_key is not None and loaded_response_key != response_key:
            raise ValueError(
                f"Repeat response_key {response_key!r} does not match loaded responses {loaded_response_key!r}."
            )
        loaded_excluded_frames = session_data.session_kwargs.get("exclude_initial_frames")
        if loaded_excluded_frames is not None and loaded_excluded_frames != exclude_initial_frames:
            raise ValueError(
                f"Repeat exclude_initial_frames={exclude_initial_frames} does not match loaded responses "
                f"({loaded_excluded_frames})."
            )
        loaded_normalization = session_data.session_kwargs.get("fr_normalization")
        if loaded_normalization is not None and loaded_normalization != fr_normalization:
            raise ValueError(
                f"Repeat fr_normalization={fr_normalization} does not match loaded responses ({loaded_normalization})."
            )

        # Load movie and compute boundaries
        movie_split = movies_dictionary[session_key]
        movie_train = torch.tensor(movie_split.train, dtype=torch.float32)  # [C, T_total, H, W]
        time_total = movie_train.shape[1]
        if time_total % len(train_movies) != 0:
            raise ValueError(f"Training time {time_total} is not divisible by {len(train_movies)} Palmer movies.")
        time_per_movie = time_total // len(train_movies)
        if clip_length != time_per_movie:
            raise ValueError(
                f"clip_length must match the post-exclusion Palmer movie length ({time_per_movie}), got {clip_length}."
            )

        movie_boundaries = [i * time_per_movie for i in range(len(train_movies) + 1)]

        num_clips = len(train_movies)
        if val_clip_indices is not None:
            val_clip_idx = list(val_clip_indices)
        else:
            rnd = np.random.RandomState(seed)
            val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))
        if len(set(val_clip_idx)) != len(val_clip_idx) or any(idx < 0 or idx >= num_clips for idx in val_clip_idx):
            raise ValueError(f"Invalid Palmer validation movie indices: {val_clip_idx}.")

        if average_repeats:
            train_dataset = PalmerAveragedDataset(
                movie=movie_train,
                averaged_responses=torch.tensor(session_data.train.T, dtype=torch.float32),
                movie_boundaries=movie_boundaries,
                chunk_size=train_chunk_size,
                chunk_stride=train_chunk_stride,
                excluded_movie_indices=val_clip_idx,
            )
        else:
            assert h5_path is not None
            # Load per-trial repeats with variable reps handling
            with h5py.File(h5_path, "r") as f:
                movie_names = _decode_movie_names(np.asarray(f["test/movie_names"]))

            train_repeats_np, n_reps_per_movie, trial_to_movie = _load_train_repeats_with_variable_reps(
                h5_path=h5_path,
                movie_names=movie_names,
                train_movies=train_movies,
                movie_boundaries=movie_boundaries,
                response_key=response_key,
                fr_normalization=fr_normalization,
                exclude_initial_frames=exclude_initial_frames,
            )

            train_repeats = torch.tensor(train_repeats_np, dtype=torch.float32)  # [trials, neurons, time]
            train_n_reps_per_movie = [trial_to_movie.count(m) for m in range(len(train_movies))]
            train_dataset = PalmerRepeatDataset(
                movie=movie_train,
                train_repeats=train_repeats,
                movie_boundaries=movie_boundaries,
                n_reps_per_movie=train_n_reps_per_movie,
                chunk_size=train_chunk_size,
                chunk_stride=train_chunk_stride,
                split="train",
                excluded_movie_indices=val_clip_idx,
            )

        dataloaders["train"][session_key] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Use NeuronDataSplit to get averaged train/val responses
        neuron_data = NeuronDataSplit(
            responses=session_data,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # Generate validation movie splits
        _, movie_val, _ = generate_movie_splits(
            movie_split.train,
            {},  # no test movies needed here
            val_clip_idc=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # Create validation dataloader with averaged responses
        dataloaders["validation"][session_key] = get_movie_dataloader(
            movie=movie_val,
            responses=neuron_data.response_dict["validation"],
            split="validation",
            chunk_size=clip_length,
            batch_size=batch_size,
            scene_length=clip_length,
            allow_over_boundaries=False,
        )

        # For test, use standard dataloader with averaged responses
        neuron_data = NeuronDataSplit(
            responses=session_data,
            val_clip_idx=[],  # no validation split needed for test
            num_clips=1,
            clip_length=time_total,
        )

        for name, test_movie in movie_split.test_dict.items():
            test_movie_tensor = torch.tensor(test_movie, dtype=torch.float32)
            dataloaders[name][session_key] = get_movie_dataloader(
                movie=test_movie_tensor,
                responses=neuron_data.response_dict_test[name],
                split="test",
                chunk_size=test_movie_tensor.shape[1],
                batch_size=batch_size,
                scene_length=clip_length,
                allow_over_boundaries=False,
            )

    return dataloaders
