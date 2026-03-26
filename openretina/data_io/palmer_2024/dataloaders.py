"""
Custom dataloaders for Palmer et al. 2024 salamander recordings.

Provides repeat-aware dataloading that leverages per-trial responses without
duplicating the stimulus movies in memory.
"""

import collections
from collections import namedtuple
from pathlib import Path
from typing import Any

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
from openretina.data_io.palmer_2024.stimuli import _decode_movie_names, _indices_for_movies, _resolve_h5_path

DataPoint = namedtuple("DataPoint", ["inputs", "targets"])


class PalmerRepeatDataset(Dataset):
    """
    Dataset for Palmer 2024 that indexes over (trial, chunk) pairs without duplicating stimuli.

    Each movie was shown to the retina multiple times (repeats), and we have trial-wise responses.
    This dataset returns chunks from the **single** movie array paired with responses from
    individual trials, so memory scales with `n_trials` (metadata) rather than `n_trials × T` (data).

    Args:
        movie: Training movie, shape [channels, time_total, height, width].
        train_repeats: Per-trial responses for training movies, shape [n_trials, n_neurons, time_total].
        movie_boundaries: Time boundaries for each training movie (cumulative), e.g. [0, 1200, 2400, ...].
        n_reps_per_movie: Number of valid (non-padded) repeats for each training movie.
        chunk_size: Number of frames per sample.
        split: "train" or "validation".
    """

    def __init__(
        self,
        movie: torch.Tensor,
        train_repeats: torch.Tensor,
        movie_boundaries: list[int],
        n_reps_per_movie: list[int],
        chunk_size: int,
        split: str = "train",
    ):
        self.movie = movie  # [C, T_total, H, W]
        self.train_repeats = train_repeats  # [n_trials, n_neurons, T_total]
        self.chunk_size = chunk_size
        self.split = split
        self.movie_boundaries = movie_boundaries
        self.n_reps_per_movie = n_reps_per_movie

        # Build a flat list of (trial_idx, chunk_start, chunk_end) tuples
        self.indices: list[tuple[int, int, int]] = []
        n_movies = len(movie_boundaries) - 1
        trial_offset = 0

        for movie_idx in range(n_movies):
            start_time = movie_boundaries[movie_idx]
            end_time = movie_boundaries[movie_idx + 1]
            movie_len = end_time - start_time
            n_chunks_per_trial = movie_len // chunk_size

            n_valid_repeats = n_reps_per_movie[movie_idx]
            for local_trial in range(n_valid_repeats):
                trial_idx = trial_offset + local_trial
                for chunk_idx in range(n_chunks_per_trial):
                    chunk_start = start_time + chunk_idx * chunk_size
                    chunk_end = chunk_start + chunk_size
                    self.indices.append((trial_idx, chunk_start, chunk_end))

            trial_offset += n_valid_repeats

        # Compute mean response (used by model for bias init)
        self.mean_response = self.train_repeats.mean(dim=(0, 2))  # [n_neurons]

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
        """Return the full movie for compatibility with eval (shape [C, T, H, W])."""
        return self.movie

    @property
    def responses(self) -> torch.Tensor:
        """Return averaged responses over trials (shape [T, n_neurons]) for eval."""
        return self.train_repeats.mean(dim=0).T


def _load_train_repeats_with_variable_reps(
    h5_path: Path,
    movie_names: list[str],
    train_movies: list[str],
    movie_boundaries: list[int],
    fr_normalization: float,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Load per-trial responses for training movies, accounting for variable repeats per movie.

    Returns:
        train_repeats: shape [total_valid_trials, n_neurons, time_total] - full concatenated timeline
        n_reps_per_movie: list of actual (non-padded) repeat counts per training movie
        trial_to_movie: list mapping trial_idx -> movie_idx
    """
    with h5py.File(h5_path, "r") as f:
        # Get number of repeats per movie from /test/nreps
        nreps_array = np.asarray(f["test/nreps"][...], dtype=int)  # shape [n_movies] = 5

        repeats_group = f["test"]["repeats"]
        cell_keys = sorted([k for k in repeats_group.keys() if "cell" in k], key=lambda x: int(x.split("cell")[-1]))
        if len(cell_keys) == 0:
            raise ValueError("No 'cell*' entries found in test/repeats.")

        # Load per-cell repeats: shape [neurons, movies, repeats_padded, time]
        # First get clip_len from the first cell
        sample_cell = np.asarray(repeats_group[cell_keys[0]][...], dtype=np.float32)
        clip_len = sample_cell.shape[2]
        per_cell = [np.asarray(repeats_group[k][...], dtype=np.float32)[:, :, :clip_len] for k in cell_keys]
        per_cell_stack = np.stack(per_cell, axis=0)  # [neurons, movies, repeats_padded, time]

        # Get indices for training movies
        train_idx = _indices_for_movies(movie_names, train_movies, "train")

        # Build full timeline for each trial
        time_total = movie_boundaries[-1]
        n_neurons = per_cell_stack.shape[0]
        all_trials = []
        n_reps_per_movie = []
        trial_to_movie = []

        for movie_idx, file_idx in enumerate(train_idx):
            n_valid_reps = int(nreps_array[file_idx])
            n_reps_per_movie.append(n_valid_reps)
            movie_start = movie_boundaries[movie_idx]
            movie_end = movie_boundaries[movie_idx + 1]
            movie_len = movie_end - movie_start

            # Extract: neurons x valid_repeats x time
            movie_repeats = per_cell_stack[:, file_idx, :n_valid_reps, :]  # [neurons, n_valid_reps, time]
            # Transpose to: valid_repeats x neurons x time
            movie_repeats = np.transpose(movie_repeats, (1, 0, 2))  # [n_valid_reps, neurons, time]

            # For each trial of this movie, create full timeline
            for trial_in_movie in range(n_valid_reps):
                trial_to_movie.append(movie_idx)
                # Create full timeline: [neurons, time_total]
                full_timeline = np.zeros((n_neurons, time_total), dtype=np.float32)
                # Clip repeats to match movie segment length (handles slight mismatches like 1203 vs 1200)
                full_timeline[:, movie_start:movie_end] = movie_repeats[trial_in_movie][:, :movie_len]
                all_trials.append(full_timeline)

        train_repeats = np.stack(all_trials, axis=0) / fr_normalization  # [total_trials, neurons, time_total]
        train_repeats = np.transpose(train_repeats, (0, 1, 2))  # Keep as [trials, neurons, time]

    return train_repeats, n_reps_per_movie, trial_to_movie


def repeats_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, MoviesTrainTestSplit],
    train_chunk_size: int = 60,
    batch_size: int = 8,
    seed: int = 42,
    clip_length: int = 1200,
    num_val_clips: int = 1,
    val_clip_indices: list[int] | None = None,
    base_data_path: str | Path | None = None,
    fr_normalization: float = 1.0,
) -> dict[str, dict[str, DataLoader]]:
    """
    Create dataloaders for Palmer 2024 that leverage per-trial repeats without duplicating stimuli.

    For training: uses PalmerRepeatDataset to index over (trial, chunk) pairs.
    For validation and test: uses the standard multiple_movies_dataloaders approach.

    Args:
        neuron_data_dictionary: Session responses (must contain train_movies info in session_kwargs).
        movies_dictionary: Session movies.
        train_chunk_size: Chunk size for training samples.
        batch_size: Batch size for dataloaders.
        seed: Random seed for validation split.
        clip_length: Length of each movie clip (used for validation splits).
        num_val_clips: Number of clips to reserve for validation.
        val_clip_indices: Optional explicit validation clip indices.
        base_data_path: Path to HDF5 file (required to load per-trial repeats).
        response_key: Which response type to use from HDF5.
        fr_normalization: Scalar to divide firing rates.

    Returns:
        Nested dict: {"train": {session: loader}, "validation": {session: loader}, test_names: ...}
    """
    assert set(neuron_data_dictionary.keys()) == set(movies_dictionary.keys()), (
        "neuron_data_dictionary and movies_dictionary keys must match."
    )

    if base_data_path is None:
        raise ValueError("base_data_path is required to load per-trial repeats from HDF5.")

    h5_path = _resolve_h5_path(base_data_path)

    dataloaders: dict[str, Any] = collections.defaultdict(dict)

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating Palmer repeat dataloaders"):
        # Extract train_movies from session_kwargs
        train_movies = session_data.session_kwargs.get("train_movies")
        if train_movies is None:
            raise ValueError(f"session_kwargs must contain 'train_movies' for session {session_key}.")

        # Load movie and compute boundaries
        movie_split = movies_dictionary[session_key]
        movie_train = torch.tensor(movie_split.train, dtype=torch.float32)  # [C, T_total, H, W]
        time_total = movie_train.shape[1]
        time_per_movie = time_total // len(train_movies)

        movie_boundaries = [i * time_per_movie for i in range(len(train_movies) + 1)]

        # Load per-trial repeats with variable reps handling
        with h5py.File(h5_path, "r") as f:
            movie_names = _decode_movie_names(np.asarray(f["test/movie_names"]))

        train_repeats_np, n_reps_per_movie, trial_to_movie = _load_train_repeats_with_variable_reps(
            h5_path=h5_path,
            movie_names=movie_names,
            train_movies=train_movies,
            movie_boundaries=movie_boundaries,
            fr_normalization=fr_normalization,
        )

        train_repeats = torch.tensor(train_repeats_np, dtype=torch.float32)  # [n_trials, n_neurons, T_total]

        # Compute n_reps_per_movie for training (all trials)
        train_n_reps_per_movie = [trial_to_movie.count(m) for m in range(len(train_movies))]

        # Create training dataset with per-trial repeats
        train_dataset = PalmerRepeatDataset(
            movie=movie_train,
            train_repeats=train_repeats,
            movie_boundaries=movie_boundaries,
            n_reps_per_movie=train_n_reps_per_movie,
            chunk_size=train_chunk_size,
            split="train",
        )

        dataloaders["train"][session_key] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # For validation, use averaged responses (standard approach)
        # Compute validation clip indices from the training movie
        num_clips = time_total // clip_length
        if val_clip_indices is not None:
            val_clip_idx = val_clip_indices
        else:
            rnd = np.random.RandomState(seed)
            val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

        # Use NeuronDataSplit to get averaged train/val responses
        neuron_data = NeuronDataSplit(
            responses=session_data,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # Generate validation movie splits
        movie_train_subset, movie_val, _ = generate_movie_splits(
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
