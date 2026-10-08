"""
Response loader for Palmer et al. 2024 salamander recordings.

The dataset only contains responses to five fixed movies. We construct a
train/test split by selecting movie names, concatenating training movies in
time, and keeping held-out movies as separate test entries.
"""

from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.palmer_2024.stimuli import (
    _decode_movie_names,
    _indices_for_movies,
    _resolve_h5_path,
    _sort_movies_by_canonical_order,
)

FIRING_RATE_WINDOW_SECONDS = 0.06


def _load_repeats(
    h5_file: h5py.File,
    movie_names: Sequence[str],
    requested_movies: Sequence[str],
    frame_start: int,
    frame_end: int,
    response_key: Literal["binned", "firing_rate_60ms"],
) -> dict[str, np.ndarray]:
    """
    Load per-trial repeats if available.

    Returns:
        Dict mapping movie name -> valid_repeats x neurons x time array. Padded repeat slots are removed and
        responses are returned in the units requested by ``response_key``.
    """
    if response_key not in {"binned", "firing_rate_60ms"}:
        raise ValueError(f"Unsupported Palmer response_key: {response_key!r}.")

    repeats_group = h5_file["test"].get("repeats")
    if repeats_group is None:
        return {}

    cell_keys = sorted([k for k in repeats_group.keys() if "cell" in k], key=lambda x: int(x.split("cell")[-1]))
    if len(cell_keys) == 0:
        return {}

    if "nreps" not in h5_file["test"]:
        raise ValueError("Per-trial responses require /test/nreps to remove padded repeat slots.")
    n_repeats = np.asarray(h5_file["test/nreps"][...], dtype=int)

    per_cell = [np.asarray(repeats_group[k][...], dtype=np.float32) for k in cell_keys]
    # shape: neurons x movies x repeats x time
    per_cell_stack = np.stack(per_cell, axis=0)
    if frame_end > per_cell_stack.shape[-1]:
        raise ValueError(
            f"Requested response frames [{frame_start}, {frame_end}) exceed repeat length {per_cell_stack.shape[-1]}."
        )

    repeats_by_movie: dict[str, np.ndarray] = {}
    for name in requested_movies:
        movie_idx = movie_names.index(name)
        valid_repeats = int(n_repeats[movie_idx])
        data = per_cell_stack[:, movie_idx, :valid_repeats, frame_start:frame_end]
        data = np.transpose(data, (1, 0, 2))  # repeats x neurons x time
        if response_key == "firing_rate_60ms":
            data = data / FIRING_RATE_WINDOW_SECONDS
        repeats_by_movie[name] = data
    return repeats_by_movie


def load_responses(
    base_data_path: str | Path,
    train_movies: Sequence[str] | None = None,
    test_movies: Sequence[str] | None = None,
    *,
    response_key: Literal["binned", "firing_rate_60ms"] = "firing_rate_60ms",
    fr_normalization: float = 1.0,
    exclude_initial_frames: int = 30,
    session_id: str = "palmer_2024_salamander",
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Load responses for Palmer 2024 and build train/test splits by movie name.

    Args:
        base_data_path: Local path or huggingface URL pointing to the folder or the H5 file.
        train_movies: Movie names to concatenate for training (order preserved). Defaults to first three movies.
        test_movies: Movie names to keep for testing. Defaults to the remaining movies.
        response_key: Dataset name under /test/response (e.g., "binned" or "firing_rate_60ms").
        fr_normalization: Scalar to divide responses (e.g., to convert counts to rates).
        exclude_initial_frames: Number of response bins to remove from the beginning of every movie. The Palmer et
            al. analysis excludes the first 500 ms, corresponding to 30 bins at 60 Hz.
        session_id: Key used for the returned dictionary.
    """
    if fr_normalization <= 0:
        raise ValueError(f"fr_normalization must be positive, got {fr_normalization}.")

    h5_path = _resolve_h5_path(base_data_path)

    with h5py.File(h5_path, "r") as f:
        raw_names = np.asarray(f["test/movie_names"])
        movie_names = _decode_movie_names(raw_names)

        response_path = f"test/response/{response_key}"
        responses = np.asarray(f[response_path], dtype=np.float32) if response_path in f else None

        available_lengths = []
        if "test/time" in f:
            available_lengths.append(int(f["test/time"].shape[0]))
        if "test/stimulus" in f:
            available_lengths.append(int(f["test/stimulus"].shape[1]))
        if responses is not None:
            available_lengths.append(int(responses.shape[2]))
        repeats_group = f["test"].get("repeats")
        if repeats_group is not None:
            cell_keys = [key for key in repeats_group.keys() if "cell" in key]
            if cell_keys:
                available_lengths.append(int(repeats_group[cell_keys[0]].shape[2]))
        if not available_lengths:
            raise ValueError("Could not determine the Palmer response length from the HDF5 file.")
        frame_end = min(available_lengths)
        if not 0 <= exclude_initial_frames < frame_end:
            raise ValueError(f"exclude_initial_frames must be in [0, {frame_end}), got {exclude_initial_frames}.")

        if train_movies is None:
            train_movies = movie_names[:3]
        if test_movies is None:
            test_movies = [m for m in movie_names if m not in train_movies]

        # Convert to lists and sort by canonical order for consistent processing
        train_movies = _sort_movies_by_canonical_order(list(train_movies))
        test_movies = _sort_movies_by_canonical_order(list(test_movies))

        overlap = set(train_movies) & set(test_movies)
        if overlap:
            raise ValueError(f"Train and test movies overlap: {sorted(overlap)}.")
        if len(train_movies) == 0 or len(test_movies) == 0:
            raise ValueError("Both train_movies and test_movies must contain at least one movie.")

        train_idx = _indices_for_movies(movie_names, train_movies, "train")
        test_idx = _indices_for_movies(movie_names, test_movies, "test")

        requested_movies = list(dict.fromkeys([*train_movies, *test_movies]))
        repeats_by_movie = _load_repeats(
            f,
            movie_names,
            requested_movies,
            frame_start=exclude_initial_frames,
            frame_end=frame_end,
            response_key=response_key,
        )

        if repeats_by_movie:
            averaged_by_movie = {name: trials.mean(axis=0) for name, trials in repeats_by_movie.items()}
        elif responses is not None:
            averaged_by_movie = {
                name: responses[:, idx, exclude_initial_frames:frame_end]
                for name, idx in zip([*train_movies, *test_movies], [*train_idx, *test_idx])
            }
        else:
            raise ValueError(f"Neither {response_path} nor per-trial responses were found in the HDF5 file.")

        train_resp = np.concatenate([averaged_by_movie[name] for name in train_movies], axis=1) / fr_normalization
        test_dict = {name: averaged_by_movie[name] / fr_normalization for name in test_movies}
        test_by_trial_dict = (
            {name: repeats_by_movie[name] / fr_normalization for name in test_movies} if repeats_by_movie else {}
        )

    return {
        session_id: ResponsesTrainTestSplit(
            train=train_resp,
            test_dict=test_dict,
            test_by_trial_dict=test_by_trial_dict,
            stim_id="palmer_2024",
            session_kwargs={
                "train_movies": list(train_movies),
                "test_movies": list(test_movies),
                "response_key": response_key,
                "fr_normalization": fr_normalization,
                "exclude_initial_frames": exclude_initial_frames,
            },
        )
    }
