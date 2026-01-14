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


def _load_repeats(
    h5_file: h5py.File, movie_names: Sequence[str], test_movies: Sequence[str], clip_len: int
) -> dict[str, np.ndarray]:
    """
    Load per-trial repeats if available.

    Returns:
        Dict mapping movie name -> repeats x neurons x time array.
    """
    repeats_group = h5_file["test"].get("repeats")
    if repeats_group is None:
        return {}

    cell_keys = sorted([k for k in repeats_group.keys() if "cell" in k], key=lambda x: int(x.split("cell")[-1]))
    if len(cell_keys) == 0:
        return {}

    per_cell = [np.asarray(repeats_group[k][...], dtype=np.float32)[:, :, :clip_len] for k in cell_keys]
    # shape: neurons x movies x repeats x time
    per_cell_stack = np.stack(per_cell, axis=0)

    test_by_trial: dict[str, np.ndarray] = {}
    for name in test_movies:
        movie_idx = movie_names.index(name)
        data = per_cell_stack[:, movie_idx, :, :]  # neurons x repeats x time
        data = np.transpose(data, (1, 0, 2))  # repeats x neurons x time
        test_by_trial[name] = data
    return test_by_trial


def load_responses(
    base_data_path: str | Path,
    train_movies: Sequence[str] | None = None,
    test_movies: Sequence[str] | None = None,
    *,
    response_key: Literal["binned", "firing_rate_60ms"] = "firing_rate_60ms",
    fr_normalization: float = 1.0,
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
        session_id: Key used for the returned dictionary.
    """
    h5_path = _resolve_h5_path(base_data_path)

    with h5py.File(h5_path, "r") as f:
        raw_names = np.asarray(f["test/movie_names"])
        movie_names = _decode_movie_names(raw_names)

        responses = np.asarray(f[f"test/response/{response_key}"], dtype=np.float32)  # neurons x movies x time
        stim_time = np.asarray(f["test/time"]) if "test/time" in f else None
        # Determine clip length based on time array or minimum of stimulus/response lengths
        if stim_time is not None:
            stim_len = int(stim_time.shape[0])
        else:
            stim_len = responses.shape[2]
        # Clip to the minimum to ensure alignment with stimuli
        clip_len = min(stim_len, responses.shape[2])
        responses = responses[:, :, :clip_len]

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

        train_resp = np.concatenate([responses[:, i, :] for i in train_idx], axis=1) / fr_normalization
        # Use sorted test_movies for keys to ensure consistency with stimuli
        # test_idx is already sorted by canonical order, matching test_movies order
        test_dict = {name: responses[:, idx, :] / fr_normalization for name, idx in zip(test_movies, test_idx)}

        test_by_trial_dict = _load_repeats(f, movie_names, test_movies, clip_len)
        if fr_normalization != 1.0 and len(test_by_trial_dict) > 0:
            test_by_trial_dict = {k: v / fr_normalization for k, v in test_by_trial_dict.items()}

    return {
        session_id: ResponsesTrainTestSplit(
            train=train_resp,
            test_dict=test_dict,
            test_by_trial_dict=test_by_trial_dict,
            stim_id="palmer_2024",
            session_kwargs={"train_movies": list(train_movies), "test_movies": list(test_movies)},
        )
    }
