"""
Stimulus loader for Palmer et al. 2024 salamander recordings.

Dataset layout (HDF5):
    /samplingfreq -> scalar (Hz)
    /test/stimulus -> uint8 [n_movies, time, height, width]
    /test/movie_names -> object/bytes [n_movies]

This dataset only contains frozen test movies. We allow selecting a subset of
movies for training and use the remaining movies for testing.
"""

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path

DEFAULT_MOVIE_NAMES = [
    "MultipleMoviesStim_1_tree.avi",
    "MultipleMoviesStim_2_water.avi",
    "MultipleMoviesStim_3_grasses.avi",
    "MultipleMoviesStim_4_fish.avi",
    "MultipleMoviesStim_5_opticflow.avi",
]


def _resolve_h5_path(base_data_path: str | Path) -> Path:
    """
    Resolve the provided path (local or huggingface URL) to the unique H5 file.
    """
    resolved = Path(get_local_file_path(str(base_data_path)))
    if resolved.is_file():
        return resolved

    h5_files = list(resolved.glob("*.h5"))
    if len(h5_files) != 1:
        raise ValueError(f"Expected a single .h5 file in {resolved}, found {len(h5_files)}.")
    return h5_files[0]


def _decode_movie_names(raw_names: np.ndarray) -> list[str]:
    decoded = []
    for n in raw_names:
        if isinstance(n, (bytes, np.bytes_)):
            decoded.append(n.decode())
        else:
            decoded.append(str(n))
    return decoded


def _sort_movies_by_canonical_order(
    movies: Sequence[str], canonical_order: Sequence[str] = DEFAULT_MOVIE_NAMES
) -> list[str]:
    """
    Sort a list of movie names according to the canonical order.

    Raises ValueError if any movie is not in the dataset movie names list.
    """
    canonical_set = set(canonical_order)
    movies_set = set(movies)

    non_canonical = movies_set - canonical_set
    if non_canonical:
        raise ValueError(
            f"Some requested movies are not in the dataset: {non_canonical}. Supported movies: {canonical_order}"
        )

    return [m for m in canonical_order if m in movies_set]


def _indices_for_movies(all_names: Sequence[str], requested: Iterable[str], field: str) -> list[int]:
    """
    Get indices of requested movies in all_names, sorted by canonical order for consistency.
    """
    requested_list = list(requested)
    # Sort by canonical order to ensure consistent concatenation
    sorted_requested = _sort_movies_by_canonical_order(requested_list)
    missing = set(sorted_requested) - set(all_names)
    if missing:
        raise ValueError(f"Requested {field} movies not found in file: {sorted(missing)}. Available: {all_names}.")
    return [all_names.index(n) for n in sorted_requested]


def _normalize(train: np.ndarray, test_dict: dict[str, np.ndarray], normalize: bool):
    if not normalize:
        return train, test_dict, {"norm_mean": None, "norm_std": None}

    train_tensor = train.astype(np.float32)
    mean = float(train_tensor.mean())
    std = float(train_tensor.std())
    if std == 0:
        std = 1.0

    train_norm = (train_tensor - mean) / std
    test_norm = {k: (v.astype(np.float32) - mean) / std for k, v in test_dict.items()}
    return train_norm, test_norm, {"norm_mean": mean, "norm_std": std}


def _resize_movies(
    train: np.ndarray,
    test_dict: dict[str, np.ndarray],
    resize_hw: tuple[int, int] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Resize train and test movies to (height, width) using bilinear interpolation.

    Expects input shape [channels, time, height, width] and resizes spatial dimensions.
    """
    if resize_hw is None:
        return train, test_dict

    def _resize(movie: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(movie, dtype=torch.float32)  # [C, T, H, W]
        # F.interpolate expects [batch, channels, height, width]
        tensor = rearrange(tensor, "c t h w -> t c h w", c=1)  # [T, C, H, W] - treat time as batch
        resized = F.interpolate(tensor, size=resize_hw, mode="bilinear", align_corners=False)  # [T, C, H_new, W_new]
        resized = rearrange(resized, "t c h w -> c t h w", c=1)  # [C, T, H_new, W_new]
        return resized.cpu().numpy()

    train_resized = _resize(train)
    test_resized = {k: _resize(v) for k, v in test_dict.items()}
    return train_resized, test_resized


def load_stimuli(
    base_data_path: str | Path,
    train_movies: Sequence[str] | None = None,
    test_movies: Sequence[str] | None = None,
    *,
    normalize_stimuli: bool = True,
    resize_hw: tuple[int, int] | None = None,
    session_id: str = "palmer_2024_salamander",
) -> dict[str, MoviesTrainTestSplit]:
    """
    Load stimuli for Palmer 2024 and construct a train/test split by movie name.

    Args:
        base_data_path: Local path or huggingface URL pointing to the folder or the H5 file.
        train_movies: Movie names to concatenate for training (order preserved). Defaults to first three movies.
        test_movies: Movie names to keep as frozen test stimuli. Defaults to remaining movies.
        normalize_stimuli: Whether to z-score using the training movie statistics.
        resize_hw: Optional (height, width) to resize both train and test movies before building splits.
        session_id: Key used for the returned dictionary.
    """
    h5_path = _resolve_h5_path(base_data_path)

    with h5py.File(h5_path, "r") as f:
        movie_data = np.asarray(f["test/stimulus"], dtype=np.float32)  # [movies, time, H, W]
        raw_names = np.asarray(f["test/movie_names"])
        movie_names = _decode_movie_names(raw_names)
        # Use time array if available to determine correct length, otherwise use movie_data shape
        stim_time = np.asarray(f["test/time"]) if "test/time" in f else None
        if stim_time is not None:
            time_bins = int(stim_time.shape[0])
        else:
            time_bins = movie_data.shape[1]
        # Clip to ensure we don't exceed available data
        time_bins = min(time_bins, movie_data.shape[1])

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

    train_movie = np.concatenate([movie_data[i, :time_bins] for i in train_idx], axis=0)  # [time, H, W]
    # Use sorted test_movies for keys to ensure consistency with responses
    # test_idx is already sorted by canonical order, matching test_movies order
    test_dict = {name: movie_data[idx, :time_bins] for name, idx in zip(test_movies, test_idx)}

    # Add channel dimension
    train_movie = train_movie[None, ...]
    test_dict = {k: v[None, ...] for k, v in test_dict.items()}

    train_movie, test_dict, norm_dict = _normalize(train_movie, test_dict, normalize_stimuli)
    train_movie, test_dict = _resize_movies(train_movie, test_dict, resize_hw)

    return {
        session_id: MoviesTrainTestSplit(
            train=train_movie,
            test_dict=test_dict,
            stim_id="palmer_2024",
            random_sequences=None,
            norm_mean=norm_dict["norm_mean"],
            norm_std=norm_dict["norm_std"],
        )
    }
