import os

import numpy as np

from ..base import MoviesTrainTestSplit
from .constants import CLIP_LENGTH, NUM_CLIPS


def movies_from_pickle(file_path: str | os.PathLike) -> MoviesTrainTestSplit:
    """
    Load movie data from a pickle file and return it as a MoviesTrainTestSplit object.
    """
    return MoviesTrainTestSplit.from_pickle(file_path)


def generate_movie_splits(
    movie_train: np.ndarray,
    movie_test_dict: dict[str, np.ndarray],
    validation_clip_indices: list[int],
    num_clips: int,
    clip_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate train, validation, and test movie splits.
    
    Args:
        movie_train: Training movie array
        movie_test_dict: Dictionary of test movies
        validation_clip_indices: Indices of validation clips
        num_clips: Number of clips
        clip_length: Length of each clip
        
    Returns:
        Tuple of (train_subset, validation, test_dict)
    """
    # Create validation movie from specified clips
    val_indices = []
    for clip_idx in validation_clip_indices:
        start_idx = clip_idx * clip_length
        end_idx = start_idx + clip_length
        val_indices.extend(range(start_idx, end_idx))
    
    movie_val = movie_train[:, val_indices, ...]
    
    # Create training subset (excluding validation clips)
    train_indices = []
    for clip_idx in range(num_clips):
        if clip_idx not in validation_clip_indices:
            start_idx = clip_idx * clip_length
            end_idx = start_idx + clip_length
            train_indices.extend(range(start_idx, end_idx))
    
    movie_train_subset = movie_train[:, train_indices, ...]
    
    return movie_train_subset, movie_val, movie_test_dict


# Helper function to assemble the datasets
def apply_random_sequences(
    movie_train: np.ndarray,
    movie_train_subset: np.ndarray,
    movie_val: np.ndarray,
    movie_test: np.ndarray,
    random_sequences: np.ndarray,
    val_clip_idx: list[int],
    clip_length: int,
):
    """Apply random sequences to create left/right eye movies with numpy operations."""
    movies = {
        "left": {
            "train": {},
            "validation": np.flip(movie_val, axis=-1),  # Flip horizontally for left eye
            "test": np.flip(movie_test, axis=-1),
        },
        "right": {"train": {}, "validation": movie_val, "test": movie_test},
        "val_clip_idx": val_clip_idx,
    }

    for sequence_index in range(random_sequences.shape[1]):
        reordered_movie = np.zeros_like(movie_train_subset)
        # explicitly cast to int, as random_sequences is a np.uint8 which is prone to overflow.
        clip_indices = [int(idx) for idx in random_sequences[:, sequence_index] if idx not in val_clip_idx]
        for k, clip_idx in enumerate(clip_indices):
            reordered_movie[:, k * clip_length : (k + 1) * clip_length, ...] = movie_train[
                :, clip_idx * clip_length : (clip_idx + 1) * clip_length, ...
            ]
        movies["right"]["train"][sequence_index] = reordered_movie  # type: ignore
        movies["left"]["train"][sequence_index] = np.flip(reordered_movie, axis=-1)  # type: ignore

    return movies


def load_hoefling_movies(
    movie_train: np.ndarray,
    movie_test: np.ndarray,
    random_sequences: np.ndarray,
    validation_clip_indices: list[int],
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
) -> dict:
    """
    Load and process Hoefling 2024 movie data.
    
    Args:
        movie_train: Training movie data
        movie_test: Test movie data  
        random_sequences: Random sequence indices for clip ordering
        validation_clip_indices: Indices of clips to use for validation
        num_clips: Number of clips (default from constants)
        clip_length: Length of each clip (default from constants)
        
    Returns:
        Dictionary containing processed movies for left/right eyes
    """

    # Generate train, validation, and test datasets
    movie_train_subset, movie_val, movie_test_dict = generate_movie_splits(
        movie_train, {"test": movie_test}, validation_clip_indices, num_clips, clip_length
    )

    # Assemble datasets into the final movies structure using the random sequences
    movies = apply_random_sequences(
        movie_train.astype(np.float32),
        movie_train_subset,
        movie_val,
        movie_test_dict["test"],
        random_sequences,
        validation_clip_indices,
        clip_length,
    )

    return movies