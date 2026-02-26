import os

import numpy as np
import torch

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.data_io.base_dataloader import generate_movie_splits
from openretina.data_io.hoefling_2024.constants import CLIP_LENGTH, NUM_CLIPS


def movies_from_pickle(file_path: str | os.PathLike) -> MoviesTrainTestSplit:
    """
    Load movie data from a pickle file and return it as a MoviesTrainTestSplit object.
    """
    return MoviesTrainTestSplit.from_pickle(file_path)


# Helper function to assemble the datasets
def apply_random_sequences(
    movie_train,
    movie_train_subset,
    movie_val,
    movie_test,
    random_sequences: np.ndarray,
    val_clip_idx: list[int],
    clip_length: int,
):
    movies = {
        "left": {
            "train": {},
            "validation": torch.flip(movie_val, [-1]),
            "test": torch.flip(movie_test, [-1]),
        },
        "right": {"train": {}, "validation": movie_val, "test": movie_test},
        "val_clip_idx": val_clip_idx,
    }

    for sequence_index in range(random_sequences.shape[1]):
        reordered_movie = torch.zeros_like(movie_train_subset)
        # explicitly cast to int, as random_sequences is a np.uint8 which is prone to overflow.
        clip_indices = [int(idx) for idx in random_sequences[:, sequence_index] if idx not in val_clip_idx]
        for k, clip_idx in enumerate(clip_indices):
            reordered_movie[:, k * clip_length : (k + 1) * clip_length, ...] = movie_train[
                :, clip_idx * clip_length : (clip_idx + 1) * clip_length, ...
            ]
        movies["right"]["train"][sequence_index] = reordered_movie  # type: ignore
        movies["left"]["train"][sequence_index] = torch.flip(reordered_movie, [-1])  # type: ignore

    return movies


# Main function that assembles everything
def get_all_movie_combinations(
    movie_train,
    movie_test,
    random_sequences: np.ndarray,
    validation_clip_indices: list[int],
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
):
    """
    Generates combinations of movie data for 'left' and 'right' perspectives and
    prepares training, validation, and test datasets. It reorders the training
    movies based on random sequences and flips the movies for the 'left' perspective.

    Parameters:
    - movie_train: Tensor representing the training movie data.
    - movie_test: Tensor representing the test movie data.
    - random_sequences: Numpy array of random sequences for reordering training movies.
    - val_clip_idx: list of indices for validation clips. Needs to be between 0 and the number of clips.

    Returns:
    - movies: Dictionary with processed movies for 'left' and 'right' perspectives, each
      containing 'train', 'validation', and 'test' datasets.
    """

    # Generate train, validation, and test datasets
    movie_train_subset, movie_val, movie_test_dict = generate_movie_splits(
        movie_train, {"test": movie_test}, validation_clip_indices, num_clips, clip_length
    )

    # Assemble datasets into the final movies structure using the random sequences
    movies = apply_random_sequences(
        torch.tensor(movie_train, dtype=torch.float),
        movie_train_subset,
        movie_val,
        movie_test_dict["test"],
        random_sequences,
        validation_clip_indices,
        clip_length,
    )

    return movies


def gen_start_indices(
    random_sequences: np.ndarray, val_clip_idx: list[int], clip_length: int, chunk_size: int, num_clips: int
):
    """
    Optimized function to generate a list of indices for training chunks while
    excluding validation clips.

    :param random_sequences: int np array; 108 x 20, giving the ordering of the
                             108 training clips for the 20 different sequences
    :param val_clip_idx:     list of integers indicating the clips to be used
                             for validation
    :param clip_length:      clip length in  (5s*30frames/s = 150 frames)
    :param chunk_size:       temporal chunk sframesize per sample in frames (50)
    :param num_clips:        total number of training clips (108)
    :return: dict; with keys train, validation, and test, and index list as
             values
    """
    # Validation clip indices are consecutive, because the validation clip and
    # stimuli are already isolated in other functions.
    val_start_idx = list(np.linspace(0, clip_length * (len(val_clip_idx) - 1), len(val_clip_idx), dtype=int))

    start_idx_dict = {"train": {}, "validation": val_start_idx, "test": [0]}

    num_train_clips = num_clips - len(val_clip_idx)

    if random_sequences.shape[1] == 1:
        start_idx_dict["train"] = list(np.arange(0, clip_length * num_train_clips, chunk_size, dtype=int))
    else:
        for sequence_index in range(random_sequences.shape[1]):
            start_idx_dict["train"][sequence_index] = list(  # type: ignore
                np.arange(0, clip_length * num_train_clips, chunk_size, dtype=int)
            )

    return start_idx_dict
