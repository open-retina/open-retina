from typing import List, Optional

import numpy as np
import torch

from openretina.constants import CLIP_LENGTH, NUM_CLIPS, NUM_VAL_CLIPS
from openretina.dataloaders import get_movie_dataloader
from openretina.neuron_data_io import NeuronData


def get_all_movie_combinations(
    movie_train,
    movie_test,
    random_sequences: np.ndarray,
    val_clip_idx: Optional[List[int]] = None,
    seed=1000,
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
    -seed: seed for random number generator, if val_clip_idx is None.

    Returns:
    - movies: Dictionary with processed movies for 'left' and 'right' perspectives, each
      containing 'train', 'validation', and 'test' datasets.
    """
    if val_clip_idx is None:
        rnd = np.random.RandomState(seed)
        val_clip_idx = list(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))

    # Convert movie data to tensors
    movie_train = torch.tensor(movie_train, dtype=torch.float)
    movie_test = torch.tensor(movie_test, dtype=torch.float)

    channels, train_length, px_y, px_x = movie_train.shape
    clip_length = train_length // random_sequences.shape[0]

    # Prepare validation movie data
    movie_val = torch.zeros((channels, len(val_clip_idx) * clip_length, px_y, px_x), dtype=torch.float)
    for i, ind in enumerate(val_clip_idx):
        movie_val[:, i * clip_length : (i + 1) * clip_length] = movie_train[
            :, ind * clip_length : (ind + 1) * clip_length
        ]

    # Initialize movie dictionaries
    movies = {
        "left": {
            "train": {},
            "validation": torch.flip(movie_val, [-1]),
            "test": torch.flip(movie_test, [-1]),
        },
        "right": {"train": {}, "validation": movie_val, "test": movie_test},
    }

    # Process training movies for each random sequence
    for i in range(random_sequences.shape[1]):
        reordered_movie = torch.zeros_like(movie_train)
        for k, ind in enumerate(random_sequences[:, i]):
            reordered_movie[:, k * clip_length : (k + 1) * clip_length] = movie_train[
                :, ind * clip_length : (ind + 1) * clip_length
            ]

        movies["right"]["train"][i] = reordered_movie
        movies["left"]["train"][i] = torch.flip(reordered_movie, [-1])

    movies["val_clip_idx"] = val_clip_idx

    return movies


def gen_start_indices(random_sequences, val_clip_idx, clip_length, chunk_size, num_clips):  # 108 x 20 integer
    """
    Generates a list of indices into movie frames that can be used as start
    indices for training chunks without including validation clips in the
    training set.

    Args:
        random_sequences (np.ndarray): Integer array of shape (108, 20) giving the ordering of the
                                       108 training clips for the 20 different sequences.
        val_clip_idx (list): List of integers indicating the 15 clips to be used for validation.
        clip_length (int): Clip length in frames (5s * 30 frames/s = 150 frames).
        chunk_size (int): Temporal chunk size per sample in frames (50).
        num_clips (int): Total number of training clips (108).

    Returns:
        dict: A dictionary with keys "train", "validation", and "test", and index lists as values.
    """
    val_start_idx = list(np.linspace(0, clip_length * (len(val_clip_idx) - 1), len(val_clip_idx), dtype=int))

    start_idx_dict = {"train": {}, "validation": val_start_idx, "test": [0]}
    for i in range(random_sequences.shape[1]):  # iterate over the 20 different movie permutations
        start_idx = 0
        current_idx = 0
        seq_start_idx = []
        seq_length = []
        for k, ind in enumerate(random_sequences[: num_clips // 2, i]):  # over first half of the clips
            if ind in val_clip_idx:
                length = current_idx - start_idx
                if length > 0:
                    seq_start_idx.append(start_idx)
                    seq_length.append(length)
                start_idx = current_idx + clip_length
            current_idx += clip_length
        length = current_idx - start_idx
        if length > 0:
            seq_start_idx.append(start_idx)
            seq_length.append(length)
        start_idx = current_idx
        for k, ind in enumerate(random_sequences[num_clips // 2 :, i]):  # over second half of the clips
            if ind in val_clip_idx:
                length = current_idx - start_idx
                if length > 0:
                    seq_start_idx.append(start_idx)
                    seq_length.append(length)
                start_idx = current_idx + clip_length
            current_idx += clip_length
        length = current_idx - start_idx
        if length > 0:
            seq_start_idx.append(start_idx)
            seq_length.append(length)

        chunk_start_idx = []
        for start, length in zip(seq_start_idx, seq_length):
            idx = np.arange(start, start + length - chunk_size + 1, chunk_size)
            chunk_start_idx += list(idx[:-1])
        start_idx_dict["train"][i] = chunk_start_idx
    return start_idx_dict


def optimized_gen_start_indices(random_sequences, val_clip_idx, clip_length, chunk_size, num_clips):
    """
    Optimized function to generate a list of indices for training chunks while
    excluding validation clips.

    :param random_sequences: int np array; 108 x 20, giving the ordering of the
                             108 training clips for the 20 different sequences
    :param val_clip_idx:     list of integers indicating the 15 clips to be used
                             for validation
    :param clip_length:      clip length in frames (5s*30frames/s = 150 frames)
    :param chunk_size:       temporal chunk size per sample in frames (50)
    :param num_clips:        total number of training clips (108)
    :return: dict; with keys train, validation, and test, and index list as
             values
    """
    val_clip_set = set(val_clip_idx)
    val_start_idx = list(np.linspace(0, clip_length * (len(val_clip_idx) - 1), len(val_clip_idx), dtype=int))

    start_idx_dict = {"train": {}, "validation": val_start_idx, "test": [0]}

    for sequence_index in range(random_sequences.shape[1]):
        start_idx = 0
        current_idx = 0
        seq_start_idx = []
        seq_length = []

        for clip_index in random_sequences[:, sequence_index]:
            if clip_index in val_clip_set:
                length = current_idx - start_idx
                if length > 0:
                    seq_start_idx.append(start_idx)
                    seq_length.append(length)
                start_idx = current_idx + clip_length
            current_idx += clip_length

        # Handling the last segment
        length = current_idx - start_idx
        if length > 0:
            seq_start_idx.append(start_idx)
            seq_length.append(length)

        chunk_start_idx = [
            idx
            for start, length in zip(seq_start_idx, seq_length)
            for idx in range(start, start + length - chunk_size + 1, chunk_size)[:-1]
        ]
        start_idx_dict["train"][sequence_index] = chunk_start_idx

    return start_idx_dict


def natmov_dataloaders_v2(
    neuron_data_dictionary,
    movies_dictionary,
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
):
    # make sure movies and responses arrive as torch tensors!!!
    rnd = np.random.RandomState(seed)  # make sure whether we want the validation set to depend on the seed

    num_clips, clip_length = NUM_CLIPS, CLIP_LENGTH
    val_clip_idx = list(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))

    clip_chunk_sizes = {
        "train": train_chunk_size,
        "validation": clip_length,
        "test": 5 * clip_length,
    }
    dataloaders = {"train": {}, "validation": {}, "test": {}}
    # draw validation indices so that a validation movie can be returned!
    random_sequences = movies_dictionary["random_sequences"]
    movies = get_all_movie_combinations(
        movies_dictionary["train"], movies_dictionary["test"], random_sequences, val_clip_idx=val_clip_idx
    )
    start_indices = gen_start_indices(random_sequences, val_clip_idx, clip_length, train_chunk_size, num_clips)
    for session_key, session_data in neuron_data_dictionary.items():
        neuron_data = NeuronData(
            **session_data,
            random_sequences=random_sequences,  # Used together with the validation index to get the validation response in the corresponding dict
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # if neuron_data.responses_train.shape[-1] == 0:
        #     print("skipped: {}".format(session_key))
        #     break
        for fold in ["train", "validation", "test"]:
            if not (hasattr(neuron_data, "roi_coords")):
                neuron_data.roi_mask = []
            dataloaders[fold][session_key] = get_movie_dataloader(
                movies[neuron_data.eye][fold],
                neuron_data.response_dict[fold],
                neuron_data.roi_ids,
                neuron_data.roi_coords,
                neuron_data.group_assignment,
                neuron_data.scan_sequence_idx,
                fold,
                clip_chunk_sizes[fold],
                start_indices[fold],
                batch_size,
            )

    return dataloaders
