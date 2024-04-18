from typing import List, Optional, TypedDict

import numpy as np
import torch
from tqdm.auto import tqdm

from openretina.constants import CLIP_LENGTH, NUM_CLIPS, NUM_VAL_CLIPS
from openretina.dataloaders import get_movie_dataloader
from openretina.neuron_data_io import NeuronData
from openretina.stimuli import load_chirp, load_moving_bar


class MoviesDict(TypedDict):
    train: np.ndarray
    test: np.ndarray
    random_sequences: Optional[np.ndarray]


def get_all_movie_combinations(
    movie_train,
    movie_test,
    random_sequences: np.ndarray,
    val_clip_idx: Optional[List[int]] = None,
    num_clips: int = NUM_CLIPS,
    num_val_clips: int = NUM_VAL_CLIPS,
    clip_length: int = CLIP_LENGTH,
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
        val_clip_idx = list(np.sort(rnd.choice(num_clips, num_val_clips, replace=False)))

    # Convert movie data to tensors
    movie_train = torch.tensor(movie_train, dtype=torch.float)
    movie_test = torch.tensor(movie_test, dtype=torch.float)

    channels, _, px_y, px_x = movie_train.shape

    # Prepare validation movie data
    movie_val = torch.zeros((channels, len(val_clip_idx) * clip_length, px_y, px_x), dtype=torch.float)
    for i, ind in enumerate(val_clip_idx):
        movie_val[:, i * clip_length : (i + 1) * clip_length, ...] = movie_train[
            :, ind * clip_length : (ind + 1) * clip_length, ...
        ]

    # Create a boolean mask to indicate which clips are not part of the validation set
    # They are going to get removed
    mask = np.ones(num_clips, dtype=bool)
    mask[val_clip_idx] = False
    train_clip_idx = np.arange(num_clips)[mask]

    movie_train_subset = torch.cat(
        [movie_train[:, i * clip_length : (i + 1) * clip_length] for i in train_clip_idx],
        dim=1,
    )

    # Initialize movie dictionaries
    multiple_train_movies = True if random_sequences.shape[1] > 1 else False
    if multiple_train_movies:
        movies = {
            "left": {
                "train": {},
                "validation": torch.flip(movie_val, [-1]),
                "test": torch.flip(movie_test, [-1]),
            },
            "right": {"train": {}, "validation": movie_val, "test": movie_test},
            "val_clip_idx": val_clip_idx,
        }
    else:
        movies = {
            "left": {
                "train": torch.flip(movie_train_subset, [-1]),
                "validation": torch.flip(movie_val, [-1]),
                "test": torch.flip(movie_test, [-1]),
            },
            "right": {
                "train": movie_train_subset,
                "validation": movie_val,
                "test": movie_test,
            },
            "val_clip_idx": val_clip_idx,
        }

    # Process training movies for each random sequence, if multiple
    if multiple_train_movies:
        for sequence_index in range(random_sequences.shape[1]):
            k = 0
            reordered_movie = torch.zeros_like(movie_train_subset)
            for clip_idx in random_sequences[:, sequence_index]:
                if clip_idx in val_clip_idx:
                    continue
                reordered_movie[:, k * clip_length : (k + 1) * clip_length, ...] = movie_train[
                    :, clip_idx * clip_length : (clip_idx + 1) * clip_length, ...
                ]
                k += 1
            movies["right"]["train"][sequence_index] = reordered_movie
            movies["left"]["train"][sequence_index] = torch.flip(reordered_movie, [-1])

    return movies


def gen_start_indices(random_sequences, val_clip_idx, clip_length, chunk_size, num_clips):
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
    # Validation clip indices are consecutive, because the validation clip and stimuli are already isolated in other functions.
    val_start_idx = list(np.linspace(0, clip_length * (len(val_clip_idx) - 1), len(val_clip_idx), dtype=int))

    start_idx_dict = {"train": {}, "validation": val_start_idx, "test": [0]}

    num_train_clips = num_clips - len(val_clip_idx)

    for sequence_index in range(random_sequences.shape[1]):
        start_idx_dict["train"][sequence_index] = list(
            np.arange(0, clip_length * (num_train_clips - 1), chunk_size, dtype=int)
        )

    return start_idx_dict


def natmov_dataloaders_v2(
    neuron_data_dictionary,
    movies_dictionary: MoviesDict,
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
    num_val_clips: int = NUM_VAL_CLIPS,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert (
        isinstance(movies_dictionary, dict) and "train" in movies_dictionary and "test" in movies_dictionary
    ), "movies_dictionary should be a dictionary with keys 'train' and 'test'."
    assert all(
        field in next(iter(neuron_data_dictionary.values())) for field in ["responses_final", "stim_id"]
    ), "Check the neuron data dictionary sub-dictionaries for the minimal required fields: 'responses_final' and 'stim_id'."

    assert next(iter(neuron_data_dictionary.values()))["stim_id"] in [
        5,
        "salamander_natural",
    ], "This function only supports natural movie stimuli."

    # Draw validation clips based on the random seed
    rnd = np.random.RandomState(seed)
    val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

    clip_chunk_sizes = {
        "train": train_chunk_size,
        "validation": clip_length,
        "test": movies_dictionary["test"].shape[1],
    }
    dataloaders = {"train": {}, "validation": {}, "test": {}}

    # Get the random sequences of movies presentatios for each session if available
    if "random_sequences" not in movies_dictionary or movies_dictionary["random_sequences"] is None:
        movie_length = movies_dictionary["train"].shape[1]
        random_sequences = np.arange(0, movie_length // clip_length)[:, np.newaxis]
    else:
        random_sequences = movies_dictionary["random_sequences"]

    movies = get_all_movie_combinations(
        movies_dictionary["train"],
        movies_dictionary["test"],
        random_sequences,
        val_clip_idx=val_clip_idx,
        clip_length=clip_length,
    )
    start_indices = gen_start_indices(random_sequences, val_clip_idx, clip_length, train_chunk_size, num_clips)
    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
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
            dataloaders[fold][session_key] = get_movie_dataloader(
                movies=movies[neuron_data.eye][fold],
                responses=neuron_data.response_dict[fold],
                roi_ids=neuron_data.roi_ids,
                roi_coords=neuron_data.roi_coords,
                group_assignment=neuron_data.group_assignment,
                scan_sequence_idx=neuron_data.scan_sequence_idx,
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                start_indices=start_indices[fold],
                batch_size=batch_size,
                scene_length=clip_length,
            )

    return dataloaders


def get_chirp_dataloaders(
    neuron_data_dictionary,
    train_chunk_size: Optional[int] = None,
    batch_size: int = 32,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert all(
        field in next(iter(neuron_data_dictionary.values()))
        for field in ["responses_final", "stim_id", "chirp_trigger_times"]
    ), "Check the neuron data dictionary sub-dictionaries for the minimal required fields: 'responses_final', 'stim_id' and 'chirp_trigger_times'."

    assert next(iter(neuron_data_dictionary.values()))["stim_id"] == 1, "This function only supports chirp stimuli."

    dataloaders = {"train": {}}

    chirp_triggers = next(iter(neuron_data_dictionary.values()))["chirp_trigger_times"][0]
    # 2 triggers per chirp presentation
    num_chirps = len(chirp_triggers) // 2

    # Get it into chan, time, height, width
    chirp_stimulus = torch.tensor(load_chirp(), dtype=torch.float32).permute(3, 0, 1, 2)

    chirp_stimulus = chirp_stimulus.repeat(1, num_chirps, 1, 1)

    # Use full chirp for training if no chunk size is provided
    clip_chunk_sizes = {
        "train": train_chunk_size if train_chunk_size is not None else chirp_stimulus.shape[1] // num_chirps,
    }

    # 5 chirp presentations
    start_indices = np.arange(0, chirp_stimulus.shape[1] - 1, chirp_stimulus.shape[1] // num_chirps).tolist()

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating chirp dataloaders"):
        neuron_data = NeuronData(
            **session_data,
            random_sequences=None,
            val_clip_idx=None,
            num_clips=None,
            clip_length=None,
        )

        session_key += "_chirp"

        dataloaders["train"][session_key] = get_movie_dataloader(
            movies=chirp_stimulus if neuron_data.eye == "right" else torch.flip(chirp_stimulus, [-1]),
            responses=neuron_data.response_dict["train"],
            roi_ids=neuron_data.roi_ids,
            roi_coords=neuron_data.roi_coords,
            group_assignment=neuron_data.group_assignment,
            scan_sequence_idx=neuron_data.scan_sequence_idx,
            split="train",
            chunk_size=clip_chunk_sizes["train"],
            start_indices=start_indices,
            batch_size=batch_size,
            scene_length=chirp_stimulus.shape[1] // num_chirps,
            drop_last=False,
        )

    return dataloaders


def get_mb_dataloaders(
    neuron_data_dictionary,
    train_chunk_size: Optional[int] = None,
    batch_size: int = 32,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert all(
        field in next(iter(neuron_data_dictionary.values()))
        for field in ["responses_final", "stim_id", "mb_trigger_times"]
    ), "Check the neuron data dictionary sub-dictionaries for the minimal required fields: 'responses_final', 'stim_id' and 'mb_trigger_times'."

    assert (
        next(iter(neuron_data_dictionary.values()))["stim_id"] == 2
    ), "This function only supports moving bar stimuli."

    dataloaders = {"train": {}}

    mb_triggers = next(iter(neuron_data_dictionary.values()))["mb_trigger_times"][0]
    num_repeats = len(mb_triggers) // 8

    # Get it into chan, time, height, width
    mb_stimulus = torch.tensor(load_moving_bar(), dtype=torch.float32).permute(3, 0, 1, 2)

    mb_stimulus = mb_stimulus.repeat(1, num_repeats, 1, 1)

    # 8 directions
    total_num_mbs = 8 * num_repeats

    # Default to each mb for training if no chunk size provided.
    clip_chunk_sizes = {
        "train": train_chunk_size if train_chunk_size is not None else mb_stimulus.shape[1] // total_num_mbs,
    }

    start_indices = np.arange(0, mb_stimulus.shape[1] - 1, step=mb_stimulus.shape[1] // total_num_mbs).tolist()

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating moving bars dataloaders"):
        neuron_data = NeuronData(
            **session_data,
            random_sequences=None,
            val_clip_idx=None,
            num_clips=None,
            clip_length=None,
        )

        session_key += "_mb"

        dataloaders["train"][session_key] = get_movie_dataloader(
            movies=mb_stimulus if neuron_data.eye == "right" else torch.flip(mb_stimulus, [-1]),
            responses=neuron_data.response_dict["train"],
            roi_ids=neuron_data.roi_ids,
            roi_coords=neuron_data.roi_coords,
            group_assignment=neuron_data.group_assignment,
            scan_sequence_idx=neuron_data.scan_sequence_idx,
            split="train",
            chunk_size=clip_chunk_sizes["train"],
            start_indices=start_indices,
            batch_size=batch_size,
            scene_length=mb_stimulus.shape[1] // total_num_mbs,
            drop_last=False,
        )

    return dataloaders
