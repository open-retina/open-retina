from typing import Any, List, Optional

import numpy as np
from tqdm.auto import tqdm

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import generate_movie_splits, get_movie_dataloader
from openretina.data_io.maheswaranathan_2023.constants import CLIP_LENGTH
from openretina.data_io.maheswaranathan_2023.responses import NeuronDataSplit


def get_movie_splits(
    movie_train,
    movie_test,
    val_clip_idx: Optional[List[int]],
    num_clips: int,
    num_val_clips: int,
    clip_length: int,
):
    movie_train_subset, movie_val, movie_test, val_clip_idx = generate_movie_splits(
        movie_train, movie_test, val_clip_idx, num_clips, num_val_clips, clip_length
    )

    movies = {
        "train": movie_train_subset,
        "validation": movie_val,
        "test": movie_test,
        "val_clip_idx": val_clip_idx,
    }

    return movies


def multiple_movies_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, MoviesTrainTestSplit],
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    clip_length: int = CLIP_LENGTH,
    num_val_clips: int = 10,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."

    assert (
        neuron_data_dictionary.keys() == movies_dictionary.keys()
    ), "The keys of neuron_data_dictionary and movies_dictionary should match."

    # assert all(field in next(iter(neuron_data_dictionary.values())) for field in ["responses_final", "stim_id"]), (
    #     "Check the neuron data dictionary sub-dictionaries for the minimal"
    #     " required fields: 'responses_final' and 'stim_id'."
    # )

    # Initialise dataloaders
    dataloaders: dict[str, Any] = {"train": {}, "validation": {}, "test": {}}

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
        # Extract all data related to the movies first
        num_clips = movies_dictionary[session_key].train.shape[1] // clip_length

        # Draw validation clips based on the random seed
        rnd = np.random.RandomState(seed)
        val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

        clip_chunk_sizes = {
            "train": train_chunk_size,
            "validation": clip_length,
            "test": movies_dictionary[session_key].test.shape[1],
        }

        all_movies = get_movie_splits(
            movies_dictionary[session_key].train,
            movies_dictionary[session_key].test,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            num_val_clips=num_val_clips,
            clip_length=clip_length,
        )

        # Extract all splits from neural data
        neuron_data = NeuronDataSplit(
            responses=session_data,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # Create dataloaders for each fold
        for fold in ["train", "validation", "test"]:
            dataloaders[fold][session_key] = get_movie_dataloader(
                movies=all_movies[fold],
                responses=neuron_data.response_dict[fold],
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                batch_size=batch_size,
                scene_length=clip_length,
            )

    return dataloaders
