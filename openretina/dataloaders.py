from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from openretina.hoefling_2024.constants import SCENE_LENGTH

DataPoint = namedtuple("DataPoint", ("inputs", "targets"))


class MovieDataSet(Dataset):
    def __init__(self, movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size):
        # Will only be a dictionary for certain types of datasets, i.e. Hoefling 2022
        if split == "test" and isinstance(responses, dict):
            self.samples = movies, responses["avg"]
            self.test_responses_by_trial = responses["by_trial"]
            self.roi_ids = roi_ids
            self.group_assignment = group_assignment
        else:
            self.samples = movies, responses
            self.roi_coords = roi_coords
        self.chunk_size = chunk_size
        # Calculate the mean response per neuron (used for bias init in the model)
        self.mean_response = torch.mean(self.samples[1], dim=0)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataPoint(*[self.samples[0][:, idx, ...], self.samples[1][idx, ...]])
        else:
            return DataPoint(
                *[
                    self.samples[0][:, idx:idx + self.chunk_size, ...],
                    self.samples[1][idx:idx + self.chunk_size, ...],
                ]
            )

    @property
    def movies(self):
        return self.samples[0]

    @property
    def responses(self):
        return self.samples[1]

    def __len__(self):
        # Returns the number of chunks of clips and responses used for training
        return self.samples[1].shape[0] // self.chunk_size

    def __str__(self):
        return (f"MovieDataSet with {self.samples[1].shape[1]} neuron responses "
                f"to a movie of shape {list(self.samples[0].shape)}.")

    def __repr__(self):
        return str(self)


class MovieSampler(Sampler):
    def __init__(self, start_indices, split, chunk_size, scene_length=None):
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size
        self.scene_length = SCENE_LENGTH if scene_length is None else scene_length

    def __iter__(self):
        if self.split == "train" and (self.scene_length != self.chunk_size):
            # Always start the clip from a random point in the scene, within the chosen chunk size
            shift = np.random.randint(0, min(self.scene_length - self.chunk_size, self.chunk_size))

            # Shuffle the indices
            indices_shuffling = np.random.permutation(len(self.indices))
        else:
            shift = 0
            indices_shuffling = np.arange(len(self.indices))

        return iter(np.array([idx + shift for idx in self.indices])[indices_shuffling])

    def __len__(self):
        return len(self.indices)


def get_movie_dataloader(
    movies: np.ndarray | torch.Tensor | dict[int, np.ndarray],
    responses: Float[np.ndarray, "n_frames n_neurons"],  # noqa
    roi_ids: Optional[Float[np.ndarray, "n_neurons"]],  # noqa
    roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]],  # noqa
    group_assignment: Optional[Float[np.ndarray, "n_neurons"]],  # noqa
    split: str,
    start_indices: List[int] | Dict[int, List[int]],
    scan_sequence_idx: Optional[int] = None,
    chunk_size: int = 50,
    batch_size: int = 32,
    scene_length: Optional[int] = None,
    drop_last=True,
    **kwargs,
):
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return None

    # for right movie: flip second frame size axis!
    if split == "train" and isinstance(movies, dict) and scan_sequence_idx is not None:
        dataset = MovieDataSet(
            movies[scan_sequence_idx], responses, roi_ids, roi_coords, group_assignment, split, chunk_size
        )
        sampler = MovieSampler(start_indices[scan_sequence_idx], split, chunk_size, scene_length=scene_length)
    else:
        dataset = MovieDataSet(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        sampler = MovieSampler(start_indices, split, chunk_size, scene_length=scene_length)

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True if (split == "train" and drop_last) else False,
        **kwargs,
    )


def get_dims_for_loader_dict(dataloaders):
    """
    Borrowed from nnfabrik/utility/nn_helpers.py.

    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.

    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.

    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def get_io_dims(data_loader):
    """
    Borrowed from nnfabrik/utility/nn_helpers.py.

    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.

    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.

    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)


def filter_nan_collate(batch):
    """
    Filters out batches containing NaN values and then calls the default_collate function.
    Can happen for inferred spikes exported with CASCADE.
    To be used as a collate_fn in a DataLoader.

    Args:
        batch (list): A list of tuples representing the batch.

    Returns:
        tuple of torch.Tensor: The collated batch after filtering out NaN values.

    """
    batch = list(filter(lambda x: not np.isnan(x[1]).any(), batch))
    return default_collate(batch)


def filter_different_size(batch):
    """
    Filters out batches that do not have the same shape as most of the other batches.
    """
    # Get the shapes of all the elements in the batch
    shapes = [element[1].shape for element in batch]

    # Find the most common shape in the batch
    most_common_shape = max(set(shapes), key=shapes.count)

    # Filter out elements that do not have the most common shape
    filtered_batch = [element for element in batch if element[1].shape == most_common_shape]

    # If the filtered batch is empty, return None
    if len(filtered_batch) == 0:
        return None

    # Collate the filtered batch using the default collate function
    collated_batch = default_collate(filtered_batch)

    return collated_batch
