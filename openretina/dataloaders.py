import bisect
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from .hoefling_2024.constants import SCENE_LENGTH

DataPoint = namedtuple("DataPoint", ("inputs", "targets"))
DataPointWithMeta = namedtuple("DataPoint", ("inputs", "categorical_metadata", "numerical_metadata", "targets"))


class MovieDataSet(Dataset):
    def __init__(self, movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size):
        # Will only be a dictionary for certain types of datasets, i.e. Hoefling 2022
        if split == "test" and isinstance(responses, dict):
            self.samples = movies, responses["avg"]
            self.test_responses_by_trial = responses["by_trial"]
            self.roi_ids = roi_ids
        else:
            self.samples = movies, responses
        self.chunk_size = chunk_size
        # Calculate the mean response per neuron (used for bias init in the model)
        self.mean_response = torch.mean(self.samples[1], dim=0)
        self.group_assignment = group_assignment
        self.roi_coords = roi_coords

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataPoint(*[self.samples[0][:, idx, ...], self.samples[1][idx, ...]])
        else:
            return DataPoint(
                *[
                    self.samples[0][:, idx : idx + self.chunk_size, ...],
                    self.samples[1][idx : idx + self.chunk_size, ...],
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
        return (
            f"MovieDataSet with {self.samples[1].shape[1]} neuron responses "
            f"to a movie of shape {list(self.samples[0].shape)}."
        )

    def __repr__(self):
        return str(self)


class MovieAndMetadataDataSet(Dataset):
    def __init__(self, movies, responses, metadata, split, chunk_size):
        # Will only be a dictionary for certain types of datasets, i.e. Hoefling 2022
        if split == "test" and isinstance(responses, dict):
            self.samples = movies, responses["avg"]
            self.test_responses_by_trial = responses["by_trial"]
        else:
            self.samples = movies, responses
        self.all_metadata = metadata
        self.categorical_metadata = [
            metadata[key].astype(int) for key in metadata if np.issubdtype(metadata[key].dtype, np.str_)
        ]

        # Concatenate and transpose to have them n_neurons x n_features
        self.numerical_metadata = np.concatenate(
            [metadata[key] for key in metadata if np.issubdtype(metadata[key].dtype, np.number)],
            axis=1,
        ).astype(np.float32)

        self.chunk_size = chunk_size
        # Calculate the mean response per neuron (used for bias init in the model)
        self.mean_response = torch.mean(self.samples[1], dim=0)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataPointWithMeta(
                *[
                    self.samples[0][:, idx, ...],
                    self.categorical_metadata,
                    self.numerical_metadata,
                    self.samples[1][idx, ...],
                ]
            )
        else:
            return DataPointWithMeta(
                *[
                    self.samples[0][:, idx : idx + self.chunk_size, ...],
                    self.categorical_metadata,
                    self.numerical_metadata,
                    self.samples[1][idx : idx + self.chunk_size, ...],
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
        return (
            f"MovieDataSet with {self.samples[1].shape[1]} neuron responses "
            f"to a movie of shape {list(self.samples[0].shape)}."
        )

    def __repr__(self):
        return str(self)


class MovieSampler(Sampler):
    def __init__(self, start_indices, split, chunk_size, movie_length, scene_length=None):
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size
        self.movie_length = movie_length
        self.scene_length = SCENE_LENGTH if scene_length is None else scene_length

    def __iter__(self):
        if self.split == "train" and (self.scene_length != self.chunk_size):
            # Always start the clip from a random point in the scene, within the chosen chunk size
            # All while making sure it does not go over the scene length bound.
            shifted_indices = gen_shifts(
                np.arange(0, self.movie_length + 1, self.scene_length),
                self.indices,
                self.chunk_size,
            )

            # Shuffle the indices
            indices_shuffling = np.random.permutation(len(self.indices))
        else:
            # shift = 0
            shifted_indices = self.indices
            indices_shuffling = np.arange(len(self.indices))

        return iter(np.array(shifted_indices)[indices_shuffling])

    def __len__(self):
        return len(self.indices)


def gen_shifts(clip_bounds, start_indices, clip_chunk_size=50):
    """
    Generate shifted indices based on clip bounds and start indices.
    Assumes that the original start indices are already within the clip bounds.
    If they are not, it changes the overflowing indexes to respect the closest bound.

    Args:
        clip_bounds (list): A list of clip bounds.
        start_indices (list): A list of start indices.
        clip_chunk_size (int, optional): The size of each clip chunk. Defaults to 50.

    Returns:
        list: A list of shifted indices.

    """

    def get_next_bound(value, bounds):
        insertion_index = bisect.bisect_right(bounds, value)
        return bounds[min(insertion_index, len(bounds) - 1)]

    shifted_indices = []
    shifts = np.random.randint(1, clip_chunk_size // 2, len(start_indices))

    for i, start_idx in enumerate(start_indices):
        next_bound = get_next_bound(start_idx, clip_bounds)
        if start_idx + shifts[i] + clip_chunk_size < next_bound:
            shifted_indices.append(start_idx + shifts[i])
        elif start_idx + clip_chunk_size > next_bound:
            shifted_indices.append(next_bound - clip_chunk_size)
        else:
            shifted_indices.append(start_idx)
    return shifted_indices


def get_movie_dataloader(
    movies: np.ndarray | torch.Tensor | dict[int, np.ndarray],
    responses: Float[np.ndarray, "n_frames n_neurons"],
    roi_ids: Optional[Float[np.ndarray, " n_neurons"]],
    roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]],
    group_assignment: Optional[Float[np.ndarray, " n_neurons"]],
    split: str,
    start_indices: List[int] | Dict[int, List[int]],
    scan_sequence_idx: Optional[int] = None,
    chunk_size: int = 50,
    batch_size: int = 32,
    scene_length: Optional[int] = None,
    drop_last=True,
    use_base_sequence=False,
    **kwargs,
):
    """
    TODO docstring
    """
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return None

    if scene_length is not None and split == "train" and chunk_size > scene_length:
        raise ValueError("Clip chunk size must be smaller than scene length to not exceed clip bounds during training.")

    # for right movie: flip second frame size axis!
    if split == "train" and isinstance(movies, dict) and scan_sequence_idx is not None:
        if use_base_sequence:
            scan_sequence_idx = 20  # 20 is the base sequence
        dataset = MovieDataSet(
            movies[scan_sequence_idx], responses, roi_ids, roi_coords, group_assignment, split, chunk_size
        )
        sampler = MovieSampler(
            start_indices[scan_sequence_idx],
            split,
            chunk_size,
            movie_length=movies[scan_sequence_idx].shape[1],
            scene_length=scene_length,
        )
    else:
        dataset = MovieDataSet(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        sampler = MovieSampler(
            start_indices, split, chunk_size, movie_length=movies.shape[1], scene_length=scene_length
        )

    return DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, drop_last=split == "train" and drop_last, **kwargs
    )


def get_dims_for_loader_dict(dataloaders: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[int, ...]] | Tuple]:
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


def get_io_dims(data_loader) -> Dict[str, Tuple[int, ...]] | Tuple:
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
        return {k: v.shape for k, v in items.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    else:
        return tuple(v.shape for v in items)


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


def filter_empty_videos(batch):
    """
    Filters out batches containing empty videos.
    To be used as a collate_fn in a DataLoader.

    Args:
        batch (list): A list of tuples representing the batch.

    Returns:
        tuple of torch.Tensor: The collated batch after filtering out empty videos.

    """
    # Filter out empty videos
    batch = [x for x in batch if x[0].shape[1] > 0]
    return default_collate(batch)
