from collections import namedtuple
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from .constants import SCENE_LENGTH


class MovieDataSet(Dataset):
    def __init__(self, movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size):
        if split == "test":
            self.samples = tuple((movies, responses["avg"]))
            self.test_responses_by_trial = responses["by_trial"]
            self.roi_ids = roi_ids
            self.group_assignment = group_assignment
        else:
            self.samples = tuple((movies, responses))
            self.roi_coords = roi_coords
        self.DataPoint = namedtuple("DataPoint", ("inputs", "targets"))
        self.chunk_size = chunk_size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.DataPoint(*[self.samples[0][:, idx, ...], self.samples[1][idx, ...]])
        else:
            return self.DataPoint(
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
        return self.samples[1].shape[1]

    def __str__(self):
        return f"MovieDataSet with {len(self)} neuron responses to a movie of shape {list(self.samples[0].shape)}."


class MovieSampler(Sampler):
    def __init__(self, start_indices, split, chunk_size):
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size

    def __iter__(self):
        if self.split == "train":
            # Always start the clip from a random point in the scene, within the chosen chunk size
            shift = np.random.randint(0, min(SCENE_LENGTH - self.chunk_size, self.chunk_size))

            # Shuffle the indices
            indices_shuffling = np.random.permutation(len(self.indices))
        else:
            shift = 0
            indices_shuffling = np.arange(len(self.indices))

        return iter(np.array([idx + shift for idx in self.indices])[indices_shuffling])

    def __len__(self):
        return len(self.indices)


def get_movie_dataloader(
    movies,
    responses,
    roi_ids,
    roi_coords,
    group_assignment,
    scan_sequence_idx,
    split,
    chunk_size,
    start_indices,
    batch_size,
    **kwargs,
):
    # for right movie: flip second frame size axis!
    if split == "train":
        dataset = MovieDataSet(
            movies[scan_sequence_idx], responses, roi_ids, roi_coords, group_assignment, split, chunk_size
        )
        sampler = MovieSampler(start_indices[scan_sequence_idx], split, chunk_size)
    else:
        dataset = MovieDataSet(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        sampler = MovieSampler(start_indices, split, chunk_size)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True if split == "train" else False)


def dataloaders_from_dictionaries(
    dataloaders_dict: dict, movies_dict: dict, batchsize: int = 32, train_chunk_size: Optional[int] = None
):
    dataloaders = {}
    for split in dataloaders_dict.keys():
        dataloaders[split] = {}
        for session_key in dataloaders_dict[split].keys():
            eye = dataloaders_dict[split][session_key]["eye"]
            dataloader_args = deepcopy(dataloaders_dict[split][session_key])
            if split == "train" and train_chunk_size is not None:
                dataloader_args["chunk_size"] = train_chunk_size
            dataloaders[split][session_key] = get_movie_dataloader(
                movies_dict[eye][split],
                **dataloader_args,
                batch_size=batchsize,
            )
    return dataloaders


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
