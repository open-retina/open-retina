import bisect
from collections import namedtuple
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from openretina.data_io.hoefling_2024.constants import SCENE_LENGTH

DataPoint = namedtuple("DataPoint", ("inputs", "targets"))


class MovieDataSet(Dataset):
    """
    A dataset class for handling movie data and corresponding neural responses.

    Args:
        movies (Float[np.ndarray | torch.Tensor, "n_channels n_frames h w"]): The movie data.
        responses (Float[np.ndarray, "n_frames n_neurons"]): The neural responses.
        roi_ids (Optional[Float[np.ndarray, " n_neurons"]]): A list of ROI IDs.
        roi_coords (Optional[Float[np.ndarray, "n_neurons 2"]]): A list of ROI coordinates.
        group_assignment (Optional[Float[np.ndarray, " n_neurons"]]): A list of group assignments (cell types).
        split (Literal["train", "validation", "val", "test"]):
                                                    The data split, either "train", "validation", "val", or "test".
        chunk_size (int): The size of the chunks to split the data into.

    Attributes:
        samples (tuple): A tuple containing movie data and neural responses.
        test_responses_by_trial (Optional[Dict[str, Any]]):
                                                A dictionary containing test responses by trial (only for test split).
        roi_ids (Optional[Float[np.ndarray, " n_neurons"]]): A list of region of interest (ROI) IDs.
        chunk_size (int): The size of the chunks to split the data into.
        mean_response (torch.Tensor): The mean response per neuron.
        group_assignment (Optional[Float[np.ndarray, " n_neurons"]]): A list of group assignments.
        roi_coords (Optional[Float[np.ndarray, "n_neurons 2"]]): A list of ROI coordinates.

    Methods:
        __getitem__(idx): Returns a DataPoint object for the given index or slice.
        movies: Returns the movie data.
        responses: Returns the neural responses.
        __len__(): Returns the number of chunks of clips and responses used for training.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a string representation of the dataset.
    """

    def __init__(
        self,
        movies: Float[np.ndarray | torch.Tensor, "n_channels n_frames h w"],
        responses: Float[np.ndarray, "n_frames n_neurons"],
        roi_ids: Optional[Float[np.ndarray, " n_neurons"]],
        roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]],
        group_assignment: Optional[Float[np.ndarray, " n_neurons"]],
        split: str | Literal["train", "validation", "val", "test"],
        chunk_size: int,
    ):
        # Will only be a dictionary for certain types of datasets, i.e. Hoefling 2022
        if split == "test" and isinstance(responses, dict):
            self.samples: tuple = movies, responses["avg"]
            self.test_responses_by_trial = responses["by_trial"]
            self.roi_ids = roi_ids
        else:
            self.samples = movies, responses

        self.chunk_size = chunk_size
        # Calculate the mean response per neuron (used for bias init in the model)
        self.mean_response = torch.mean(torch.Tensor(self.samples[1]), dim=0)
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


class MovieSampler(Sampler):
    """
    A custom sampler for selecting movie frames for training, validation, or testing.

    Args:
        start_indices (list[int]): List of starting indices for the movie sections to select.
        split (Literal["train", "validation", "val", "test"]): The type of data split.
        chunk_size (int): The size of each contiguous chunk of frames to select.
        movie_length (int): The total length of the movie.
        scene_length (Optional[int], optional): The length of each scene, if the movie is divided in any scenes.
                                                Defaults to None.
        allow_over_boundaries (bool, optional): Whether to allow selected chunks to go over scene boundaries.
                                                Defaults to False.

    Attributes:
        indices (list[int]): The starting indices for the movie sections to sample.
        split (str): The type of data split.
        chunk_size (int): The size of each chunk of frames.
        movie_length (int): The total length of the movie.
        scene_length (int): The length of each scene, if the movie is made up of scenes.
        allow_over_boundaries (bool): Whether to allow chunks to go over scene boundaries.

    Methods:
        __iter__(): Returns an iterator over the sampled indices.
        __len__(): Returns the number of starting indices (which will corresponds to the number of sampled clips).
    """

    def __init__(
        self,
        start_indices: list[int],
        split: str | Literal["train", "validation", "val", "test"],
        chunk_size: int,
        movie_length: int,
        scene_length: Optional[int] = None,
        allow_over_boundaries: bool = False,
    ):
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size
        self.movie_length = movie_length
        self.scene_length = SCENE_LENGTH if scene_length is None else scene_length
        self.allow_over_boundaries = allow_over_boundaries

    def __iter__(self):
        if self.split == "train" and (self.scene_length != self.chunk_size):
            if self.allow_over_boundaries:
                shifts = np.random.randint(0, self.chunk_size, len(self.indices))
                # apply shifts while making sure we do not exceed the movie length
                shifted_indices = np.minimum(self.indices + shifts, self.movie_length - self.chunk_size)
            else:
                shifted_indices = gen_shifts_with_boundaries(
                    np.arange(0, self.movie_length + 1, self.scene_length),
                    self.indices,
                    self.chunk_size,
                )
            # Shuffle the indices
            indices_shuffling = np.random.permutation(len(self.indices))
        else:
            shifted_indices = self.indices
            indices_shuffling = np.arange(len(self.indices))

        return iter(np.array(shifted_indices)[indices_shuffling])

    def __len__(self):
        return len(self.indices)


def gen_shifts_with_boundaries(
    clip_bounds: list[int] | np.ndarray, start_indices: list[int] | np.ndarray, clip_chunk_size: int = 50
) -> list[int]:
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
    shifts = np.random.randint(0, clip_chunk_size // 2, len(start_indices))

    for i, start_idx in enumerate(start_indices):
        next_bound = get_next_bound(start_idx, clip_bounds)
        if start_idx + shifts[i] + clip_chunk_size < next_bound:
            shifted_indices.append(start_idx + shifts[i])
        elif start_idx + clip_chunk_size > next_bound:
            shifted_indices.append(next_bound - clip_chunk_size)
        else:
            shifted_indices.append(start_idx)

    # Ensure we do not exceed the movie length when allowing over boundaries
    if shifted_indices[-1] + clip_chunk_size > clip_bounds[-1]:
        shifted_indices[-1] = clip_bounds[-1] - clip_chunk_size
    return shifted_indices


def get_movie_dataloader(
    movies: np.ndarray | torch.Tensor | dict[int, np.ndarray],
    responses: Float[np.ndarray, "n_frames n_neurons"],
    *,
    split: str | Literal["train", "validation", "val", "test"],
    start_indices: Optional[list[int] | Dict[int, list[int]]] = None,
    roi_ids: Optional[Float[np.ndarray, " n_neurons"]] = None,
    roi_coords: Optional[Float[np.ndarray, "n_neurons 2"]] = None,
    group_assignment: Optional[Float[np.ndarray, " n_neurons"]] = None,
    scan_sequence_idx: Optional[int] = None,
    chunk_size: int = 50,
    batch_size: int = 32,
    scene_length: Optional[int] = None,
    drop_last: bool = True,
    use_base_sequence: bool = False,
    allow_over_boundaries: bool = True,
    **kwargs,
):
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return None

    if not allow_over_boundaries and scene_length is not None and split == "train" and chunk_size > scene_length:
        raise ValueError("Clip chunk size must be smaller than scene length to not exceed clip bounds during training.")

    if start_indices is None:
        start_indices = handle_missing_start_indices(movies, chunk_size, scene_length, split)

    # for right movie: flip second frame size axis!
    if split == "train" and isinstance(movies, dict) and scan_sequence_idx is not None:
        assert isinstance(start_indices, dict), "Start indices should be a dictionary for this case."

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
            allow_over_boundaries=allow_over_boundaries,
        )
    else:
        assert not isinstance(movies, dict), "Movies should not be a dictionary for this case."
        assert not isinstance(start_indices, dict), "Start indices should not be a dictionary for this case."
        dataset = MovieDataSet(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        sampler = MovieSampler(
            start_indices,
            split,
            chunk_size,
            movie_length=movies.shape[1],
            scene_length=scene_length,
            allow_over_boundaries=allow_over_boundaries,
        )

    return DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, drop_last=split == "train" and drop_last, **kwargs
    )


<<<<<<< HEAD:openretina/data_io/dataloaders.py
def get_dims_for_loader_dict(dataloaders: dict[str, Dict[str, Any]]) -> dict[str, dict[str, tuple[int, ...]] | tuple]:
=======
def handle_missing_start_indices(
    movies: dict | np.ndarray | torch.Tensor, chunk_size: Optional[int], scene_length: Optional[int], split: str
):
    """
    Handle missing start indices for different splits of the dataset.

    Parameters:
    movies (dict or np.ndarray): The movies data, either as a dictionary of arrays or a single array.
    chunk_size (int or None): The size of each chunk for training split. Required if split is "train".
    scene_length (int or None): The length of each scene. Required if split is "validation" or "val".
    split (str): The type of split, one of "train", "validation", "val", or "test".

    Returns:
    dict or list: The generated or provided start indices for each movie.

    Raises:
    AssertionError: If chunk_size is not provided for training split when start_indices is None.
    AssertionError: If scene_length is not provided for validation split when start_indices is None.
    NotImplementedError: If start_indices is None and split is not one of "train", "validation", "val", or "test".
    """

    def get_chunking_interval(split_name):
        if split_name == "train":
            assert chunk_size is not None, "Chunk size or start indices must be provided for training."
            return chunk_size
        elif split_name in {"validation", "val"}:
            assert scene_length is not None, "Scene length or start indices must be provided for validation."
            return scene_length
        elif split_name == "test":
            return None
        else:
            raise NotImplementedError("Start indices could not be recovered.")

    interval = get_chunking_interval(split)

    if isinstance(movies, dict):
        if split == "test":
            return {k: [0] for k in movies.keys()}
        return {k: np.arange(0, movies[k].shape[1], interval).tolist() for k in movies.keys()}
    else:
        if split == "test":
            return [0]
        return np.arange(0, movies.shape[1], interval).tolist()


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


def get_io_dims(data_loader) -> dict[str, tuple[int, ...]] | tuple:
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
    return default_collate(filtered_batch) if filtered_batch else None


def extract_data_info_from_dataloaders(
    dataloaders: dict[str, dict[str, Any]] | dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Extracts the data_info dictionary from the provided dataloaders.
    Args:
        dataloaders: A dictionary of dataloaders for different sessions.
    Returns:
        data_info: A dictionary containing input_dimensions, input_channels, and output_dimension for each session,
                   nested with these attributes as the first level keys and sessions as the second level.
    """
    # Ensure train loader is used if available and not provided directly
    dataloaders = dataloaders.get("train", dataloaders)

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name, *_ = next(iter(list(dataloaders.values())[0]))._fields  # type: ignore

    # Get the input and output dimensions for each session
    session_shape_dict = get_dims_for_loader_dict(dataloaders)

    # Initialize the new structure
    data_info: dict[str, Dict[str, Any]] = {k: {} for k in session_shape_dict.keys()}

    # Populate the new structure
    for session_key, shapes in session_shape_dict.items():
        data_info[session_key]["input_dimensions"] = shapes[in_name]
        data_info[session_key]["input_channels"] = shapes[in_name][1]
        data_info[session_key]["output_dimension"] = shapes[out_name][-1]
        data_info[session_key]["mean_response"] = np.array(dataloaders[session_key].dataset.mean_response)  # type: ignore
        data_info[session_key]["roi_coords"] = np.array(dataloaders[session_key].dataset.roi_coords)  # type: ignore

    return data_info
