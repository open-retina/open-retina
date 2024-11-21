import bisect
import pickle
from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler

DataPoint = namedtuple("DataPoint", ["inputs", "targets"])


@dataclass
class MoviesTrainTestSplit:
    train: np.ndarray
    test: np.ndarray
    random_sequences: Optional[np.ndarray]

    @classmethod
    def from_pickle(cls, file_path: str):
        with open(file_path, "rb") as f:
            movies_dict = pickle.load(f)
        return cls(
            train=movies_dict["train"],
            test=movies_dict["test"],
            random_sequences=movies_dict.get("random_sequences", None),
        )


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
        roi_ids: Float[np.ndarray, " n_neurons"] | None,
        roi_coords: Float[np.ndarray, "n_neurons 2"] | None,
        group_assignment: Float[np.ndarray, " n_neurons"] | None,
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

    def __getitem__(self, idx: int | slice) -> DataPoint:
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

    def __len__(self) -> int:
        # Returns the number of chunks of clips and responses used for training
        return self.samples[1].shape[0] // self.chunk_size

    def __str__(self) -> str:
        return (
            f"MovieDataSet with {self.samples[1].shape[1]} neuron responses "
            f"to a movie of shape {list(self.samples[0].shape)}."
        )

    def __repr__(self) -> str:
        return str(self)


def generate_movie_splits(
    movie_train,
    movie_test,
    val_clip_idx: list[int] | None,
    num_clips: int,
    num_val_clips: int,
    clip_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    if val_clip_idx is None:
        val_clip_idx = list(np.sort(np.random.choice(num_clips, num_val_clips, replace=False)))

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
    mask = np.ones(num_clips, dtype=bool)
    mask[val_clip_idx] = False
    train_clip_idx = np.arange(num_clips)[mask]

    movie_train_subset = torch.cat(
        [movie_train[:, i * clip_length : (i + 1) * clip_length] for i in train_clip_idx],
        dim=1,
    )

    return movie_train_subset, movie_val, movie_test, val_clip_idx


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
        scene_length: int,
        allow_over_boundaries: bool = False,
    ):
        super().__init__()
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size
        self.movie_length = movie_length
        self.scene_length = scene_length
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

    def __len__(self) -> int:
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


def handle_missing_start_indices(
    movies: dict | np.ndarray | torch.Tensor, chunk_size: int | None, scene_length: int | None, split: str
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


def get_movie_dataloader(
    movies: np.ndarray | torch.Tensor | dict[int, np.ndarray],
    responses: Float[np.ndarray, "n_frames n_neurons"],
    *,
    split: str | Literal["train", "validation", "val", "test"],
    scene_length: int,
    chunk_size: int,
    batch_size: int,
    start_indices: list[int] | dict[int, list[int]] | None = None,
    roi_ids: Float[np.ndarray, " n_neurons"] | None = None,
    roi_coords: Float[np.ndarray, "n_neurons 2"] | None = None,
    group_assignment: Float[np.ndarray, " n_neurons"] | None = None,
    scan_sequence_idx: int | None = None,
    drop_last: bool = True,
    use_base_sequence: bool = False,
    allow_over_boundaries: bool = True,
    **kwargs,
) -> DataLoader:
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return None

    if not allow_over_boundaries and split == "train" and chunk_size > scene_length:
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
