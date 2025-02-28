import bisect
import collections
from collections import namedtuple
from typing import Any, List, Literal, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit

DataPoint = namedtuple("DataPoint", ["inputs", "targets"])


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
        responses: Float[np.ndarray | torch.Tensor, "n_frames n_neurons"],
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
    movie_test: dict[str, np.ndarray],
    val_clip_idc: list[int],
    num_clips: int,
    clip_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    movie_train = torch.tensor(movie_train, dtype=torch.float)
    movie_test_dict = {n: torch.tensor(movie, dtype=torch.float) for n, movie in movie_test.items()}

    channels, _, px_y, px_x = movie_train.shape

    # Prepare validation movie data
    movie_val = torch.zeros((channels, len(val_clip_idc) * clip_length, px_y, px_x), dtype=torch.float)
    for i, idx in enumerate(val_clip_idc):
        movie_val[:, i * clip_length : (i + 1) * clip_length, ...] = movie_train[
            :, idx * clip_length : (idx + 1) * clip_length, ...
        ]

    # Create a boolean mask to indicate which clips are not part of the validation set
    mask = np.ones(num_clips, dtype=bool)
    mask[val_clip_idc] = False
    train_clip_idx = np.arange(num_clips)[mask]

    movie_train_subset = torch.cat(
        [movie_train[:, i * clip_length : (i + 1) * clip_length] for i in train_clip_idx],
        dim=1,
    )

    return movie_train_subset, movie_val, movie_test_dict


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
    movie_length: int, chunk_size: int | None, scene_length: int | None, split: str
) -> list[int]:
    """
    Handle missing start indices for different splits of the dataset.

    Parameters:
    movies (np.ndarray or torch.Tensor): The movies data, as an array.
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

    if split == "train":
        assert chunk_size is not None, "Chunk size or start indices must be provided for training."
        interval = chunk_size
    elif split in {"validation", "val"}:
        assert scene_length is not None, "Scene length or start indices must be provided for validation."
        interval = scene_length
    elif split == "test":
        interval = movie_length
    else:
        raise NotImplementedError("Start indices could not be recovered.")

    return np.arange(0, movie_length, interval).tolist()  # type: ignore


def get_movie_dataloader(
    movie: Float[np.ndarray | torch.Tensor, "n_channels n_frames h w"],
    responses: Float[np.ndarray | torch.Tensor, "n_frames n_neurons"],
    *,
    split: str | Literal["train", "validation", "val", "test"],
    scene_length: int,
    chunk_size: int,
    batch_size: int,
    start_indices: list[int] | None = None,
    roi_ids: Float[np.ndarray, " n_neurons"] | None = None,
    roi_coords: Float[np.ndarray, "n_neurons 2"] | None = None,
    group_assignment: Float[np.ndarray, " n_neurons"] | None = None,
    drop_last: bool = True,
    allow_over_boundaries: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for processing movie data and associated responses.
    This function prepares the dataset and sampler for training or evaluation based on the specified parameters.

    Args:
        movie (Float[np.ndarray | torch.Tensor, "n_channels n_frames h w"]):
            The movie data represented as a multi-dimensional array or tensor.
        responses (Float[np.ndarray, "n_frames n_neurons"]):
            The responses corresponding to the frames of the movie.
        split (str | Literal["train", "validation", "val", "test"]):
            The dataset split to use (train, validation, or test).
        scene_length (int):
            The length of the scene to be processed.
        chunk_size (int):
            The size of each chunk to be extracted from the movie.
        batch_size (int):
            The number of samples per batch.
        start_indices (list[int] | None, optional):
            The starting indices for each chunk. If None, will be computed.
        roi_ids (Float[np.ndarray, " n_neurons"] | None, optional):
            The region of interest IDs. If None, will not be used.
        roi_coords (Float[np.ndarray, "n_neurons 2"] | None, optional):
            The coordinates of the regions of interest. If None, will not be used.
        group_assignment (Float[np.ndarray, " n_neurons"] | None, optional):
            The group assignments (cell types) for the neurons. If None, will not be used.
        drop_last (bool, optional):
            Whether to drop the last incomplete batch. Defaults to True.
        allow_over_boundaries (bool, optional):
            Whether to allow chunks that exceed the scene boundaries. Defaults to True.
        **kwargs:
            Additional keyword arguments for the DataLoader.

    Returns:
        DataLoader:
            A DataLoader instance configured with the specified dataset and sampler.

    Raises:
        ValueError:
            If `allow_over_boundaries` is False and `chunk_size` exceeds `scene_length` during training.
    """
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return  # type: ignore

    if not allow_over_boundaries and split == "train" and chunk_size > scene_length:
        raise ValueError("Clip chunk size must be smaller than scene length to not exceed clip bounds during training.")

    if start_indices is None:
        start_indices = handle_missing_start_indices(movie.shape[1], chunk_size, scene_length, split)
    dataset = MovieDataSet(movie, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
    sampler = MovieSampler(
        start_indices,
        split,
        chunk_size,
        movie_length=movie.shape[1],
        scene_length=scene_length,
        allow_over_boundaries=allow_over_boundaries,
    )

    return DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, drop_last=split == "train" and drop_last, **kwargs
    )


class NeuronDataSplit:
    def __init__(
        self,
        responses: ResponsesTrainTestSplit,
        val_clip_idx: List[int],
        num_clips: int,
        clip_length: int,
        key: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the NeuronData object.
        Boilerplate class to compute and store neuron data train/test/validation splits before feeding into a dataloader

        Args:
            key (dict): The key information for the neuron data,
                        includes date, exp_num, experimenter, field_id, stim_id.
            responses (ResponsesTrainTestSplit): The train and test responses of neurons.
            val_clip_idx (List[int]): The indices of validation clips.
            num_clips (int): The number of clips.
            clip_length (int): The length of each clip.
            key (dict, optional): Additional key information.
        """
        self.neural_responses = responses
        self.num_neurons = self.neural_responses.n_neurons
        self.key = key
        self.roi_coords = ()
        self.clip_length = clip_length
        self.num_clips = num_clips
        self.val_clip_idx = val_clip_idx

        # Transpose the responses to have the shape (n_timepoints, n_neurons)
        self.responses_train_and_val = self.neural_responses.train.T

        self.responses_train, self.responses_val = self.split_data_train_val()
        self.test_responses_by_trial = np.array([])  # Added for compatibility with Hoefling et al., 2024

    def split_data_train_val(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute validation responses and updated train responses stripped from validation clips.
        Can deal with unsorted validation clip indices, and parallels the way movie validation clips are handled.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated train and validation responses.
        """
        # Initialise validation responses
        base_movie_sorting = np.arange(self.num_clips)

        validation_mask = np.ones_like(self.responses_train_and_val, dtype=bool)
        responses_val = np.zeros([len(self.val_clip_idx) * self.clip_length, self.num_neurons])

        # Compute validation responses and remove sections from training responses
        for i, ind1 in enumerate(self.val_clip_idx):
            grab_index = base_movie_sorting[ind1]
            responses_val[i * self.clip_length : (i + 1) * self.clip_length, :] = self.responses_train_and_val[
                grab_index * self.clip_length : (grab_index + 1) * self.clip_length,
                :,
            ]
            validation_mask[
                (grab_index * self.clip_length) : (grab_index + 1) * self.clip_length,
                :,
            ] = False

        responses_train = self.responses_train_and_val[validation_mask].reshape(-1, self.num_neurons)

        return responses_train, responses_val

    @property
    def response_dict(self) -> dict:
        """
        Create and return a dictionary of neural responses for train, validation, and test datasets.
        """
        return {
            "train": torch.tensor(self.responses_train, dtype=torch.float),
            "validation": torch.tensor(self.responses_val, dtype=torch.float),
            "test": {
                "avg": self.response_dict_test,
                "by_trial": torch.tensor(self.test_responses_by_trial, dtype=torch.float),
            },
        }

    @property
    def response_dict_test(self) -> dict[str, torch.Tensor]:
        return {name: torch.tensor(responses.T) for name, responses in self.neural_responses.test_dict.items()}


def get_movie_splits(
    movie_train,
    movie_test: dict[str, np.ndarray],
    val_clip_idx: list[int],
    num_clips: int,
    clip_length: int,
) -> dict[str, Any]:
    movie_train_subset, movie_val, movie_test_tensor = generate_movie_splits(
        movie_train, movie_test, val_clip_idx, num_clips, clip_length
    )

    movies = {
        "train": movie_train_subset,
        "validation": movie_val,
        "test_dict": movie_test_tensor,
        "val_clip_idx": val_clip_idx,
    }

    return movies


def multiple_movies_dataloaders(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, MoviesTrainTestSplit],
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    clip_length: int = 100,
    num_val_clips: int = 10,
    val_clip_indices: list[int] | None = None,
    allow_over_boundaries: bool = True,
) -> dict[str, dict[str, DataLoader]]:
    """
    Create multiple dataloaders for training, validation, and testing from given neuron and movie data.
    This function ensures that the neuron data and movie data are aligned and generates dataloaders for each session.
    It does not make assumptions about the movies in different sessions to be the same, the same length, composed
    of the same clips or in the same order.

    Args:
        neuron_data_dictionary (dict[str, ResponsesTrainTestSplit]):
            A dictionary containing neuron response data split for training and testing.
        movies_dictionary (dict[str, MoviesTrainTestSplit]):
            A dictionary containing movie data split for training and testing.
        train_chunk_size (int, optional):
            The size of the chunks for training data. Defaults to 50.
        batch_size (int, optional):
            The number of samples per batch. Defaults to 32.
        seed (int, optional):
            The random seed for reproducibility. Defaults to 42.
        clip_length (int, optional):
            The length of each clip. Defaults to 100.
        num_val_clips (int, optional):
            The number of validation clips to draw. Defaults to 10.
        val_clip_indices (list[int], optional): The indices of validation clips to use. If provided, num_val_clips is
                                                ignored. Defaults to None.
        allow_over_boundaries (bool, optional):  Whether to allow selected chunks to go over scene boundaries.

    Returns:
        dict:
            A dictionary containing dataloaders for training, validation, and testing for each session.

    Raises:
        AssertionError:
            If the keys of neuron_data_dictionary and movies_dictionary do not match exactly.
    """
    assert set(neuron_data_dictionary.keys()) == set(movies_dictionary.keys()), (
        "The keys of neuron_data_dictionary and movies_dictionary should match exactly."
    )

    # Initialise dataloaders
    dataloaders: dict[str, Any] = collections.defaultdict(dict)

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
        # Extract all data related to the movies first
        num_clips = movies_dictionary[session_key].train.shape[1] // clip_length

        if val_clip_indices is not None:
            val_clip_idx = val_clip_indices
        else:
            # Draw validation clips based on the random seed
            rnd = np.random.RandomState(seed)
            val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

        all_movies = get_movie_splits(
            movies_dictionary[session_key].train,
            movies_dictionary[session_key].test_dict,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        # Extract all splits from neural data
        neuron_data = NeuronDataSplit(
            responses=session_data,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
        )

        clip_chunk_sizes = {
            "train": train_chunk_size,
            "validation": clip_length,
        }
        # Create dataloaders for each fold
        for fold in ["train", "validation"]:
            dataloaders[fold][session_key] = get_movie_dataloader(
                movie=all_movies[fold],
                responses=neuron_data.response_dict[fold],
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                batch_size=batch_size,
                scene_length=clip_length,
                allow_over_boundaries=allow_over_boundaries,
            )
        # test movies
        for name in all_movies["test_dict"].keys():
            movie = all_movies["test_dict"][name]
            dataloaders[name][session_key] = get_movie_dataloader(
                movie=movie,
                responses=neuron_data.response_dict_test[name],
                split="test",
                chunk_size=movie.shape[1],
                batch_size=batch_size,
                scene_length=clip_length,
                allow_over_boundaries=allow_over_boundaries,
            )

    return dataloaders
