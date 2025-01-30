import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from jaxtyping import Float


@dataclass
class MoviesTrainTestSplit:
    train: Float[np.ndarray, "channels train_time height width"]
    test: Float[np.ndarray, "channels test_time height width"]
    stim_id: Optional[str] = None
    random_sequences: Optional[np.ndarray] = None
    norm_mean: Optional[float] = None
    norm_std: Optional[float] = None

    def __post_init__(self):
        assert self.train.ndim == 4, "Train movie should have 4 dimensions."
        assert self.test.ndim == 4, "Test movie should have 4 dimensions."
        assert self.train.shape[0] == self.test.shape[0], "Channel dimension does not match in train and test movies."
        assert self.train.shape[2:] == self.test.shape[2:], "Spatial dimensions do not match in train and test movies."
        if self.train.shape[0] > self.train.shape[1]:
            warnings.warn(
                "The number of channels is greater than the number of timebins in the train movie. "
                "Check if the provided data is in the correct format.",
                category=UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_pickle(cls, file_path: str | Path):
        with open(file_path, "rb") as f:
            movies_dict = pickle.load(f)
        return cls(
            train=movies_dict["train"],
            test=movies_dict["test"],
            random_sequences=movies_dict.get("random_sequences", None),
            norm_mean=movies_dict.get("movie_stats", {}).get("dichromatic", {}).get("mean", np.array([np.nan])).item(),
            norm_std=movies_dict.get("movie_stats", {}).get("dichromatic", {}).get("sd", np.array([np.nan])).item(),
        )


@dataclass
class ResponsesTrainTestSplit:
    train: Float[np.ndarray, "neurons train_time"]
    test: Float[np.ndarray, "neurons test_time"]
    test_by_trial: Optional[Float[np.ndarray, "trials neurons test_time"]] = None
    stim_id: Optional[str] = None
    session_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert self.train.shape[0] == self.test.shape[0], (
            "Train and test responses should have the same number of neurons."
        )
        if self.train.shape[0] > self.train.shape[1]:
            warnings.warn(
                "The number of neurons is greater than the number of timebins in the train responses. "
                "Check if the provided data is in the correct format.",
                category=UserWarning,
                stacklevel=2,
            )

    def check_matching_stimulus(self, movies: MoviesTrainTestSplit):
        assert self.train.shape[1] == movies.train.shape[1], (
            "Train movie and responses should have the same timebins."
            f"Got {self.train.shape[1]} and {movies.train.shape[1]}."
        )
        assert self.test.shape[1] == movies.test.shape[1], (
            "Test movie and responses should have the same timebins.",
            f"Got {self.test.shape[1]} and {movies.test.shape[1]}.",
        )
        if self.stim_id is not None and movies.stim_id is not None:
            assert self.stim_id == movies.stim_id, "Stimulus ID in responses and movies do not match."

    @property
    def n_neurons(self) -> int:
        return self.train.shape[0]


def get_n_neurons_per_session(responses_dict: dict[str, ResponsesTrainTestSplit]) -> dict[str, int]:
    return {name: responses.n_neurons for name, responses in responses_dict.items()}


def normalize_train_test_movies(
    train: Float[np.ndarray, "channels train_time height width"],
    test: Float[np.ndarray, "channels test_time height width"],
) -> tuple[
    Float[np.ndarray, "channels train_time height width"],
    Float[np.ndarray, "channels test_time height width"],
    dict[str, float | None],
]:
    """
    z-score normalization of train and test movies using the mean and standard deviation of the train movie.

    Parameters:
    - train: train movie with shape (channels, time, height, width)
    - test: test movie with shape (channels, time, height, width)

    Returns:
    - train_video_preproc: normalized train movie
    - test_video_preproc: normalized test movie
    - norm_stats: dictionary containing the mean and standard deviation of the train movie

    Note: The functions casts the input to torch tensors to calculate the mean and standard deviation of large
    inputs more efficiently.
    """
    train_tensor = torch.tensor(train, dtype=torch.float32)
    test_tensor = torch.tensor(test, dtype=torch.float32)
    train_mean = train_tensor.mean()
    train_std = train_tensor.std()
    train_video_preproc = (train_tensor - train_mean) / train_std
    test_video = (test_tensor - train_mean) / train_std
    return (
        train_video_preproc.cpu().detach().numpy(),
        test_video.cpu().detach().numpy(),
        {"norm_mean": train_mean.item(), "norm_std": train_std.item()},
    )


def compute_data_info(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: dict[str, MoviesTrainTestSplit] | MoviesTrainTestSplit,
) -> dict[str, Any]:
    """
    Computes information related to the data used to train a model, including the number of neurons, the shape of the
    movies, and the normalization statistics. This information can be fed to and saved with the models.

    Parameters:
    - neuron_data_dictionary: dictionary of responses for each session
    - movies_dictionary: dictionary of movies for each session

    Returns:
    - data_info: dictionary containing the number of neurons, the shape of the movies, the movie normalization
        statistics, and any extra session kwargs related to the data.
    """
    n_neurons_dict = get_n_neurons_per_session(neuron_data_dictionary)
    if isinstance(movies_dictionary, MoviesTrainTestSplit):
        movies_dictionary = {"default": movies_dictionary}
    input_shape = tuple((movie.train.shape[0], *movie.train.shape[2:]) for movie in movies_dictionary.values())[0]
    sessions_kwargs = {
        session_name: responses.session_kwargs for session_name, responses in neuron_data_dictionary.items()
    }

    return {
        "n_neurons_dict": n_neurons_dict,
        "input_shape": input_shape,
        "sessions_kwargs": sessions_kwargs,
        "movie_norm_dict": {
            name: {"norm_mean": movie.norm_mean, "norm_std": movie.norm_std}
            for name, movie in movies_dictionary.items()
        },
    }
