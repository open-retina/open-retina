import os
import pickle
import warnings
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import Any, Optional

import numpy as np
import torch
from jaxtyping import Float


@dataclass(frozen=True)
class MoviesTrainTestSplit:
    train: Float[np.ndarray, "channels train_time height width"]
    # test_dict: dict[str, Float[np.ndarray, "channels train_time height width"]]
    # (dataclass complains about unsupported value type when using full annotation)
    test_dict: dict = field(default_factory=lambda: {})
    test: InitVar[Float[np.ndarray, "channels test_time height width"] | None] = None
    stim_id: Optional[str] = None
    random_sequences: Optional[np.ndarray] = None
    norm_mean: Optional[float] = None
    norm_std: Optional[float] = None

    def __post_init__(self, test: Float[np.ndarray, "channels test_time height width"] | None):
        if (len(self.test_dict) == 0) == (test is None):
            raise ValueError(f"Exactly one of test_dict and test should be set, but {test=} {self.test_dict=}.")
        if len(self.test_dict) == 0:
            self.test_dict["test"] = test

        assert self.train.ndim == 4, "Train movie should have 4 dimensions."
        assert len(self.test_shape) == 4, "Test movie should have 4 dimensions."
        assert self.train.shape[0] == self.test_shape[0], "Channel dimension does not match in train and test movies."
        assert self.train.shape[2:] == self.test_shape[2:], "Spatial dimensions do not match in train and test movies."
        if self.train.shape[0] > self.train.shape[1]:
            warnings.warn(
                "The number of channels is greater than the number of timebins in the train movie. "
                "Check if the provided data is in the correct format.",
                category=UserWarning,
                stacklevel=2,
            )

    @cached_property
    def test_shape(self) -> tuple[int, int, int, int]:
        for n, t in self.test_dict.items():
            if t.ndim != 4:
                raise ValueError(f"Test stimulus {n} is not 4 dimensional: {t.shape}")
            max_temp_dim = max(t.shape[1] for t in self.test_dict.values())
            test_shapes = {(t.shape[0], max_temp_dim, t.shape[2], t.shape[3]) for t in self.test_dict.values()}
            if len(test_shapes) > 1:
                raise ValueError(f"Inconsistent test shapes: {test_shapes=}")
            return next(iter(test_shapes))
        raise ValueError("No test stimuli present.")

    @property
    def test_movie(self) -> np.ndarray:
        if len(self.test_dict) > 1:
            raise ValueError(f"Multiple test responses present: {list(self.test_dict.keys())}")
        return self.test_dict[next(iter(self.test_dict.keys()))]

    @classmethod
    def from_pickle(cls, file_path: str | os.PathLike):
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
    # test_dict: dict[str, Float[np.ndarray, "neurons test_time"]]
    # (dataclass complains about unsupported value type when using full annotation)
    test_dict: dict = field(default_factory=lambda: {})
    test: InitVar[Float[np.ndarray, "neurons test_time"] | None] = None
    test_by_trial: Float[np.ndarray, "trials neurons test_time"] | None = None
    stim_id: str | None = None
    session_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self, test):
        if (len(self.test_dict) == 0) == (test is None):
            raise ValueError(f"Exactly one of test_dict and test should be set, but {test=} {self.test_dict=}.")
        if len(self.test_dict) == 0:
            self.test_dict["test"] = test

        assert self.train.shape[0] == self.test_neurons, (
            "Train and test responses should have the same number of neurons."
        )
        if self.train.shape[0] > self.train.shape[1]:
            warnings.warn(
                "The number of neurons is greater than the number of timebins in the train responses. "
                "Check if the provided data is in the correct format.",
                category=UserWarning,
                stacklevel=2,
            )

    @cached_property
    def test_neurons(self) -> int:
        for name, t in self.test_dict.items():
            if t.ndim != 2:
                raise ValueError(f"Test responses for {name=} are not two dimensions: {t.shape}")
        test_neurons = set(x.shape[0] for x in self.test_dict.values())
        if len(test_neurons) > 1:
            raise ValueError(f"Test responses have inconsistent number of neurons: {test_neurons=}")
        return next(iter(test_neurons))

    def check_matching_stimulus(self, movies: MoviesTrainTestSplit):
        assert self.train.shape[1] == movies.train.shape[1], (
            "Train movie and responses should have the same timebins."
            f"Got {self.train.shape[1]} and {movies.train.shape[1]}."
        )
        assert set(self.test_dict.keys()) == set(movies.test_dict.keys()), "Test movie and responses should match"
        for k in self.test_dict.keys():
            responses = self.test_dict[k]
            movie = movies.test_dict[k]
            assert responses.shape[1] == movie.shape[1], (
                "Test movie and responses should have the same timebins.",
                f"Got {movie.shape[1]=} and {responses.shape[1]}.",
            )
        if self.stim_id is not None and movies.stim_id is not None:
            assert self.stim_id == movies.stim_id, "Stimulus ID in responses and movies do not match."

    @property
    def n_neurons(self) -> int:
        return self.train.shape[0]

    @property
    def test_response(self) -> Float[np.ndarray, "neurons test_time"]:
        if len(self.test_dict) > 1:
            raise ValueError(f"Multiple test stimuli: {list(self.test_dict.keys())}")
        return self.test_dict[next(iter(self.test_dict.keys()))]


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
    partial_data_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Computes information related to the data used to train a model, including the number of neurons, the shape of the
    movies, and the normalization statistics. This information should be fed to and saved with the models.

    Parameters:
    - neuron_data_dictionary: dictionary of responses for each session
    - movies_dictionary: dictionary of movies for each session
    - partial_data_info: dictionary of partial data info from the config, to be merged with the computed data info

    Returns:
    - data_info: dictionary containing various data info useful for downstream tasks, including the number of neurons,
    the shape of the movies, the movie normalization statistics, and any extra session kwargs related to the data,
    including partial data information passed in the training config.
    """
    n_neurons_dict = get_n_neurons_per_session(neuron_data_dictionary)
    if isinstance(movies_dictionary, MoviesTrainTestSplit):
        stim_mean = movies_dictionary.norm_mean
        stim_std = movies_dictionary.norm_std
        input_shape = (
            movies_dictionary.train.shape[0],
            *movies_dictionary.train.shape[2:],
        )
    else:
        norm_means = [movie.norm_mean for movie in movies_dictionary.values()]
        norm_stds = [movie.norm_std for movie in movies_dictionary.values()]

        if any(mean != norm_means[0] for mean in norm_means):
            raise ValueError(f"Normalization means are not consistent across stimuli: {norm_means}")
        if any(std != norm_stds[0] for std in norm_stds):
            raise ValueError(f"Normalization stds are not consistent across stimuli: {norm_stds}")

        stim_mean = norm_means[0]
        stim_std = norm_stds[0]

        # Do the same for the input shape
        input_shapes = [(movie.train.shape[0], *movie.train.shape[2:]) for movie in movies_dictionary.values()]
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError(f"Input shapes are not consistent across stimuli: {input_shapes}")

        input_shape = input_shapes[0]

    sessions_kwargs = {
        session_name: responses.session_kwargs for session_name, responses in neuron_data_dictionary.items()
    }

    return {
        "n_neurons_dict": n_neurons_dict,
        "input_shape": input_shape,
        "sessions_kwargs": sessions_kwargs,
        "stim_mean": stim_mean,
        "stim_std": stim_std,
        **(partial_data_info or {}),
    }
