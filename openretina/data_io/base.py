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
    """
    Container for stimulus movies used during training and evaluation.

    Attributes:
        train: Continuous movie shown during training.
        test_dict: Named dictionary of frozen test stimuli. For legacy single-test datasets
            pass `test`; it will automatically be wrapped into `{"test": test}`.
        test: Convenience field to pass a single frozen movie.
        stim_id: Optional identifier (e.g. "natural") to keep responses/movies aligned.
        random_sequences: Optional clip permutations (Höfling 2024 format).
        norm_mean / norm_std: Normalization statistics applied to both train and test movies.
    """

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
    """
    Container for neural responses paired with `MoviesTrainTestSplit`.

    Supports multiple test stimuli via `test_dict` and per-trial traces via
    `test_by_trial_dict`. For single-test datasets you may provide `test` and
    optionally `test_by_trial`; both will be lifted into the matching dictionaries.
    """

    train: Float[np.ndarray, "neurons train_time"]
    # test_dict: dict[str, Float[np.ndarray, "neurons test_time"]]
    # (dataclass and Omegaconf complain about unsupported value type when using full annotation)
    test_dict: dict = field(default_factory=lambda: {})
    test: InitVar[Float[np.ndarray, "neurons test_time"] | None] = None
    test_by_trial: Float[np.ndarray, "trials neurons test_time"] | None = None
    test_by_trial_dict: dict = field(default_factory=lambda: {})
    stim_id: str | None = None
    session_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self, test):
        if (len(self.test_dict) == 0) == (test is None):
            raise ValueError(f"Exactly one of test_dict and test should be set, but {test=} {self.test_dict=}.")
        if len(self.test_dict) == 0:
            self.test_dict["test"] = test

        # When only a single test stimulus exists we can lift the array into a dict automatically.
        if self.test_by_trial is not None:
            if len(self.test_dict) > 1:
                raise ValueError(
                    "Provide test_by_trial_dict when multiple test stimuli are present to keep keys disambiguated."
                )
            key = next(iter(self.test_dict.keys()))
            self.test_by_trial_dict.setdefault(key, self.test_by_trial)

        extra_trial_keys = set(self.test_by_trial_dict.keys()) - set(self.test_dict.keys())
        if extra_trial_keys:
            raise ValueError(
                f"test_by_trial_dict contains keys without matching test stimuli: {sorted(extra_trial_keys)}"
            )

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

    def get_test_by_trial(self, name: str = "test") -> Float[np.ndarray, "trials neurons test_time"] | None:
        """
        Return the per-trial responses for a specific stimulus.

        Args:
            name: Key inside `test_dict`. Default is "test", for the default single test stimulus case.

        Returns:
            Array of shape (trials, neurons, time) if available, otherwise `None`.
        """
        if not self.test_by_trial_dict:
            return None
        return self.test_by_trial_dict.get(name)


def get_n_neurons_per_session(responses_dict: dict[str, ResponsesTrainTestSplit]) -> dict[str, int]:
    return {name: responses.n_neurons for name, responses in responses_dict.items()}


@dataclass
class DatasetStatistics:
    """Statistics about unique frames across sessions.

    Attributes:
        unique_train_frames: Number of unique training frames (0 if train/val split unknown).
        unique_val_frames: Number of unique validation frames (0 if train/val split unknown).
        unique_train_val_frames: Total unique train+val frames (always computed).
        unique_test_frames: Dict mapping test stimulus name to unique frame count.
        n_sessions: Total number of sessions.
        n_unique_stimuli: Number of unique stimulus IDs (sessions sharing stimuli counted once).
    """

    unique_train_frames: int
    unique_val_frames: int
    unique_train_val_frames: int
    unique_test_frames: dict[str, int]
    n_sessions: int
    n_unique_stimuli: int


def _compute_movie_fingerprint(movie: np.ndarray, n_samples: int = 10) -> tuple[float, ...]:
    """
    Compute a fingerprint from a movie by sampling frames and computing their mean values.

    This provides a fast heuristic to identify identical movies without comparing all frames.
    Two movies with the same fingerprint are very likely identical.

    Args:
        movie: Movie array with shape (channels, time, height, width).
        n_samples: Number of evenly spaced frames to sample.

    Returns:
        Tuple of mean values from sampled frames, suitable as a dict key.
    """
    time_dim = movie.shape[1]
    if time_dim == 0:
        return ()

    # Sample evenly spaced frame indices
    n_samples = min(n_samples, time_dim)
    indices = np.linspace(0, time_dim - 1, n_samples, dtype=int)

    # Compute mean of each sampled frame (across channels and spatial dims)
    fingerprint = tuple(float(movie[:, idx, :, :].mean()) for idx in indices)
    return fingerprint


def compute_unique_frame_counts(
    movies_dict: dict[str, MoviesTrainTestSplit] | MoviesTrainTestSplit,
    clip_length: int | None = None,
    num_val_clips: int | None = None,
    train_frac: float | None = None,
) -> DatasetStatistics:
    """
    Compute unique frame counts across sessions.

    Sessions with the same stim_id share the same stimulus and are counted once.
    If stim_id is not set, sessions are compared using a fingerprint computed from
    sampled frames (mean values of ~10 evenly spaced frames).

    Supports two methods for computing train/val split:
    1. Clip-based: Provide `clip_length` and `num_val_clips` (e.g., hoefling, karamanlis)
    2. Fraction-based: Provide `train_frac` (e.g., sridhar with trial-wise splitting)

    Args:
        movies_dict: Dictionary mapping session names to MoviesTrainTestSplit, or a single
            MoviesTrainTestSplit (for datasets where all sessions share the same stimulus).
        clip_length: Length of each clip in frames. Used with num_val_clips for clip-based split.
        num_val_clips: Number of validation clips. Used with clip_length for clip-based split.
        train_frac: Fraction of data used for training (e.g., 0.8). Used for trial-wise splitting.

    Returns:
        DatasetStatistics with unique frame counts.
    """
    # Handle single MoviesTrainTestSplit (e.g., hoefling_2024 where all sessions share one stimulus)
    if isinstance(movies_dict, MoviesTrainTestSplit):
        movies_dict = {"_shared": movies_dict}

    # Group sessions by stimulus identity
    # Key: stim_id, or (shape, fingerprint) tuple as fallback
    stim_groups: dict[str | tuple[tuple[int, ...], tuple[float, ...]], MoviesTrainTestSplit] = {}

    for session_name, movies in movies_dict.items():
        if movies.stim_id is not None:
            key: str | tuple[tuple[int, ...], tuple[float, ...]] = movies.stim_id
        else:
            # Fallback: use shape + frame fingerprint to identify unique movies
            # This catches cases where different movies have the same shape
            shape = movies.train.shape
            fingerprint = _compute_movie_fingerprint(movies.train)
            key = (shape, fingerprint)

        if key not in stim_groups:
            stim_groups[key] = movies

    # Compute unique frames across unique stimuli
    unique_train_val_frames = 0
    unique_test_frames: dict[str, int] = {}

    for movies in stim_groups.values():
        unique_train_val_frames += movies.train.shape[1]

        for test_name, test_movie in movies.test_dict.items():
            if test_name not in unique_test_frames:
                unique_test_frames[test_name] = 0
            unique_test_frames[test_name] += test_movie.shape[1]

    # Split train/val using one of two methods:
    # 1. Clip-based (clip_length + num_val_clips)
    # 2. Fraction-based (train_frac)
    if clip_length is not None and num_val_clips is not None:
        # Clip-based split (hoefling, karamanlis, maheswaranathan)
        unique_val_frames = 0
        unique_train_frames = 0
        for movies in stim_groups.values():
            num_clips = movies.train.shape[1] // clip_length
            val_frames = num_val_clips * clip_length
            train_frames = (num_clips - num_val_clips) * clip_length
            unique_val_frames += val_frames
            unique_train_frames += train_frames
    elif train_frac is not None:
        # Fraction-based split (sridhar with trial-wise splitting)
        # This is an estimate based on the fraction; actual split depends on n_trials
        unique_train_frames = int(unique_train_val_frames * train_frac)
        unique_val_frames = unique_train_val_frames - unique_train_frames
    else:
        unique_train_frames = 0
        unique_val_frames = 0

    return DatasetStatistics(
        unique_train_frames=unique_train_frames,
        unique_val_frames=unique_val_frames,
        unique_train_val_frames=unique_train_val_frames,
        unique_test_frames=unique_test_frames,
        n_sessions=len(movies_dict),
        n_unique_stimuli=len(stim_groups),
    )


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

    # Compute mean activity for each session from training responses
    mean_activity_dict = {}
    for session_name, responses in neuron_data_dictionary.items():
        # responses.train has shape (n_neurons, n_timepoints)
        # Compute mean across time dimension
        mean_activity = torch.tensor(responses.train.mean(axis=1), dtype=torch.float32)
        mean_activity_dict[session_name] = mean_activity

    if isinstance(movies_dictionary, MoviesTrainTestSplit):
        stim_mean = movies_dictionary.norm_mean
        stim_std = movies_dictionary.norm_std
        input_shape = (
            movies_dictionary.train.shape[0],
            *movies_dictionary.train.shape[2:],
        )
    else:
        norm_means = [movie.norm_mean for movie in movies_dictionary.values() if movie.norm_mean is not None]
        norm_stds = [movie.norm_std for movie in movies_dictionary.values() if movie.norm_std is not None]

        if len(norm_means) > 0:
            if not np.allclose(norm_means, norm_means[0], atol=1, rtol=0):
                raise ValueError(f"Normalization means are not consistent across stimuli: {norm_means}")
            stim_mean = norm_means[0]
        else:
            stim_mean = 0.0
            warnings.warn(f"No stimulus mean set, setting {stim_mean=}")
        if len(norm_stds) > 0:
            if not np.allclose(norm_stds, norm_stds[0], atol=1, rtol=0):
                raise ValueError(f"Normalization stds are not consistent across stimuli: {norm_stds}")
            stim_std = norm_stds[0]
        else:
            stim_std = 1.0
            warnings.warn(f"No stimulus stds set, setting {stim_std=}")

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
        "mean_activity_dict": mean_activity_dict,
        "input_shape": input_shape,
        "sessions_kwargs": sessions_kwargs,
        "stim_mean": stim_mean,
        "stim_std": stim_std,
        **(partial_data_info or {}),
    }
