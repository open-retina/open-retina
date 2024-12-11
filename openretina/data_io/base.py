import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jaxtyping import Float


@dataclass
class MoviesTrainTestSplit:
    train: Float[np.ndarray, "channels train_time height width"]
    test: Float[np.ndarray, "channels test_time height width"]
    stim_id: Optional[str] = None
    random_sequences: Optional[np.ndarray] = None

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
        )


@dataclass
class ResponsesTrainTestSplit:
    train: Float[np.ndarray, "neurons train_time"]
    test: Float[np.ndarray, "neurons test_time"]
    test_by_trial: Optional[Float[np.ndarray, "trials neurons test_time"]] = None
    stim_id: Optional[str] = None
    session_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert (
            self.train.shape[0] == self.test.shape[0]
        ), "Train and test responses should have the same number of neurons."
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
