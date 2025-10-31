"""
OpenRetina Data Loading Package

This package contains pure numpy-based data loading functionality for the OpenRetina project.
It provides data structures and loading utilities without PyTorch dependencies.
"""

from .base import MoviesTrainTestSplit, ResponsesTrainTestSplit, compute_data_info, get_n_neurons_per_session, normalize_train_test_movies
from .artificial_stimuli import normalize_stimulus, discretize_triggers, load_chirp, load_moving_bar
from .h5_dataset_reader import load_stimuli, load_responses

__all__ = [
    "MoviesTrainTestSplit",
    "ResponsesTrainTestSplit", 
    "compute_data_info",
    "get_n_neurons_per_session",
    "normalize_train_test_movies",
    "normalize_stimulus",
    "discretize_triggers", 
    "load_chirp",
    "load_moving_bar",
    "load_stimuli",
    "load_responses",
]