"""
OpenRetina Data I/O Module

This module provides PyTorch-based data loading functionality that builds on top
of the numpy-based openretina-dataloading package.
"""

# Import numpy-based functionality from the dataloading package
import openretina_dataloading as odl

# Import PyTorch wrappers and dataset classes
from .pytorch_wrappers import (
    PyTorchMoviesTrainTestSplit,
    PyTorchResponsesTrainTestSplit,
    convert_hoefling_response_dict_to_torch,
    convert_hoefling_movies_to_torch,
    normalize_train_test_movies,
    load_stimuli_torch,
    load_responses_torch,
)

from .base_dataloader import (
    MovieDataSet,
    DataPoint,
    generate_movie_splits,
    get_movie_dataloader,
)

from .cyclers import (
    Cycler,
    ShuffleCycler,
    LongCycler,
)

# Re-export commonly used items for backward compatibility
MoviesTrainTestSplit = PyTorchMoviesTrainTestSplit
ResponsesTrainTestSplit = PyTorchResponsesTrainTestSplit

# Re-export numpy-based functions that don't need PyTorch
compute_data_info = odl.compute_data_info
get_n_neurons_per_session = odl.get_n_neurons_per_session

__all__ = [
    # PyTorch wrappers
    "PyTorchMoviesTrainTestSplit",
    "PyTorchResponsesTrainTestSplit", 
    "convert_hoefling_response_dict_to_torch",
    "convert_hoefling_movies_to_torch",
    "normalize_train_test_movies",
    "load_stimuli_torch",
    "load_responses_torch",
    
    # Dataset classes
    "MovieDataSet",
    "DataPoint",
    "generate_movie_splits",
    "get_movie_dataloader",
    
    # Cyclers
    "Cycler",
    "ShuffleCycler",
    "LongCycler",
    
    # Backward compatibility
    "MoviesTrainTestSplit",
    "ResponsesTrainTestSplit",
    "compute_data_info",
    "get_n_neurons_per_session",
]