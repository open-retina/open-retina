"""
PyTorch wrappers for the numpy-based dataloading functionality.

This module provides PyTorch-specific functionality like DataLoaders, Datasets,
and tensor conversions that build on top of the numpy-based dataloading package.
"""

import torch
from typing import Dict, Any, Optional, Literal
import numpy as np
from jaxtyping import Float

# Import from the new dataloading package
import openretina_dataloading as odl


class PyTorchMoviesTrainTestSplit:
    """
    PyTorch wrapper for MoviesTrainTestSplit that provides tensor conversion.
    """
    
    def __init__(self, numpy_split: odl.MoviesTrainTestSplit):
        self._numpy_split = numpy_split
    
    @property
    def train(self) -> torch.Tensor:
        """Convert train data to PyTorch tensor"""
        return torch.tensor(self._numpy_split.train, dtype=torch.float32)
    
    @property
    def test(self) -> torch.Tensor:
        """Convert test data to PyTorch tensor"""
        return torch.tensor(self._numpy_split.test, dtype=torch.float32)
    
    @property
    def norm_mean(self) -> Optional[torch.Tensor]:
        """Convert normalization mean to PyTorch tensor"""
        if self._numpy_split.norm_mean is not None:
            return torch.tensor(self._numpy_split.norm_mean, dtype=torch.float32)
        return None
    
    @property
    def norm_std(self) -> Optional[torch.Tensor]:
        """Convert normalization std to PyTorch tensor"""
        if self._numpy_split.norm_std is not None:
            return torch.tensor(self._numpy_split.norm_std, dtype=torch.float32)
        return None
    
    @property
    def stim_id(self) -> Optional[str]:
        """Get stimulus ID"""
        return self._numpy_split.stim_id
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary with PyTorch tensors"""
        result = {
            "train": self.train,
            "test": self.test,
        }
        if self.norm_mean is not None:
            result["norm_mean"] = self.norm_mean
        if self.norm_std is not None:
            result["norm_std"] = self.norm_std
        return result


class PyTorchResponsesTrainTestSplit:
    """
    PyTorch wrapper for ResponsesTrainTestSplit that provides tensor conversion.
    """
    
    def __init__(self, numpy_split: odl.ResponsesTrainTestSplit):
        self._numpy_split = numpy_split
    
    @property
    def train(self) -> torch.Tensor:
        """Convert train responses to PyTorch tensor"""
        return torch.tensor(self._numpy_split.train, dtype=torch.float32)
    
    @property
    def test_response(self) -> torch.Tensor:
        """Convert test responses to PyTorch tensor"""
        return torch.tensor(self._numpy_split.test_response, dtype=torch.float32)
    
    @property
    def test_by_trial(self) -> Optional[torch.Tensor]:
        """Convert test by trial responses to PyTorch tensor"""
        if self._numpy_split.test_by_trial is not None:
            return torch.tensor(self._numpy_split.test_by_trial, dtype=torch.float32)
        return None
    
    @property
    def stim_id(self) -> Optional[str]:
        """Get stimulus ID"""
        return self._numpy_split.stim_id
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary with PyTorch tensors"""
        result = {
            "train": self.train,
            "test_response": self.test_response,
        }
        if self.test_by_trial is not None:
            result["test_by_trial"] = self.test_by_trial
        return result


def convert_hoefling_response_dict_to_torch(response_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert Hoefling response dictionary from numpy to PyTorch tensors.
    
    Args:
        response_dict: Dictionary with numpy arrays
        
    Returns:
        Dictionary with PyTorch tensors
    """
    return {
        "train": torch.tensor(response_dict["train"], dtype=torch.float32),
        "validation": torch.tensor(response_dict["validation"], dtype=torch.float32),
        "test": {
            "avg": torch.tensor(response_dict["test"]["avg"], dtype=torch.float32),
            "by_trial": torch.tensor(response_dict["test"]["by_trial"], dtype=torch.float32) 
                       if response_dict["test"]["by_trial"] is not None else None,
        },
    }


def convert_hoefling_movies_to_torch(movies_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Hoefling movies dictionary from numpy to PyTorch tensors.
    
    Args:
        movies_dict: Dictionary with numpy arrays
        
    Returns:
        Dictionary with PyTorch tensors
    """
    result = {
        "left": {
            "train": {},
            "validation": torch.tensor(movies_dict["left"]["validation"], dtype=torch.float32),
            "test": torch.tensor(movies_dict["left"]["test"], dtype=torch.float32),
        },
        "right": {
            "train": {},
            "validation": torch.tensor(movies_dict["right"]["validation"], dtype=torch.float32),
            "test": torch.tensor(movies_dict["right"]["test"], dtype=torch.float32),
        },
        "val_clip_idx": movies_dict["val_clip_idx"],
    }
    
    # Convert training sequences
    for sequence_idx, sequence_data in movies_dict["left"]["train"].items():
        result["left"]["train"][sequence_idx] = torch.tensor(sequence_data, dtype=torch.float32)
    
    for sequence_idx, sequence_data in movies_dict["right"]["train"].items():
        result["right"]["train"][sequence_idx] = torch.tensor(sequence_data, dtype=torch.float32)
    
    return result


# Convenience functions for backward compatibility
def normalize_train_test_movies(
    train_movies: np.ndarray, 
    test_movies: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Normalize movies and return PyTorch tensors.
    
    This is a convenience wrapper that calls the numpy version and converts to PyTorch.
    """
    train_norm, test_norm, stats = odl.normalize_train_test_movies(train_movies, test_movies)
    
    return (
        torch.tensor(train_norm, dtype=torch.float32),
        torch.tensor(test_norm, dtype=torch.float32),
        {
            "norm_mean": torch.tensor(stats["norm_mean"], dtype=torch.float32),
            "norm_std": torch.tensor(stats["norm_std"], dtype=torch.float32),
        }
    )


def load_stimuli_torch(
    file_path: str,
    stim_id: str,
    normalize: bool = True
) -> PyTorchMoviesTrainTestSplit:
    """
    Load stimuli and return PyTorch wrapper.
    
    Args:
        file_path: Path to stimuli file
        stim_id: Stimulus ID
        normalize: Whether to normalize
        
    Returns:
        PyTorch wrapper for movies
    """
    numpy_movies = odl.load_stimuli(file_path, stim_id, normalize)
    return PyTorchMoviesTrainTestSplit(numpy_movies)


def load_responses_torch(
    file_path: str,
    stim_id: str
) -> PyTorchResponsesTrainTestSplit:
    """
    Load responses and return PyTorch wrapper.
    
    Args:
        file_path: Path to responses file
        stim_id: Stimulus ID
        
    Returns:
        PyTorch wrapper for responses
    """
    numpy_responses = odl.load_responses(file_path, stim_id)
    return PyTorchResponsesTrainTestSplit(numpy_responses)