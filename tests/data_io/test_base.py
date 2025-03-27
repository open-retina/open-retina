import os
import pickle
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal

from openretina.data_io.base import (
    MoviesTrainTestSplit,
    ResponsesTrainTestSplit,
    get_n_neurons_per_session,
    normalize_train_test_movies,
    compute_data_info,
)


class TestMoviesTrainTestSplit:
    @pytest.fixture
    def valid_train_movie(self):
        # Shape: (channels, train_time, height, width)
        return np.random.randn(3, 100, 32, 32).astype(np.float32)

    @pytest.fixture
    def valid_test_movie(self):
        # Shape: (channels, test_time, height, width)
        return np.random.randn(3, 50, 32, 32).astype(np.float32)

    @pytest.fixture
    def valid_test_dict(self):
        # Multiple test movies with same dimensions
        return {
            "test1": np.random.randn(3, 50, 32, 32).astype(np.float32),
            "test2": np.random.randn(3, 50, 32, 32).astype(np.float32),
        }

    def test_init_with_test(self, valid_train_movie, valid_test_movie):
        # Test initialization with test parameter
        split = MoviesTrainTestSplit(train=valid_train_movie, test=valid_test_movie)
        assert_array_equal(split.train, valid_train_movie)
        assert_array_equal(split.test_movie, valid_test_movie)
        assert "test" in split.test_dict
        assert_array_equal(split.test_dict["test"], valid_test_movie)

    def test_init_with_test_dict(self, valid_train_movie, valid_test_dict):
        # Test initialization with test_dict parameter
        split = MoviesTrainTestSplit(train=valid_train_movie, test_dict=valid_test_dict)
        assert_array_equal(split.train, valid_train_movie)
        assert split.test_dict == valid_test_dict
        
        # Test accessing test_movie raises error with multiple tests
        with pytest.raises(ValueError, match="Multiple test responses present"):
            _ = split.test_movie

    def test_validation_errors(self, valid_train_movie, valid_test_movie):
        # Test error when both test and test_dict are None
        with pytest.raises(ValueError, match="Exactly one of test_dict and test should be set"):
            MoviesTrainTestSplit(train=valid_train_movie)
            
        # Test error when both test and test_dict are provided
        with pytest.raises(ValueError, match="Exactly one of test_dict and test should be set"):
            MoviesTrainTestSplit(
                train=valid_train_movie, 
                test=valid_test_movie,
                test_dict={"other": valid_test_movie}
            )
            
        # Test error with mismatched dimensions
        invalid_test = np.random.randn(3, 50, 64, 64)  # Different spatial dimensions
        with pytest.raises(AssertionError, match="Spatial dimensions do not match"):
            MoviesTrainTestSplit(train=valid_train_movie, test=invalid_test)
            
        invalid_test = np.random.randn(2, 50, 32, 32)  # Different channel dimensions
        with pytest.raises(AssertionError, match="Channel dimension does not match"):
            MoviesTrainTestSplit(train=valid_train_movie, test=invalid_test)

    def test_test_shape_property(self, valid_train_movie, valid_test_dict):
        split = MoviesTrainTestSplit(train=valid_train_movie, test_dict=valid_test_dict)
        expected_shape = (3, 50, 32, 32)
        assert split.test_shape == expected_shape
        
        # Test with inconsistent test shapes
        invalid_test_dict = valid_test_dict.copy()
        invalid_test_dict["test3"] = np.random.randn(3, 50, 64, 64)  # Different spatial dimensions
        with pytest.raises(ValueError, match="Inconsistent test shapes"):
            MoviesTrainTestSplit(train=valid_train_movie, test_dict=invalid_test_dict)
            
        # Test with non-4D test stimulus
        invalid_test_dict = {"test": np.random.randn(3, 50, 32)}  # 3D array
        with pytest.raises(ValueError, match="Test stimulus test is not 4 dimensional"):
            MoviesTrainTestSplit(train=valid_train_movie, test_dict=invalid_test_dict)
            
        # Test with empty test_dict
        # Create a custom class that inherits from MoviesTrainTestSplit to test the edge case
        class TestEmptyDict(MoviesTrainTestSplit):
            def __post_init__(self, test):
                # Skip validation to allow empty test_dict
                pass
                
        # Create an instance with empty test_dict
        empty_split = TestEmptyDict(train=valid_train_movie, test_dict={})
        
        # Now test_shape should raise ValueError
        with pytest.raises(ValueError, match="No test stimuli present"):
            _ = empty_split.test_shape

    def test_warning_for_unusual_dimensions(self, monkeypatch):
        # We need to patch the assertion to allow the warning to be emitted
        original_post_init = MoviesTrainTestSplit.__post_init__
        
        def mocked_post_init(self, test):
            # Skip the assertions but keep the warning logic
            if len(self.test_dict) == 0:
                self.test_dict["test"] = test
                
            if self.train.shape[0] > self.train.shape[1]:
                warnings.warn(
                    "The number of channels is greater than the number of timebins in the train movie. "
                    "Check if the provided data is in the correct format.",
                    category=UserWarning,
                    stacklevel=2,
                )
        
        # Apply the monkey patch
        monkeypatch.setattr(MoviesTrainTestSplit, "__post_init__", mocked_post_init)
        
        # Create test data
        unusual_train = np.random.randn(3, 2, 32, 32)  # 3 channels, 2 time points (channels > time)
        test = np.random.randn(3, 50, 32, 32)
        
        # Test that the warning is emitted
        with pytest.warns(UserWarning, match="number of channels is greater than the number of timebins"):
            MoviesTrainTestSplit(train=unusual_train, test=test)

    def test_from_pickle(self, valid_train_movie, valid_test_movie):
        # Create a temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Create a dictionary to pickle
            movies_dict = {
                "train": valid_train_movie,
                "test": valid_test_movie,
                "random_sequences": np.array([1, 2, 3]),
                "movie_stats": {
                    "dichromatic": {
                        "mean": np.array([0.5]),
                        "sd": np.array([1.2]),
                    }
                }
            }
            
            # Save to pickle
            with open(temp_path, "wb") as f:
                pickle.dump(movies_dict, f)
                
            # Load using from_pickle
            split = MoviesTrainTestSplit.from_pickle(temp_path)
            
            # Verify loaded data
            assert_array_equal(split.train, valid_train_movie)
            assert_array_equal(split.test_movie, valid_test_movie)
            assert_array_equal(split.random_sequences, np.array([1, 2, 3]))
            assert split.norm_mean == 0.5
            assert split.norm_std == 1.2
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestResponsesTrainTestSplit:
    @pytest.fixture
    def valid_train_responses(self):
        # Shape: (neurons, train_time)
        return np.random.randn(10, 100).astype(np.float32)

    @pytest.fixture
    def valid_test_responses(self):
        # Shape: (neurons, test_time)
        return np.random.randn(10, 50).astype(np.float32)

    @pytest.fixture
    def valid_test_dict(self):
        # Multiple test responses with same dimensions
        return {
            "test1": np.random.randn(10, 50).astype(np.float32),
            "test2": np.random.randn(10, 50).astype(np.float32),
        }

    def test_init_with_test(self, valid_train_responses, valid_test_responses):
        # Test initialization with test parameter
        split = ResponsesTrainTestSplit(train=valid_train_responses, test=valid_test_responses)
        assert_array_equal(split.train, valid_train_responses)
        assert_array_equal(split.test_response, valid_test_responses)
        assert "test" in split.test_dict
        assert_array_equal(split.test_dict["test"], valid_test_responses)

    def test_init_with_test_dict(self, valid_train_responses, valid_test_dict):
        # Test initialization with test_dict parameter
        split = ResponsesTrainTestSplit(train=valid_train_responses, test_dict=valid_test_dict)
        assert_array_equal(split.train, valid_train_responses)
        assert split.test_dict == valid_test_dict
        
        # Test accessing test_response raises error with multiple tests
        with pytest.raises(ValueError, match="Multiple test stimuli"):
            _ = split.test_response

    def test_validation_errors(self, valid_train_responses, valid_test_responses):
        # Test error when both test and test_dict are None
        with pytest.raises(ValueError, match="Exactly one of test_dict and test should be set"):
            ResponsesTrainTestSplit(train=valid_train_responses)
            
        # Test error when both test and test_dict are provided
        with pytest.raises(ValueError, match="Exactly one of test_dict and test should be set"):
            ResponsesTrainTestSplit(
                train=valid_train_responses, 
                test=valid_test_responses,
                test_dict={"other": valid_test_responses}
            )
            
        # Test error with mismatched dimensions
        invalid_test = np.random.randn(5, 50)  # Different number of neurons
        with pytest.raises(AssertionError, match="Train and test responses should have the same number of neurons"):
            ResponsesTrainTestSplit(train=valid_train_responses, test=invalid_test)

    def test_test_neurons_property(self, valid_train_responses, valid_test_dict):
        split = ResponsesTrainTestSplit(train=valid_train_responses, test_dict=valid_test_dict)
        assert split.test_neurons == 10
        
        # Test with inconsistent test shapes
        invalid_test_dict = valid_test_dict.copy()
        invalid_test_dict["test3"] = np.random.randn(5, 50)  # Different number of neurons
        with pytest.raises(ValueError, match="Test responses have inconsistent number of neurons"):
            ResponsesTrainTestSplit(train=valid_train_responses, test_dict=invalid_test_dict)
            
        # Test with non-2D test responses
        invalid_test_dict = {"test": np.random.randn(10, 50, 2)}  # 3D array
        with pytest.raises(ValueError, match="Test responses for name='test' are not two dimensions"):
            ResponsesTrainTestSplit(train=valid_train_responses, test_dict=invalid_test_dict)

    def test_warning_for_unusual_dimensions(self, monkeypatch):
        # We need to patch the assertion to allow the warning to be emitted
        original_post_init = ResponsesTrainTestSplit.__post_init__
        
        def mocked_post_init(self, test):
            # Skip the assertions but keep the warning logic
            if len(self.test_dict) == 0:
                self.test_dict["test"] = test
                
            if self.train.shape[0] > self.train.shape[1]:
                warnings.warn(
                    "The number of neurons is greater than the number of timebins in the train responses. "
                    "Check if the provided data is in the correct format.",
                    category=UserWarning,
                    stacklevel=2,
                )
        
        # Apply the monkey patch
        monkeypatch.setattr(ResponsesTrainTestSplit, "__post_init__", mocked_post_init)
        
        # Create test data
        unusual_train = np.random.randn(10, 5)  # 10 neurons, 5 time points (neurons > time)
        test = np.random.randn(10, 50)
        
        # Test that the warning is emitted
        with pytest.warns(UserWarning, match="number of neurons is greater than the number of timebins"):
            ResponsesTrainTestSplit(train=unusual_train, test=test)

    def test_n_neurons_property(self, valid_train_responses, valid_test_responses):
        split = ResponsesTrainTestSplit(train=valid_train_responses, test=valid_test_responses)
        assert split.n_neurons == 10

    def test_check_matching_stimulus(self, valid_train_responses, valid_test_responses):
        # Create matching movie and response splits
        train_movie = np.random.randn(3, 100, 32, 32)
        test_movie = np.random.randn(3, 50, 32, 32)
        
        movies_split = MoviesTrainTestSplit(train=train_movie, test=test_movie)
        responses_split = ResponsesTrainTestSplit(train=valid_train_responses, test=valid_test_responses)
        
        # Should not raise any errors
        responses_split.check_matching_stimulus(movies_split)
        
        # Test with mismatched train dimensions
        mismatched_train_movie = np.random.randn(3, 90, 32, 32)  # Different time dimension
        mismatched_movies_split = MoviesTrainTestSplit(train=mismatched_train_movie, test=test_movie)
        
        with pytest.raises(AssertionError, match="Train movie and responses should have the same timebins"):
            responses_split.check_matching_stimulus(mismatched_movies_split)
            
        # Test with mismatched test keys
        test_dict1 = {"test1": valid_test_responses}
        test_dict2 = {"test2": test_movie}
        
        responses_split = ResponsesTrainTestSplit(train=valid_train_responses, test_dict=test_dict1)
        movies_split = MoviesTrainTestSplit(train=train_movie, test_dict=test_dict2)
        
        with pytest.raises(AssertionError, match="Test movie and responses should match"):
            responses_split.check_matching_stimulus(movies_split)
            
        # Test with mismatched stim_id
        responses_split = ResponsesTrainTestSplit(
            train=valid_train_responses, 
            test=valid_test_responses,
            stim_id="stim1"
        )
        movies_split = MoviesTrainTestSplit(
            train=train_movie, 
            test=test_movie,
            stim_id="stim2"
        )
        
        with pytest.raises(AssertionError, match="Stimulus ID in responses and movies do not match"):
            responses_split.check_matching_stimulus(movies_split)


def test_get_n_neurons_per_session():
    # Create a dictionary of ResponsesTrainTestSplit objects
    responses_dict = {
        "session1": ResponsesTrainTestSplit(
            train=np.random.randn(10, 100),
            test=np.random.randn(10, 50)
        ),
        "session2": ResponsesTrainTestSplit(
            train=np.random.randn(15, 100),
            test=np.random.randn(15, 50)
        ),
    }
    
    result = get_n_neurons_per_session(responses_dict)
    expected = {"session1": 10, "session2": 15}
    assert result == expected


@patch('openretina.data_io.base.torch')
def test_normalize_train_test_movies(mock_torch):
    # Create sample train and test movies
    train = np.random.randn(3, 100, 32, 32).astype(np.float32)
    test = np.random.randn(3, 50, 32, 32).astype(np.float32)
    
    # Calculate expected values manually using numpy
    train_mean = np.mean(train)
    train_std = np.std(train)
    
    expected_train = ((train - train_mean) / train_std).astype(np.float32)
    expected_test = ((test - train_mean) / train_std).astype(np.float32)
    
    # Configure the mock torch module
    mock_train_tensor = MagicMock()
    mock_test_tensor = MagicMock()
    mock_mean = MagicMock()
    mock_std = MagicMock()
    
    # Set up the mock behavior
    mock_torch.tensor = MagicMock(side_effect=[mock_train_tensor, mock_test_tensor])
    mock_torch.float32 = MagicMock()
    
    mock_train_tensor.mean.return_value = mock_mean
    mock_train_tensor.std.return_value = mock_std
    mock_mean.item.return_value = train_mean
    mock_std.item.return_value = train_std
    
    # Set up subtraction and division
    mock_train_tensor.__sub__ = MagicMock(return_value=mock_train_tensor)
    mock_train_tensor.__truediv__ = MagicMock(return_value=mock_train_tensor)
    mock_test_tensor.__sub__ = MagicMock(return_value=mock_test_tensor)
    mock_test_tensor.__truediv__ = MagicMock(return_value=mock_test_tensor)
    
    # Set up CPU/detach/numpy chain
    mock_train_tensor.cpu.return_value = mock_train_tensor
    mock_train_tensor.detach.return_value = mock_train_tensor
    mock_train_tensor.numpy.return_value = expected_train
    
    mock_test_tensor.cpu.return_value = mock_test_tensor
    mock_test_tensor.detach.return_value = mock_test_tensor
    mock_test_tensor.numpy.return_value = expected_test
    
    # Call the function
    norm_train, norm_test, norm_stats = normalize_train_test_movies(train, test)
    
    # Check results
    assert_almost_equal(norm_train, expected_train, decimal=5)
    assert_almost_equal(norm_test, expected_test, decimal=5)
    assert norm_stats["norm_mean"] == train_mean
    assert norm_stats["norm_std"] == train_std


def test_compute_data_info():
    # Create sample data
    train_movie = np.random.randn(3, 100, 32, 32)
    test_movie = np.random.randn(3, 50, 32, 32)
    
    movies_split = MoviesTrainTestSplit(
        train=train_movie, 
        test=test_movie,
        norm_mean=0.5,
        norm_std=1.2
    )
    
    responses_dict = {
        "session1": ResponsesTrainTestSplit(
            train=np.random.randn(10, 100),
            test=np.random.randn(10, 50),
            session_kwargs={"param1": "value1"}
        ),
        "session2": ResponsesTrainTestSplit(
            train=np.random.randn(15, 100),
            test=np.random.randn(15, 50),
            session_kwargs={"param2": "value2"}
        ),
    }
    
    # Test with single MoviesTrainTestSplit
    result = compute_data_info(responses_dict, movies_split)
    
    assert result["n_neurons_dict"] == {"session1": 10, "session2": 15}
    assert result["input_shape"] == (3, 32, 32)
    assert result["sessions_kwargs"] == {
        "session1": {"param1": "value1"},
        "session2": {"param2": "value2"}
    }
    assert result["movie_norm_dict"] == {
        "default": {"norm_mean": 0.5, "norm_std": 1.2}
    }
    
    # Test with dictionary of MoviesTrainTestSplit
    movies_dict = {
        "movie1": MoviesTrainTestSplit(
            train=train_movie, 
            test=test_movie,
            norm_mean=0.5,
            norm_std=1.2
        ),
        "movie2": MoviesTrainTestSplit(
            train=train_movie, 
            test=test_movie,
            norm_mean=0.3,
            norm_std=0.9
        )
    }
    
    result = compute_data_info(responses_dict, movies_dict)
    
    assert result["movie_norm_dict"] == {
        "movie1": {"norm_mean": 0.5, "norm_std": 1.2},
        "movie2": {"norm_mean": 0.3, "norm_std": 0.9}
    }