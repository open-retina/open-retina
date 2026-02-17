"""
Unit tests for the Spatial Contrast model and STA processing utilities.
"""

import numpy as np
import pytest
import torch

from openretina.models.spatial_contrast import SingleCellSpatialContrast, SpatialContrastNonlinearity
from openretina.utils.sta_processing import (
    create_gaussian_mask,
    extract_filters_from_sta,
    fit_2d_gaussian,
    gaussian_2d,
)


class TestGaussian2d:
    """Tests for the 2D Gaussian function."""

    def test_gaussian_at_center(self):
        """Test that Gaussian is maximum at center."""
        x = np.arange(20)
        y = np.arange(20)
        x_grid, y_grid = np.meshgrid(x, y)

        center_x, center_y = 10.0, 10.0
        sigma_x, sigma_y = 2.0, 2.0
        theta = 0.0
        amplitude = 1.0

        result = gaussian_2d((x_grid, y_grid), center_x, center_y, sigma_x, sigma_y, theta, amplitude)
        result_2d = result.reshape(20, 20)

        # Maximum should be at center
        max_idx = np.unravel_index(np.argmax(result_2d), result_2d.shape)
        assert max_idx == (10, 10)

    def test_gaussian_symmetry(self):
        """Test that isotropic Gaussian is symmetric."""
        x = np.arange(21)
        y = np.arange(21)
        x_grid, y_grid = np.meshgrid(x, y)

        center_x, center_y = 10.0, 10.0
        sigma = 3.0

        result = gaussian_2d((x_grid, y_grid), center_x, center_y, sigma, sigma, 0.0, 1.0)
        result_2d = result.reshape(21, 21)

        # Should be symmetric
        np.testing.assert_array_almost_equal(result_2d, result_2d.T)

    def test_gaussian_negative_amplitude(self):
        """Test Gaussian with negative amplitude."""
        x = np.arange(10)
        y = np.arange(10)
        x_grid, y_grid = np.meshgrid(x, y)

        result = gaussian_2d((x_grid, y_grid), 5.0, 5.0, 2.0, 2.0, 0.0, -1.0)
        result_2d = result.reshape(10, 10)

        # Should be negative at center
        assert result_2d[5, 5] < 0


class TestFit2dGaussian:
    """Tests for 2D Gaussian fitting."""

    def test_fit_known_gaussian(self):
        """Test fitting recovers known Gaussian parameters."""
        # Create a known Gaussian
        height, width = 30, 30
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)

        true_params = {
            "center_x": 15.0,
            "center_y": 15.0,
            "sigma_x": 4.0,
            "sigma_y": 4.0,
            "theta": 0.0,
            "amplitude": 2.0,
        }

        data = gaussian_2d(
            (x_grid, y_grid),
            true_params["center_x"],
            true_params["center_y"],
            true_params["sigma_x"],
            true_params["sigma_y"],
            true_params["theta"],
            true_params["amplitude"],
        ).reshape(height, width)

        # Add small noise
        np.random.seed(42)
        data += np.random.normal(0, 0.01, data.shape)

        # Fit
        result = fit_2d_gaussian(data)

        assert result["success"]
        assert abs(result["center_x"] - true_params["center_x"]) < 0.5
        assert abs(result["center_y"] - true_params["center_y"]) < 0.5
        assert abs(result["sigma_x"] - true_params["sigma_x"]) < 0.5
        assert abs(result["sigma_y"] - true_params["sigma_y"]) < 0.5

    def test_fit_handles_noise(self):
        """Test that fitting handles noisy data gracefully."""
        np.random.seed(42)
        noisy_data = np.random.randn(20, 20)

        # Should not crash
        result = fit_2d_gaussian(noisy_data)

        assert "success" in result
        assert "center_x" in result
        assert "center_y" in result


class TestCreateGaussianMask:
    """Tests for Gaussian mask creation."""

    def test_mask_shape(self):
        """Test mask has correct shape."""
        shape = (40, 50)
        mask = create_gaussian_mask(shape, 25.0, 20.0, 5.0, 5.0, 0.0, 3.0)

        assert mask.shape == shape

    def test_mask_values(self):
        """Test mask has binary values."""
        mask = create_gaussian_mask((30, 30), 15.0, 15.0, 3.0, 3.0, 0.0, 3.0)

        # Should be either 0 or 1
        unique_values = np.unique(mask)
        assert len(unique_values) <= 2
        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_mask_center_included(self):
        """Test that center is inside the mask."""
        mask = create_gaussian_mask((30, 30), 15.0, 15.0, 3.0, 3.0, 0.0, 3.0)

        assert mask[15, 15] == 1.0


class TestExtractFiltersFromSta:
    """Tests for STA filter extraction."""

    @pytest.fixture
    def synthetic_sta(self):
        """Create a synthetic STA with known properties."""
        np.random.seed(42)
        num_frames, height, width = 30, 40, 50

        # Create a spatial pattern with clear peak
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)
        spatial = gaussian_2d((x_grid, y_grid), 25.0, 20.0, 3.0, 3.0, 0.0, 2.0).reshape(height, width)

        # Create a temporal pattern with clear peak
        t = np.arange(num_frames)
        temporal = np.exp(-((t - 15) ** 2) / 20)

        # Combine into STA
        sta = np.outer(temporal, spatial.ravel()).reshape(num_frames, height, width)

        # Add some noise
        sta += np.random.normal(0, 0.05, sta.shape)

        return sta.astype(np.float32)

    def test_filter_shapes(self, synthetic_sta):
        """Test that extracted filters have correct shapes."""
        spatial_filter, temporal_filter, params = extract_filters_from_sta(synthetic_sta)

        assert spatial_filter.shape == (40, 50)
        assert len(temporal_filter.shape) == 1

    def test_spatial_filter_normalized(self, synthetic_sta):
        """Test spatial filter has unit L2 norm."""
        spatial_filter, _, _ = extract_filters_from_sta(synthetic_sta)

        norm = np.linalg.norm(spatial_filter)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_temporal_filter_normalized(self, synthetic_sta):
        """Test temporal filter has unit L2 norm."""
        _, temporal_filter, _ = extract_filters_from_sta(synthetic_sta)

        norm = np.linalg.norm(temporal_filter)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_spatial_filter_positive(self, synthetic_sta):
        """Test spatial filter is non-negative (positive definite)."""
        spatial_filter, _, _ = extract_filters_from_sta(synthetic_sta)

        assert np.all(spatial_filter >= 0)

    def test_temporal_crop(self, synthetic_sta):
        """Test temporal filter cropping."""
        _, temporal_filter, _ = extract_filters_from_sta(synthetic_sta, temporal_crop_frames=10)

        assert temporal_filter.shape[0] == 10


class TestSpatialContrastNonlinearity:
    """Tests for the parametrized softplus nonlinearity."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        nl = SpatialContrastNonlinearity()
        x = torch.randn(4, 10)
        y = nl(x)

        assert y.shape == x.shape

    def test_positive_output(self):
        """Test that softplus produces positive outputs."""
        nl = SpatialContrastNonlinearity(a=1.0, b=1.0, c=0.0)
        x = torch.randn(100)
        y = nl(x)

        assert (y >= 0).all()

    def test_parameters_learnable(self):
        """Test all parameters have gradients."""
        nl = SpatialContrastNonlinearity()
        x = torch.randn(10, requires_grad=True)
        y = nl(x).sum()
        y.backward()

        assert nl.a.grad is not None
        assert nl.b.grad is not None
        assert nl.c.grad is not None


class TestSingleCellSpatialContrast:
    """Tests for the SingleCellSpatialContrast model."""

    @pytest.fixture
    def mock_sta_dir(self, tmp_path):
        """Create mock STA files for testing."""
        np.random.seed(42)

        # Create a synthetic STA
        num_frames, height, width = 30, 40, 50
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)
        spatial = gaussian_2d((x_grid, y_grid), 25.0, 20.0, 3.0, 3.0, 0.0, 2.0).reshape(height, width)
        t = np.arange(num_frames)
        temporal = np.exp(-((t - 15) ** 2) / 20)
        sta = np.outer(temporal, spatial.ravel()).reshape(num_frames, height, width)
        sta += np.random.normal(0, 0.05, sta.shape)

        # Save to file
        sta_path = tmp_path / "stas"
        sta_path.mkdir()
        np.save(sta_path / "cell_data_01_WN_stas_cell_0.npy", sta.astype(np.float32))

        return str(sta_path)

    def test_model_instantiation(self, mock_sta_dir):
        """Test model can be instantiated."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )

        assert model is not None

    def test_forward_pass_shape(self, mock_sta_dir):
        """Test forward pass produces correct output shape."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )
        x = torch.randn(2, 1, 30, 40, 50)
        y = model(x)

        # Output time = input time - temporal_filter_length + 1
        expected_time = 30 - model.temporal_filter.shape[0] + 1
        assert y.shape == (2, expected_time, 1)

    def test_only_four_learnable_params(self, mock_sta_dir):
        """Test model has exactly 4 learnable parameters."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params == 4  # a, b, c, w

    def test_filters_are_frozen(self, mock_sta_dir):
        """Test spatial and temporal filters are not learnable."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )

        assert not model.spatial_filter.requires_grad
        assert not model.temporal_filter.requires_grad

    def test_gradient_flow(self, mock_sta_dir):
        """Test gradients flow through all learnable parameters."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )
        x = torch.randn(2, 1, 30, 40, 50)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients for learnable parameters
        assert model.w.grad is not None
        assert model.nonlinearity.a.grad is not None
        assert model.nonlinearity.b.grad is not None
        assert model.nonlinearity.c.grad is not None

    def test_output_positive(self, mock_sta_dir):
        """Test model output is non-negative (due to softplus)."""
        model = SingleCellSpatialContrast(
            in_shape=(1, 30, 40, 50),
            sta_dir=mock_sta_dir,
            sta_file_name="cell_data_01_WN_stas_cell_0.npy",
        )
        x = torch.randn(2, 1, 30, 40, 50)
        y = model(x)

        assert (y >= 0).all()
