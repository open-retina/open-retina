"""
STA (Spike-Triggered Average) processing utilities for the Spatial Contrast model.

This module provides functions for:
- Fitting 2D elliptical Gaussians to spatial patterns
- Extracting spatial and temporal filters from STAs
- Creating Gaussian contour masks
"""

import os
from typing import Tuple

import numpy as np
from jaxtyping import Float
from scipy.optimize import curve_fit


def gaussian_2d(
    coords: Tuple[np.ndarray, np.ndarray],
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    amplitude: float,
) -> np.ndarray:
    """
    2D elliptical Gaussian function for curve fitting.

    The Gaussian is parameterized by its center, principal axis standard deviations,
    rotation angle, and amplitude.

    Args:
        coords: Tuple of (x_grid, y_grid) meshgrid arrays
        center_x: X-coordinate of the Gaussian center
        center_y: Y-coordinate of the Gaussian center
        sigma_x: Standard deviation along the first principal axis
        sigma_y: Standard deviation along the second principal axis
        theta: Rotation angle in radians (counter-clockwise from x-axis)
        amplitude: Peak amplitude (can be negative for OFF cells)

    Returns:
        Flattened array of Gaussian values at each grid point
    """
    x, y = coords
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
    b = np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)

    x_diff = x - center_x
    y_diff = y - center_y

    gaussian = amplitude * np.exp(-(a * x_diff**2 + 2 * b * x_diff * y_diff + c * y_diff**2))
    return gaussian.ravel()


def fit_2d_gaussian(
    spatial_frame: Float[np.ndarray, "height width"],
    initial_center: Tuple[int, int] | None = None,
) -> dict:
    """
    Fit a 2D elliptical Gaussian to a spatial frame from STA.

    Args:
        spatial_frame: 2D array representing spatial pattern at peak temporal frame
        initial_center: Optional (y, x) initial guess for center. If None, uses
            location of maximum absolute value.

    Returns:
        Dictionary with keys:
            - 'center_x', 'center_y': Fitted center coordinates
            - 'sigma_x', 'sigma_y': Fitted standard deviations along principal axes
            - 'theta': Fitted rotation angle in radians
            - 'amplitude': Fitted peak amplitude
            - 'success': Boolean indicating if fitting succeeded
            - 'covariance': Covariance matrix of fitted parameters (None if failed)
    """
    height, width = spatial_frame.shape

    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Initial guesses
    if initial_center is None:
        # Use location of maximum absolute value
        max_idx = np.unravel_index(np.argmax(np.abs(spatial_frame)), spatial_frame.shape)
        init_center_y, init_center_x = int(max_idx[0]), int(max_idx[1])
    else:
        init_center_y, init_center_x = int(initial_center[0]), int(initial_center[1])

    init_amplitude = spatial_frame[int(init_center_y), int(init_center_x)]
    init_sigma = min(height, width) / 6.0  # Reasonable initial sigma

    # Initial parameter guess: (center_x, center_y, sigma_x, sigma_y, theta, amplitude)
    p0 = [init_center_x, init_center_y, init_sigma, init_sigma, 0.0, init_amplitude]

    # Parameter bounds
    bounds_lower = [0, 0, 0.5, 0.5, -np.pi, -np.inf]
    bounds_upper = [width - 1, height - 1, width, height, np.pi, np.inf]

    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (x_grid, y_grid),
            spatial_frame.ravel(),
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000,
        )
        return {
            "center_x": popt[0],
            "center_y": popt[1],
            "sigma_x": popt[2],
            "sigma_y": popt[3],
            "theta": popt[4],
            "amplitude": popt[5],
            "success": True,
            "covariance": pcov,
        }
    except (RuntimeError, ValueError) as e:
        # Fitting failed - return initial guess with success=False
        return {
            "center_x": float(init_center_x),
            "center_y": float(init_center_y),
            "sigma_x": init_sigma,
            "sigma_y": init_sigma,
            "theta": 0.0,
            "amplitude": init_amplitude,
            "success": False,
            "covariance": None,
            "error": str(e),
        }


def create_gaussian_mask(
    shape: Tuple[int, int],
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    n_sigma: float = 3.0,
) -> Float[np.ndarray, "height width"]:
    """
    Create a binary mask for pixels within n_sigma of the Gaussian center.

    The mask is 1 inside the n-sigma elliptical contour and 0 outside.

    Args:
        shape: (height, width) of the output mask
        center_x: X-coordinate of the Gaussian center
        center_y: Y-coordinate of the Gaussian center
        sigma_x: Standard deviation along the first principal axis
        sigma_y: Standard deviation along the second principal axis
        theta: Rotation angle in radians
        n_sigma: Number of standard deviations for the contour (default 3.0)

    Returns:
        Binary mask array of shape (height, width)
    """
    height, width = shape
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Rotate coordinates to align with principal axes
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_diff = x_grid - center_x
    y_diff = y_grid - center_y

    # Rotated coordinates
    x_rot = cos_theta * x_diff + sin_theta * y_diff
    y_rot = -sin_theta * x_diff + cos_theta * y_diff

    # Compute normalized distance from center (in units of sigma)
    normalized_distance = np.sqrt((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)

    # Create mask
    mask = (normalized_distance <= n_sigma).astype(np.float32)

    return mask


def extract_filters_from_sta(
    sta: Float[np.ndarray, "num_frames height width"],
    temporal_crop_frames: int | None = None,
    sigma_contour: float = 3.0,
) -> Tuple[Float[np.ndarray, "height width"], Float[np.ndarray, " temporal_frames"], dict]:
    """
    Extract spatial and temporal filters from STA.

    The spatial filter is extracted from the peak temporal frame (max variance),
    fitted with a 2D elliptical Gaussian, masked to sigma_contour standard deviations,
    and normalized to unit L2 norm. For OFF cells (negative amplitude), the polarity
    is flipped to ensure a positive-definite spatial filter.

    The temporal filter is extracted at the fitted RF center location, cropped to the
    last temporal_crop_frames if specified, reversed for correct convolution, and
    normalized to unit L2 norm.

    Args:
        sta: Spike-triggered average array (num_frames, height, width)
        temporal_crop_frames: Optional number of frames to keep in temporal filter.
            If None, uses all frames.
        sigma_contour: Number of sigmas for the spatial filter contour (default 3.0)

    Returns:
        Tuple of (spatial_filter, temporal_filter, gaussian_params)
        - spatial_filter: 2D array (height, width), positive definite, unit L2 norm
        - temporal_filter: 1D array (temporal_frames,), unit L2 norm
        - gaussian_params: Dictionary with fitted Gaussian parameters
    """
    num_frames, height, width = sta.shape

    # Find peak temporal frame (max variance across spatial pixels)
    temporal_variances = np.var(sta, axis=(1, 2))
    peak_temporal_idx = np.argmax(temporal_variances)

    # Extract spatial pattern at peak frame
    spatial_frame = sta[peak_temporal_idx].copy()

    # Fit 2D elliptical Gaussian
    gaussian_params = fit_2d_gaussian(spatial_frame)

    # Handle polarity (flip if amplitude is negative for OFF cells)
    polarity_flip = gaussian_params["amplitude"] < 0
    if polarity_flip:
        gaussian_params["amplitude"] = -gaussian_params["amplitude"]

    # Use the fitted 2D Gaussian as the spatial filter
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    gaussian_values = gaussian_2d(
        (x_grid, y_grid),
        center_x=gaussian_params["center_x"],
        center_y=gaussian_params["center_y"],
        sigma_x=gaussian_params["sigma_x"],
        sigma_y=gaussian_params["sigma_y"],
        theta=gaussian_params["theta"],
        amplitude=gaussian_params["amplitude"],
    ).reshape(height, width)

    # Create a mask of sigma_contour sigmas around the gaussian center
    mask = create_gaussian_mask(
        shape=(height, width),
        center_x=gaussian_params["center_x"],
        center_y=gaussian_params["center_y"],
        sigma_x=gaussian_params["sigma_x"],
        sigma_y=gaussian_params["sigma_y"],
        theta=gaussian_params["theta"],
        n_sigma=sigma_contour,
    )

    # Set spatial filter as the masked 2d-gaussian fit
    spatial_filter = gaussian_values * mask

    # Normalize spatial filter to unit L2 norm
    spatial_norm = np.linalg.norm(spatial_filter)
    if spatial_norm > 0:
        spatial_filter = spatial_filter / spatial_norm

    # Extract temporal filter at fitted Gaussian center
    center_y = int(np.clip(np.round(gaussian_params["center_y"]), 0, height - 1))
    center_x = int(np.clip(np.round(gaussian_params["center_x"]), 0, width - 1))
    temporal_filter = sta[:, center_y, center_x].copy()

    # Crop temporal filter to keep the most recent frames
    if temporal_crop_frames is not None and temporal_crop_frames < num_frames:
        temporal_filter = temporal_filter[-temporal_crop_frames:]

    # Normalize temporal filter to unit L2 norm
    temporal_norm = np.linalg.norm(temporal_filter)
    if temporal_norm > 0:
        temporal_filter = temporal_filter / temporal_norm

    return spatial_filter.astype(np.float32), temporal_filter.astype(np.float32), gaussian_params


def load_sta_and_extract_filters(
    sta_dir: str,
    file_name: str,
    flip_sta: bool = False,
    crop: int | Tuple[int, int, int, int] = 0,
    temporal_crop_frames: int | None = None,
    sigma_contour: float = 3.0,
) -> Tuple[Float[np.ndarray, "height width"], Float[np.ndarray, " temporal_frames"], dict]:
    """
    Load STA from file and extract filters.

    Combines loading logic with the filter extraction pipeline.

    Args:
        sta_dir: Directory containing STA files
        file_name: Name of the STA file (e.g., 'cell_data_01_WN_stas_cell_8.npy')
        flip_sta: Whether to flip the STA horizontally (for NM dataset compatibility)
        crop: Pixels to crop from (top, bottom, left, right). If int, crops same from all sides.
        temporal_crop_frames: Number of frames to keep in temporal filter
        sigma_contour: Number of sigmas for spatial filter contour (default 3.0)

    Returns:
        Tuple of (spatial_filter, temporal_filter, gaussian_params)
    """
    # Load STA
    sta_path = os.path.join(sta_dir, file_name)
    sta = np.load(sta_path)

    # Handle crop parameter
    if isinstance(crop, int):
        crop = (crop, crop, crop, crop)

    # Flip STA horizontally if needed (for NM dataset)
    if flip_sta:
        sta = np.flip(sta, axis=2).copy()  # Flip along width axis

    # Apply spatial cropping
    if sum(crop) > 0:
        num_frames, h, w = sta.shape
        sta = sta[:, crop[0] : h - crop[1], crop[2] : w - crop[3]]

    # Extract filters
    return extract_filters_from_sta(
        sta=sta,
        temporal_crop_frames=temporal_crop_frames,
        sigma_contour=sigma_contour,
    )
