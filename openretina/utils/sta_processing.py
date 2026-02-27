"""
STA (Spike-Triggered Average) processing utilities for the Spatial Contrast model.

Provides functions for fitting 2D elliptical Gaussians to spatial STA patterns,
extracting spatial and temporal filters, and creating Gaussian contour masks.
"""

import logging
import os

import numpy as np
from jaxtyping import Float
from scipy.optimize import curve_fit

LOGGER = logging.getLogger(__name__)


def gaussian_2d(
    coords: tuple[np.ndarray, np.ndarray],
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    amplitude: float,
) -> np.ndarray:
    """
    2D elliptical Gaussian function for curve fitting.

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
    initial_center: tuple[int, int] | None = None,
) -> dict:
    """
    Fit a 2D elliptical Gaussian to a spatial frame from an STA.

    Args:
        spatial_frame: 2D array representing spatial pattern at peak temporal frame
        initial_center: Optional (y, x) initial guess for center. If None, uses
            location of maximum absolute value.

    Returns:
        Dictionary with fitted parameters: center_x, center_y, sigma_x, sigma_y,
        theta, amplitude, success, covariance (None if failed).
    """
    height, width = spatial_frame.shape

    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    if initial_center is None:
        max_idx = np.unravel_index(np.argmax(np.abs(spatial_frame)), spatial_frame.shape)
        init_center_y, init_center_x = int(max_idx[0]), int(max_idx[1])
    else:
        init_center_y, init_center_x = int(initial_center[0]), int(initial_center[1])

    init_amplitude = spatial_frame[int(init_center_y), int(init_center_x)]
    init_sigma = min(height, width) / 6.0

    p0 = [init_center_x, init_center_y, init_sigma, init_sigma, 0.0, init_amplitude]

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
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    n_sigma: float = 3.0,
) -> Float[np.ndarray, "height width"]:
    """
    Create a binary elliptical mask at n_sigma standard deviations from the center.

    Args:
        shape: (height, width) of the output mask
        center_x, center_y: Gaussian center coordinates
        sigma_x, sigma_y: Standard deviations along principal axes
        theta: Rotation angle in radians
        n_sigma: Number of standard deviations for the contour (default 3.0)

    Returns:
        Binary mask array of shape (height, width), 1 inside the contour, 0 outside
    """
    height, width = shape
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_diff = x_grid - center_x
    y_diff = y_grid - center_y

    x_rot = cos_theta * x_diff + sin_theta * y_diff
    y_rot = -sin_theta * x_diff + cos_theta * y_diff

    normalized_distance = np.sqrt((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)

    mask = (normalized_distance <= n_sigma).astype(np.float32)

    return mask


def extract_filters_from_sta(
    sta: Float[np.ndarray, "num_frames height width"],
    temporal_crop_frames: int | None = None,
    sigma_contour: float = 3.0,
) -> tuple[Float[np.ndarray, "height width"], Float[np.ndarray, " temporal_frames"], dict]:
    """
    Extract spatial and temporal filters from an STA.

    The spatial filter is a 2D Gaussian fit to the peak temporal frame of the STA,
    masked to sigma_contour standard deviations and normalized to unit L2 norm.
    For OFF cells (negative amplitude), the polarity is flipped so the spatial
    filter is always positive.

    The temporal filter is the STA time course at the RF center, optionally cropped
    to the last temporal_crop_frames, and normalized to unit L2 norm.

    Args:
        sta: Spike-triggered average array (num_frames, height, width)
        temporal_crop_frames: Number of most-recent frames to keep. None = all frames.
        sigma_contour: Number of sigmas for the spatial filter contour (default 3.0)

    Returns:
        (spatial_filter, temporal_filter, gaussian_params) where gaussian_params is
        the dictionary returned by fit_2d_gaussian.
    """
    num_frames, height, width = sta.shape

    # Find peak temporal frame (max variance across spatial pixels)
    temporal_variances = np.var(sta, axis=(1, 2))
    peak_temporal_idx = np.argmax(temporal_variances)

    spatial_frame = sta[peak_temporal_idx].copy()

    gaussian_params = fit_2d_gaussian(spatial_frame)

    # Flip polarity for OFF cells so spatial filter is always positive
    if gaussian_params["amplitude"] < 0:
        gaussian_params["amplitude"] = -gaussian_params["amplitude"]

    # Build the spatial filter as a masked 2D Gaussian
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

    mask = create_gaussian_mask(
        shape=(height, width),
        center_x=gaussian_params["center_x"],
        center_y=gaussian_params["center_y"],
        sigma_x=gaussian_params["sigma_x"],
        sigma_y=gaussian_params["sigma_y"],
        theta=gaussian_params["theta"],
        n_sigma=sigma_contour,
    )

    spatial_filter = gaussian_values * mask

    spatial_norm = np.linalg.norm(spatial_filter)
    if spatial_norm > 0:
        spatial_filter = spatial_filter / spatial_norm

    # Extract temporal filter at the RF center
    center_y = int(np.clip(np.round(gaussian_params["center_y"]), 0, height - 1))
    center_x = int(np.clip(np.round(gaussian_params["center_x"]), 0, width - 1))
    temporal_filter = sta[:, center_y, center_x].copy()

    if temporal_crop_frames is not None and temporal_crop_frames < num_frames:
        temporal_filter = temporal_filter[-temporal_crop_frames:]

    temporal_norm = np.linalg.norm(temporal_filter)
    if temporal_norm > 0:
        temporal_filter = temporal_filter / temporal_norm

    return spatial_filter.astype(np.float32), temporal_filter.astype(np.float32), gaussian_params


def load_sta_and_extract_filters(
    sta_dir: str,
    file_name: str,
    flip_sta: bool = False,
    target_spatial_shape: tuple[int, int] | None = None,
    temporal_crop_frames: int | None = None,
    sigma_contour: float = 3.0,
) -> tuple[Float[np.ndarray, "height width"], Float[np.ndarray, " temporal_frames"], dict]:
    """
    Load an STA from file and extract spatial/temporal filters.

    If target_spatial_shape is provided, the STA is symmetrically cropped to match
    the stimulus dimensions used by the dataloader. This ensures that the STA and
    stimulus share the same coordinate system.

    Args:
        sta_dir: Directory containing STA .npy files
        file_name: STA filename (e.g., 'cell_data_01_WN_stas_cell_8.npy')
        flip_sta: Whether to flip the STA horizontally (needed when the stimulus
            and STA coordinate systems are mirrored, as in the NM dataset)
        target_spatial_shape: (height, width) that the stimulus is cropped to.
            The STA will be symmetrically cropped to match. None = no cropping.
        temporal_crop_frames: Number of most-recent frames to keep in temporal filter
        sigma_contour: Number of sigmas for spatial filter contour (default 3.0)

    Returns:
        (spatial_filter, temporal_filter, gaussian_params)
    """
    sta_path = os.path.join(sta_dir, file_name)
    sta = np.load(sta_path)

    if flip_sta:
        sta = np.flip(sta, axis=2).copy()  # Flip along width axis

    # Crop STA to match the stimulus spatial dimensions
    if target_spatial_shape is not None:
        _, sta_h, sta_w = sta.shape
        target_h, target_w = target_spatial_shape

        if sta_h != target_h or sta_w != target_w:
            crop_top = (sta_h - target_h) // 2
            crop_left = (sta_w - target_w) // 2

            if crop_top < 0 or crop_left < 0:
                raise ValueError(
                    f"STA spatial dimensions ({sta_h}, {sta_w}) are smaller than "
                    f"target shape ({target_h}, {target_w}). Cannot crop."
                )

            LOGGER.info(
                f"Cropping STA from ({sta_h}, {sta_w}) to ({target_h}, {target_w}): "
                f"crop_top={crop_top}, crop_left={crop_left}"
            )
            sta = sta[:, crop_top : crop_top + target_h, crop_left : crop_left + target_w]

    return extract_filters_from_sta(
        sta=sta,
        temporal_crop_frames=temporal_crop_frames,
        sigma_contour=sigma_contour,
    )
