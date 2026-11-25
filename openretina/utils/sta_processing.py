"""
STA (Spike-Triggered Average) processing utilities for the Spatial Contrast model.

Provides functions for fitting 2D elliptical Gaussians to spatial STA patterns,
extracting spatial and temporal filters, and creating Gaussian contour masks.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from scipy.optimize import curve_fit

LOGGER = logging.getLogger(__name__)


def gaussian_2d(
    coords: Any,
    /,
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
        the dictionary returned by fit_2d_gaussian, with "amplitude" made positive and two
        extra keys: "signed_amplitude" (the fitted amplitude before taking its absolute value)
        and "peak_frame" (the STA frame the Gaussian was fitted to).
    """
    num_frames, height, width = sta.shape

    # Guard against NaN/Inf in the STA data — return zero filters so the neuron
    # contributes a constant baseline output without corrupting other neurons.
    if not np.all(np.isfinite(sta)):
        LOGGER.warning("STA contains NaN or Inf values — returning zero filters for this neuron.")
        # Same length as the non-NaN path below, so filters of one session can still be stacked
        n_temporal = min(temporal_crop_frames, num_frames) if temporal_crop_frames is not None else num_frames
        spatial_filter = np.zeros((height, width), dtype=np.float32)
        temporal_filter = np.zeros(n_temporal, dtype=np.float32)
        gaussian_params = {
            "center_x": width / 2.0,
            "center_y": height / 2.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "theta": 0.0,
            "amplitude": 0.0,
            "signed_amplitude": 0.0,
            "peak_frame": 0,
            "success": False,
        }
        return spatial_filter, temporal_filter, gaussian_params

    # Find peak temporal frame (max variance across spatial pixels)
    temporal_variances = np.var(sta, axis=(1, 2))
    peak_temporal_idx = np.argmax(temporal_variances)

    spatial_frame = sta[peak_temporal_idx].copy()

    gaussian_params = fit_2d_gaussian(spatial_frame)

    # Flip polarity for OFF cells so spatial filter is always positive. The sign is kept separately so that callers
    # can check it against the temporal filter, which is what carries the polarity from here on.
    gaussian_params["signed_amplitude"] = gaussian_params["amplitude"]
    gaussian_params["peak_frame"] = int(peak_temporal_idx)
    gaussian_params["amplitude"] = np.abs(gaussian_params["amplitude"])

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
    sta_dir: Union[Path, str],
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


def export_lnp_filters_as_stas(
    model: Any,
    output_dir: Union[Path, str],
    file_pattern: str = "{session}_roi{roi_id}.npy",
    source: str | None = None,
) -> pd.DataFrame:
    """
    Save the linear filters of a trained greyscale LNP model as STA files for the Spatial Contrast model.

    Each neuron's filter is written as a float32 ``(time, height, width)`` array, the layout that
    `load_sta_and_extract_filters` reads. Frame order follows the SC readout's convention: index 0 is the
    furthest past, index -1 the current frame. A lag-0 LNP yields one frame per neuron.

    Files are named after ``data_info["sessions_kwargs"][session]["roi_ids"]`` rather than the neuron's position in
    the readout, because positions shift whenever the quality filtering changes while roi ids do not. The SC model
    then finds the files with ``sta_file_pattern="{retina_index}_roi{cell_index}.npy"`` and ``cell_id_key="roi_ids"``.

    Alongside the filters this writes ``manifest.csv`` (one row per neuron, including how the SC model's filter
    extraction sees each exported filter) and ``export_info.json`` (provenance, including the colour squashing
    weights, which the SC model checks against its own).

    Args:
        model: A trained core-readout model whose per-session readouts implement ``spatiotemporal_filters()``,
            i.e. an LNP built from ``LNPReadout`` or ``SeparableLNPReadout``. The readout must see a single
            channel, so a colour stimulus has to be squashed to greyscale inside the core.
        output_dir: Directory to write the filters, manifest and export info to. Created if missing.
        file_pattern: Filename pattern with ``{session}`` and ``{roi_id}`` placeholders.
        source: Optional free-text description of where the model came from (name or checkpoint path).

    Returns:
        The manifest as a DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_info = model.data_info
    sessions_kwargs = data_info.get("sessions_kwargs", {})
    performance = data_info.get("pretrained_performance", {})

    rows: list[dict[str, Any]] = []
    n_frames: int | None = None
    for session in model.readout.keys():
        with torch.no_grad():
            filters = model.readout[session].spatiotemporal_filters().detach().cpu().numpy()
        n_neurons, n_channels = filters.shape[:2]
        if n_channels != 1:
            raise ValueError(
                f"Session {session}: the readout sees {n_channels} channels, but the Spatial Contrast model needs "
                "greyscale filters. Squash the colour channels in the core (e.g. DummyCore color_squashing_weights)."
            )
        if n_frames is None:
            n_frames = filters.shape[2]

        roi_ids = sessions_kwargs.get(session, {}).get("roi_ids")
        if roi_ids is None:
            raise ValueError(f"Session {session}: data_info['sessions_kwargs'] has no 'roi_ids' to name files by.")
        if len(roi_ids) != n_neurons:
            raise ValueError(f"Session {session}: {len(roi_ids)} roi_ids for {n_neurons} neurons in the readout.")
        group_assignment = sessions_kwargs[session].get("group_assignment")
        session_performance = performance.get(session)

        for neuron_idx in range(n_neurons):
            roi_id = int(roi_ids[neuron_idx])
            sta = filters[neuron_idx, 0].astype(np.float32)
            file_name = file_pattern.format(session=session, roi_id=roi_id)
            np.save(output_dir / file_name, sta)

            # Run the SC model's own extraction, so problems show up here rather than mid-training
            _, temporal_filter, gaussian_params = extract_filters_from_sta(sta)
            polarity = float(np.sign(gaussian_params["signed_amplitude"]))
            temporal_sign_at_peak = float(np.sign(temporal_filter[gaussian_params["peak_frame"]]))
            rows.append(
                {
                    "session": session,
                    "neuron_idx": neuron_idx,
                    "roi_id": roi_id,
                    "file": file_name,
                    "group_assignment": (int(group_assignment[neuron_idx]) if group_assignment is not None else np.nan),
                    "lnp_test_corr": (
                        float(session_performance[neuron_idx]) if session_performance is not None else np.nan
                    ),
                    "n_frames": sta.shape[0],
                    "fit_success": bool(gaussian_params["success"]),
                    "center_x": float(gaussian_params["center_x"]),
                    "center_y": float(gaussian_params["center_y"]),
                    "sigma_x": float(gaussian_params["sigma_x"]),
                    "sigma_y": float(gaussian_params["sigma_y"]),
                    "polarity": polarity,
                    # The SC model takes the polarity from the RF centre pixel's time course, while the spatial
                    # Gaussian is made positive. If the two disagree, the cell would be modelled inverted.
                    "polarity_consistent": polarity != 0 and temporal_sign_at_peak == polarity,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    squashing_layer = getattr(model.core, "color_squashing_layer", None)
    export_info = {
        "source": source,
        "color_squashing_weights": (
            None if squashing_layer is None else squashing_layer.channel_weights.detach().cpu().tolist()
        ),
        "n_frames": n_frames,
        "model_cut_frames": data_info.get("model_cut_frames"),
        "file_pattern": file_pattern,
        "n_sessions": len(model.readout.keys()),
        "n_neurons": len(manifest),
    }
    with open(output_dir / "export_info.json", "w") as f:
        json.dump(export_info, f, indent=2)

    LOGGER.info(
        f"Exported {len(manifest)} filters of {n_frames} frame(s) to {output_dir}: "
        f"{int(manifest['fit_success'].sum())} Gaussian fits succeeded, "
        f"{int((~manifest['polarity_consistent']).sum())} with inconsistent polarity."
    )
    return manifest
