"""Natural-movie filter initialization for the SC model on Höfling 2024 data."""

from pathlib import Path

import numpy as np
import torch

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.data_io.hoefling_2024.constants import CLIP_LENGTH, NUM_CLIPS

_MISLABELED_LEFT_SESSION = "session_2_ventral2_20200626"


def _roi_centers(roi_mask: np.ndarray, roi_ids: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Map recording ROI centers to the normalized readout grid used by Höfling loaders."""
    centers = []
    for roi_id in roi_ids:
        y, x = np.nonzero(roi_mask == -roi_id)
        if not len(y):
            raise ValueError(f"ROI {roi_id} is absent from its ROI mask")
        centers.append((((y.mean() / 25 + 2.75) / 8) * 2 - 1, ((x.mean() / 25 + 2.75) / 8) * 2 - 1))
    normalized = np.asarray(centers, dtype=np.float32)
    pixel_y = np.clip(np.rint((normalized[:, 0] + 1) * (height - 1) / 2), 0, height - 1).astype(int)
    pixel_x = np.clip(np.rint((normalized[:, 1] + 1) * (width - 1) / 2), 0, width - 1).astype(int)
    return pixel_y, pixel_x


def _gaussian_filters(
    center_y: np.ndarray,
    center_x: np.ndarray,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    grid_y, grid_x = np.mgrid[:height, :width]
    distance_sq = (grid_y - center_y[:, None, None]) ** 2 + (grid_x - center_x[:, None, None]) ** 2
    filters = np.exp(-distance_sq / (2 * sigma**2)).astype(np.float32)
    filters[distance_sq > (3 * sigma) ** 2] = 0
    filters /= filters.sum(axis=(1, 2), keepdims=True)
    return torch.from_numpy(filters)


def estimate_filter_bank(
    movies: MoviesTrainTestSplit,
    responses: dict[str, ResponsesTrainTestSplit],
    validation_clip_indices: list[int],
    output_path: str | Path,
    temporal_filter_length: int = 30,
    spatial_sigma: float = 1.5,
) -> Path:
    """Estimate ROI-centered spatial and reverse-correlation temporal filters.

    Temporal filters use only non-validation training clips. The temporal course
    is sampled at each ROI center, matching how the original SC model extracts
    the temporal component at the receptive-field center.
    """
    if movies.random_sequences is None:
        raise ValueError("Höfling random presentation sequences are required")
    channels, _, height, width = movies.train.shape
    movie_clips = movies.train.reshape(channels, NUM_CLIPS, CLIP_LENGTH, height, width)
    is_validation = np.isin(movies.random_sequences, validation_clip_indices)

    spatial_filters_dict: dict[str, torch.Tensor] = {}
    temporal_filters_dict: dict[str, torch.Tensor] = {}
    neuron_ids_dict: dict[str, list[int]] = {}

    for session, response in responses.items():
        kwargs = response.session_kwargs
        sequence_index = int(np.asarray(kwargs["scan_sequence_idx"]).item())
        presentation_order = movies.random_sequences[:, sequence_index].astype(int)
        keep = ~is_validation[:, sequence_index]
        stimulus = movie_clips[:, presentation_order[keep]].reshape(channels, -1, height, width)
        if kwargs["eye"] == "left" and session != _MISLABELED_LEFT_SESSION:
            stimulus = stimulus[..., ::-1].copy()  # type: ignore[assignment]

        center_y, center_x = _roi_centers(np.asarray(kwargs["roi_mask"]), np.asarray(kwargs["roi_ids"]), height, width)
        spatial_filters_dict[session] = _gaussian_filters(center_y, center_x, height, width, spatial_sigma)
        assert response.neuron_ids is not None
        neuron_ids_dict[session] = response.neuron_ids.tolist()

        local_stimulus = stimulus[:, :, center_y, center_x].transpose(2, 0, 1).astype(np.float32)
        local_stimulus -= local_stimulus.mean(axis=2, keepdims=True)
        response_train = response.train.reshape(response.n_neurons, NUM_CLIPS, CLIP_LENGTH)[:, keep].reshape(
            response.n_neurons, -1
        )
        response_train = response_train.astype(np.float32)
        response_train -= response_train.mean(axis=1, keepdims=True)

        stimulus_windows = torch.from_numpy(local_stimulus).unfold(2, temporal_filter_length, 1)
        aligned_response = torch.from_numpy(response_train[:, temporal_filter_length - 1 :])
        temporal_filters = torch.einsum("nctl,nt->ncl", stimulus_windows, aligned_response)
        temporal_filters /= aligned_response.shape[1]
        norms = temporal_filters.square().sum(dim=(1, 2), keepdim=True).sqrt()
        degenerate = norms.flatten() < 1e-8
        temporal_filters /= norms.clamp_min(1e-8)
        temporal_filters[degenerate, :, -1] = 1 / channels**0.5
        temporal_filters_dict[session] = temporal_filters

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "spatial_filters_dict": spatial_filters_dict,
            "temporal_filters_dict": temporal_filters_dict,
            "neuron_ids_dict": neuron_ids_dict,
            "metadata": {
                "method": "roi_center_reverse_correlation",
                "validation_clip_indices": validation_clip_indices,
                "temporal_filter_length": temporal_filter_length,
                "spatial_sigma": spatial_sigma,
            },
        },
        output_path,
    )
    return output_path
