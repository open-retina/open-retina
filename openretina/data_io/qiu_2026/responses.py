"""qiu_2026 response loader.

Loads raw calcium fluorescence, applies the per-session ``neurons_fluor_good`` index mask, trims NaN
padding, and assembles a continuous train trace plus per-``condition_hash`` test dictionaries (both the
trial-averaged ``test_dict`` and the per-repeat ``test_by_trial_dict``). Masked cortex coordinates are
attached via ``session_kwargs`` so they can reach the readout's ``grid_mean_predictor``.

``spike_inference`` controls the response transform:
- ``"raw"``: fluorescence untouched (default; used by fast smoke tests).
- ``"subtract_min"``: subtract each neuron's train-derived minimum → non-negative, trainable interim
  target for the Poisson loss until CASCADE is wired in.
- ``"cascade"``: CASCADE inferred spike rates (Rupprecht 2021) — not yet implemented (Open Question 1).
"""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.qiu_2026.trials import (
    STREAM_TIME_AXIS,
    discover_quality_masks,
    discover_sessions,
    load_trial_array,
    match_quality_mask,
    read_condition_hash,
    read_tiers,
    resolve_session_path,
    session_key,
    test_conditions,
    train_val_indices,
    trim_time,
    valid_length,
    validation_clip_indices,
)
from openretina.utils.file_utils import get_local_file_path

SpikeInference = Literal["raw", "subtract_min", "cascade"]


def _spike_offset(train: np.ndarray, spike_inference: SpikeInference) -> np.ndarray:
    """Per-neuron offset (shape ``(N, 1)``) subtracted from every split so train/test stay consistent."""
    if spike_inference == "raw":
        return np.zeros((train.shape[0], 1), dtype=np.float32)
    if spike_inference == "subtract_min":
        return train.min(axis=1, keepdims=True).astype(np.float32)
    if spike_inference == "cascade":
        raise NotImplementedError(
            "spike_inference='cascade' is not yet implemented (see Open Question 1 in the integration "
            "plan). Use 'raw' or 'subtract_min' for now."
        )
    raise ValueError(f"Unknown spike_inference={spike_inference!r}; expected 'raw', 'subtract_min' or 'cascade'.")


def load_responses_for_session(
    session_path: str | os.PathLike,
    good_idx: np.ndarray | None = None,
    *,
    spike_inference: SpikeInference = "raw",
    coordinate_dims: int = 2,
) -> ResponsesTrainTestSplit:
    """Build a :class:`ResponsesTrainTestSplit` for one qiu_2026 session.

    ``good_idx`` is the ``neurons_fluor_good`` index array (applied to both responses and cortex
    coordinates); ``None`` keeps all neurons. ``coordinate_dims`` selects the leading cortex-coordinate
    columns stored as the readout ``source_grid`` (2 → X,Y).
    """
    session_path = resolve_session_path(session_path)
    tiers = read_tiers(session_path)
    condition_hash = read_condition_hash(session_path)
    time_axis = STREAM_TIME_AXIS["responses"]

    def load_trimmed(i: int) -> np.ndarray:
        responses = load_trial_array(session_path, "responses", i)  # (N, T)
        if good_idx is not None:
            responses = responses[good_idx]
        n_valid = valid_length(responses, time_axis)
        return trim_time(responses, time_axis, n_valid).astype(np.float32)

    train = np.concatenate([load_trimmed(i) for i in train_val_indices(tiers)], axis=1)  # (N, T_total)
    offset = _spike_offset(train, spike_inference)
    train = (train - offset).astype(np.float32)

    test_dict: dict[str, np.ndarray] = {}
    test_by_trial_dict: dict[str, np.ndarray] = {}
    for condition, indices in test_conditions(tiers, condition_hash).items():
        by_trial = np.stack([load_trimmed(i) for i in indices], axis=0)  # (repeats, N, T_cond)
        by_trial = (by_trial - offset[None]).astype(np.float32)
        test_by_trial_dict[condition] = by_trial
        test_dict[condition] = by_trial.mean(axis=0).astype(np.float32)  # (N, T_cond)

    cell_motor_coordinates = np.load(session_path / "meta" / "neurons" / "cell_motor_coordinates.npy").astype(
        np.float32
    )
    if good_idx is not None:
        cell_motor_coordinates = cell_motor_coordinates[good_idx]
    source_grid = cell_motor_coordinates[:, :coordinate_dims]  # (N, coordinate_dims)

    session_kwargs = {
        "cell_motor_coordinates": source_grid,
        "validation_clip_indices": validation_clip_indices(tiers),
    }
    return ResponsesTrainTestSplit(
        train=train,
        test_dict=test_dict,
        test_by_trial_dict=test_by_trial_dict,
        stim_id="qiu_2026",
        session_kwargs=session_kwargs,
    )


def load_all_responses(
    base_data_path: str | os.PathLike,
    *,
    apply_quality_mask: bool = True,
    spike_inference: SpikeInference = "raw",
    coordinate_dims: int = 2,
    sessions: list[str] | None = None,
) -> dict[str, ResponsesTrainTestSplit]:
    """Load every discovered qiu_2026 session into a ``{session_key: ResponsesTrainTestSplit}`` dict."""
    base_data_path = Path(get_local_file_path(str(base_data_path)))
    session_paths = discover_sessions(base_data_path, sessions)
    masks = discover_quality_masks(base_data_path) if apply_quality_mask else {}

    responses_all_sessions: dict[str, ResponsesTrainTestSplit] = {}
    for path in tqdm(session_paths, desc="qiu_2026 responses"):
        key = session_key(path)
        good_idx = match_quality_mask(key, masks) if apply_quality_mask else None
        responses_all_sessions[key] = load_responses_for_session(
            path, good_idx=good_idx, spike_inference=spike_inference, coordinate_dims=coordinate_dims
        )
    return responses_all_sessions
