"""qiu_2026 pupil-center loader.

Loads the ``(2, T)`` eye-position trace per trial, z-scores it with the shipped pupil statistics, trims
NaN padding, and splits it with the *same* train/test logic used by the stimulus and response loaders so
the trace stays frame-aligned. There is no container for pupil data (it is not a model target); the
per-session dict returned here is consumed by the qiu dataloader, which slices it into the shifter input.

Layout mirrors the response split: ``train`` is the continuous train+validation trace; ``test_dict`` holds
one trace per ``condition_hash``, averaged over the condition's repeats to pair with the trial-averaged
test responses.
"""

import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from openretina.data_io.qiu_2026.trials import (
    STREAM_TIME_AXIS,
    discover_sessions,
    load_trial_array,
    read_condition_hash,
    read_tiers,
    resolve_session_path,
    session_key,
    test_conditions,
    train_val_indices,
    trim_time,
    valid_length,
)
from openretina.utils.file_utils import get_local_file_path


def load_pupil_for_session(session_path: str | os.PathLike) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Return ``{"train": (2, T_total), "test_dict": {condition: (2, T_cond)}}`` for one session."""
    session_path = resolve_session_path(session_path)
    tiers = read_tiers(session_path)
    condition_hash = read_condition_hash(session_path)
    time_axis = STREAM_TIME_AXIS["pupil_center"]

    stat = session_path / "meta" / "statistics" / "pupil_center" / "all"
    pupil_mean = np.asarray(np.load(stat / "mean.npy")[..., 0], dtype=np.float32)[:, None]  # (2, 1)
    pupil_std = np.asarray(np.load(stat / "std.npy")[..., 0], dtype=np.float32)[:, None]

    def load_trimmed(i: int) -> np.ndarray:
        pupil = load_trial_array(session_path, "pupil_center", i).astype(np.float32)  # (2, T)
        n_valid = valid_length(pupil, time_axis)
        pupil = trim_time(pupil, time_axis, n_valid)
        return (pupil - pupil_mean) / pupil_std

    train = np.concatenate([load_trimmed(i) for i in train_val_indices(tiers)], axis=1)  # (2, T_total)

    test_dict: dict[str, np.ndarray] = {}
    for condition, indices in test_conditions(tiers, condition_hash).items():
        repeats = np.stack([load_trimmed(i) for i in indices], axis=0)  # (repeats, 2, T_cond)
        # TODO(qiu_2026): pupil genuinely differs across repeats; averaging pairs it with the
        # trial-averaged responses. Revisit whether per-repeat pupil is more appropriate.
        # Tracked in qiu_2026_integration_plan.md (Open Questions).
        test_dict[condition] = repeats.mean(axis=0).astype(np.float32)

    return {"train": train, "test_dict": test_dict}


def load_all_pupil(
    base_data_path: str | os.PathLike,
    *,
    sessions: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
    """Load every discovered qiu_2026 session's pupil trace into a ``{session_key: {...}}`` dict."""
    base_data_path = Path(get_local_file_path(str(base_data_path)))
    pupil_all_sessions: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = {}
    for path in tqdm(discover_sessions(base_data_path, sessions), desc="qiu_2026 pupil"):
        pupil_all_sessions[session_key(path)] = load_pupil_for_session(path)
    return pupil_all_sessions
