"""Shared trial-splitting, NaN-trimming and session-discovery helpers for the qiu_2026 loaders.

Every per-trial array lives at ``data/<stream>/{i}.npy`` for ``i`` in ``0..n_trials-1``, aligned
positionally to row ``i`` of every ``meta/trials/*.npy`` array (NOT to the value stored in
``trial_idx.npy``, which carries the original DataJoint ids with gaps). NaN padding is strictly
*trailing* (whole-frame NaN) and identical across the four streams for a given trial, so the same
trim/split logic must be applied to all of them — hence these shared helpers, imported by
``stimuli.py``, ``responses.py`` and ``pupil.py`` so their outputs stay frame-aligned.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from openretina.data_io.qiu_2026 import constants as C
from openretina.utils.file_utils import unzip_and_cleanup

# Time axis of each per-trial stream array.
STREAM_TIME_AXIS = {"videos": 2, "responses": 1, "behavior": 1, "pupil_center": 1}

# Substring identifying a session archive/dir (all sessions carry the fluorescence hash suffix).
_SESSION_MARKER = "-Fluorescence-"


def resolve_session_path(session_path: str | os.PathLike) -> Path:
    """Return the on-disk session directory, unzipping (and deleting) a ``.zip`` if given one."""
    path = Path(session_path)
    if str(path).endswith(".zip"):
        path = Path(unzip_and_cleanup(path))
    return path


def _trials_dir(session_path: str | os.PathLike) -> Path:
    return Path(session_path) / "meta" / "trials"


def read_tiers(session_path: str | os.PathLike) -> np.ndarray:
    """Per-trial split labels ("train"/"validation"/"test"), one entry per trial file index."""
    return np.load(_trials_dir(session_path) / "tiers.npy")


def read_condition_hash(session_path: str | os.PathLike) -> np.ndarray:
    """Per-trial stimulus-condition hash (string); test repeats share a hash."""
    return np.load(_trials_dir(session_path) / "condition_hash.npy")


def load_trial_array(session_path: str | os.PathLike, stream: str, index: int) -> np.ndarray:
    """Load ``data/<stream>/{index}.npy`` (NaN-padded to the full clip length)."""
    return np.load(Path(session_path) / "data" / stream / f"{index}.npy")


def valid_length(arr: np.ndarray, time_axis: int) -> int:
    """Number of leading valid frames along ``time_axis`` (padding is trailing whole-frame NaN)."""
    other_axes = tuple(d for d in range(arr.ndim) if d != time_axis)
    fully_nan = np.isnan(arr).all(axis=other_axes)
    return int((~fully_nan).sum())


def trim_time(arr: np.ndarray, time_axis: int, n_frames: int) -> np.ndarray:
    """Slice ``arr`` to its first ``n_frames`` along ``time_axis``."""
    index: list[slice] = [slice(None)] * arr.ndim
    index[time_axis] = slice(0, n_frames)
    return arr[tuple(index)]


def train_val_indices(tiers: np.ndarray) -> list[int]:
    """File indices of train+validation trials, in file-index order (== concatenated clip order)."""
    return [i for i in range(len(tiers)) if str(tiers[i]) in ("train", "validation")]


def validation_clip_indices(tiers: np.ndarray) -> list[int]:
    """Positions of validation-tier trials within the train+val concatenation.

    With one trial per fixed-length clip, clip index == rank in the train+val order, so these can be
    passed as ``val_clip_indices`` to the dataloader to honour the dataset's curated validation split.
    """
    train_val = train_val_indices(tiers)
    return [rank for rank, i in enumerate(train_val) if str(tiers[i]) == "validation"]


def test_conditions(tiers: np.ndarray, condition_hash: np.ndarray) -> dict[str, list[int]]:
    """Map each test ``condition_hash`` to its ordered list of trial file indices (first-seen order)."""
    conditions: dict[str, list[int]] = {}
    for i in range(len(tiers)):
        if str(tiers[i]) == "test":
            conditions.setdefault(str(condition_hash[i]), []).append(i)
    return conditions


def session_key(session_path: str | os.PathLike) -> str:
    """Stable session id: the directory/archive name with a trailing ``.zip`` removed."""
    name = os.path.basename(os.path.normpath(str(session_path)))
    return name[: -len(".zip")] if name.endswith(".zip") else name


def discover_sessions(base_path: str | os.PathLike, sessions: list[str] | None = None) -> list[Path]:
    """List session archives/directories under ``base_path`` (skipping the ``data-quality`` folder)."""
    base_path = Path(base_path)
    found: list[Path] = []
    for name in sorted(os.listdir(base_path)):
        if name == C.DATA_QUALITY_DIRNAME:
            continue
        full = base_path / name
        is_session = name.endswith(C.SESSION_ZIP_SUFFIX) or (full.is_dir() and _SESSION_MARKER in name)
        if is_session and (sessions is None or session_key(full) in sessions):
            found.append(full)
    if not found:
        raise FileNotFoundError(
            f"No qiu_2026 session archives/directories found under {base_path}. "
            "Point paths.data_dir at the folder containing the '-Fluorescence-...' session zips."
        )
    return found


def discover_quality_masks(base_path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load ``data-quality/<prefix>_neurons_fluor_good.npy`` index arrays, keyed by ``<prefix>``.

    The ``<prefix>`` truncates the session's hash (e.g. ``...-Fluorescence-7b7``), so masks are matched
    to sessions by ``session_key.startswith(prefix)`` in :func:`match_quality_mask`.
    """
    quality_dir = Path(base_path) / C.DATA_QUALITY_DIRNAME
    masks: dict[str, np.ndarray] = {}
    if not quality_dir.is_dir():
        return masks
    for name in sorted(os.listdir(quality_dir)):
        if name.endswith(C.QUALITY_MASK_SUFFIX):
            prefix = name[: -len(C.QUALITY_MASK_SUFFIX)]
            masks[prefix] = np.load(quality_dir / name)
    return masks


def match_quality_mask(key: str, masks: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return the ``neurons_fluor_good`` index array whose (truncated) prefix matches ``key``, or None."""
    for prefix, idx in masks.items():
        if key.startswith(prefix):
            return idx
    warnings.warn(
        f"No neurons_fluor_good quality mask found for session '{key}'; proceeding without masking.",
        stacklevel=2,
    )
    return None
