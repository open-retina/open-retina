"""qiu_2026 stimulus loader.

Builds a ``(C, T, H, W)`` input where ``C = 1 video + len(behavior_channels) behavior`` channels: the
grayscale movie plus the selected behavior traces (pupil size + locomotion) broadcast spatially across
every pixel. The video channel is z-scored with the session's shipped video statistics (a shared scalar);
each behavior channel is z-scored with its own shipped statistics *inside* the tensor, so the
``norm_mean``/``norm_std`` stored on the container describe only the video channel.
"""

import os
from pathlib import Path

import numpy as np
from einops import rearrange
from tqdm.auto import tqdm

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.data_io.qiu_2026 import constants as C
from openretina.data_io.qiu_2026.trials import (
    STREAM_TIME_AXIS,
    discover_sessions,
    load_trial_array,
    read_condition_hash,
    read_stimulus_type,
    read_tiers,
    resolve_session_path,
    session_key,
    test_conditions,
    train_val_indices,
    trim_time,
    valid_length,
)
from openretina.utils.file_utils import get_local_file_path


def _read_norm_stats(session_path: Path) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (video_mean, video_std, behavior_mean[3], behavior_std[3]) from the shipped statistics."""
    stat = session_path / "meta" / "statistics"
    video_mean = float(np.load(stat / "videos" / "all" / "mean.npy").flat[0])
    video_std = float(np.load(stat / "videos" / "all" / "std.npy").flat[0])
    behavior_mean = np.asarray(np.load(stat / "behavior" / "all" / "mean.npy")[..., 0], dtype=np.float32)
    behavior_std = np.asarray(np.load(stat / "behavior" / "all" / "std.npy")[..., 0], dtype=np.float32)
    return video_mean, video_std, behavior_mean, behavior_std


def _build_input_tensor(
    video: np.ndarray,  # (H, W, T) already trimmed to valid length
    behavior: np.ndarray,  # (3, T) already trimmed to valid length
    video_mean: float,
    video_std: float,
    behavior_mean: np.ndarray,  # (3,)
    behavior_std: np.ndarray,  # (3,)
    behavior_channels: tuple[int, ...],
) -> np.ndarray:
    """Normalize and assemble one trial into a ``(1 + n_behavior, T, H, W)`` float32 tensor."""
    height, width, n_time = video.shape

    video_norm = (video.astype(np.float32) - video_mean) / video_std
    video_norm = rearrange(video_norm, "h w t -> 1 t h w")

    channels = list(behavior_channels)
    beh = behavior[channels, :].astype(np.float32)  # (n, T)
    beh = (beh - behavior_mean[channels][:, None]) / behavior_std[channels][:, None]
    beh = rearrange(beh, "c t -> c t 1 1")
    beh = np.broadcast_to(beh, (len(channels), n_time, height, width))

    return np.concatenate([video_norm, beh], axis=0).astype(np.float32)


def load_stimuli_for_session(
    session_path: str | os.PathLike,
    *,
    behavior_channels: tuple[int, ...] = C.BEHAVIOR_CHANNELS,
    stimulus_type: str = "clip",
) -> MoviesTrainTestSplit:
    """Build a :class:`MoviesTrainTestSplit` for one qiu_2026 session.

    The continuous train movie concatenates all train+validation trials (each trimmed to its valid
    length) in file-index order; ``test_dict`` holds one clip per ``condition_hash``, averaged over the
    condition's repeats (the movie is identical across repeats; averaging also pools the behavior traces
    to pair with the trial-averaged responses). Only trials matching ``stimulus_type`` (see
    :func:`~openretina.data_io.qiu_2026.trials.read_stimulus_type`) are included; some sessions' ``test``
    tier additionally contains non-clip functional-characterization trials with per-repeat frame-count
    jitter that would otherwise break the repeat-stacking below.
    """
    session_path = resolve_session_path(session_path)
    tiers = read_tiers(session_path)
    condition_hash = read_condition_hash(session_path)
    stimulus_types = read_stimulus_type(session_path)
    video_mean, video_std, behavior_mean, behavior_std = _read_norm_stats(session_path)

    def build_trial(i: int) -> np.ndarray:
        video = load_trial_array(session_path, "videos", i)
        behavior = load_trial_array(session_path, "behavior", i)
        n_valid = valid_length(video, STREAM_TIME_AXIS["videos"])
        video = trim_time(video, STREAM_TIME_AXIS["videos"], n_valid)
        behavior = trim_time(behavior, STREAM_TIME_AXIS["behavior"], n_valid)
        return _build_input_tensor(
            video, behavior, video_mean, video_std, behavior_mean, behavior_std, behavior_channels
        )

    train = np.concatenate([build_trial(i) for i in train_val_indices(tiers, stimulus_types, stimulus_type)], axis=1)

    test_dict: dict[str, np.ndarray] = {}
    for condition, indices in test_conditions(tiers, condition_hash, stimulus_types, stimulus_type).items():
        repeats = np.stack([build_trial(i) for i in indices], axis=0)  # (repeats, C, T, H, W)
        # TODO(qiu_2026): the video is identical across repeats, but the behavior channels are not.
        # Averaging them here pairs the test input with the trial-averaged responses; revisit whether
        # per-repeat inputs are more appropriate. Tracked in qiu_2026_integration_plan.md (Open Questions).
        test_dict[condition] = repeats.mean(axis=0).astype(np.float32)

    return MoviesTrainTestSplit(
        train=train,
        test_dict=test_dict,
        stim_id="qiu_2026",
        norm_mean=video_mean,
        norm_std=video_std,
    )


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    *,
    behavior_channels: tuple[int, ...] = C.BEHAVIOR_CHANNELS,
    stimulus_type: str = "clip",
    sessions: list[str] | None = None,
) -> dict[str, MoviesTrainTestSplit]:
    """Load every discovered qiu_2026 session into a ``{session_key: MoviesTrainTestSplit}`` dict."""
    base_data_path = Path(get_local_file_path(str(base_data_path)))
    stimuli_all_sessions: dict[str, MoviesTrainTestSplit] = {}
    for path in tqdm(discover_sessions(base_data_path, sessions), desc="qiu_2026 stimuli"):
        stimuli_all_sessions[session_key(path)] = load_stimuli_for_session(
            path, behavior_channels=behavior_channels, stimulus_type=stimulus_type
        )
    return stimuli_all_sessions
