"""qiu_2026 stimulus loader.

Builds a ``(C, T, H, W)`` input where ``C = 1 video + len(behavior_channels) behavior`` channels: the
grayscale movie plus the selected behavior traces (pupil size + locomotion) broadcast spatially across
every pixel. The video channel is z-scored with the session's shipped video statistics (a shared scalar);
each behavior channel is z-scored with its own shipped statistics *inside* the tensor, so the
``norm_mean``/``norm_std`` stored on the container describe only the video channel.

Repeated test stimuli additionally keep their behavior per repeat (``test_behavior_by_repeat_dict``),
since the movie is identical across a condition's repeats but the animal's behavior is not.
"""

import os
import warnings
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


def _warn_on_video_deviation(
    session_path: Path,
    condition: str,
    indices: list[int],
    videos: list[np.ndarray],
    tolerance: float,
) -> None:
    """Warn if a condition's repeats do not look like the same clip.

    A ``condition_hash`` is supposed to identify one stimulus, and the test path leans on that: the
    repeats' videos are averaged into one test input and their responses into one trial-averaged target.
    The shipped repeats are *not* bit-identical, though -- on ``dynamic28188-16-3`` the worst pair within
    a condition differs by a mean of ~0.01-0.02 (in units of the video std, which is what these arrays
    are scaled to), while two genuinely different clips differ by ~1.1. The default tolerance sits in
    that gap, so this stays quiet on sound data and fires if a condition ever mixes two stimuli.
    """
    if len(videos) < 2 or tolerance <= 0.0:
        return
    reference = videos[0]
    for index, video in zip(indices[1:], videos[1:]):
        deviation = float(np.abs(video - reference).mean())
        if deviation > tolerance:
            warnings.warn(
                f"Session {session_path.name}, condition {condition}: trials {indices[0]} and {index} "
                f"share a condition_hash but their videos differ by a mean of {deviation:.3f} video "
                f"standard deviations (tolerance {tolerance}). Averaging them into one test stimulus, "
                f"and their responses into one trial-averaged target, assumes they are the same clip.",
                category=UserWarning,
                stacklevel=3,
            )
            return


def _normalize_video(
    video: np.ndarray,  # (H, W, T) already trimmed to valid length
    video_mean: float,
    video_std: float,
) -> np.ndarray:
    """Z-score one trial's video and reorder it to ``(1, T, H, W)``."""
    video_norm = (video.astype(np.float32) - video_mean) / video_std
    return rearrange(video_norm, "h w t -> 1 t h w")


def _normalize_behavior(
    behavior: np.ndarray,  # (3, T) already trimmed to valid length
    behavior_mean: np.ndarray,  # (3,)
    behavior_std: np.ndarray,  # (3,)
    behavior_channels: tuple[int, ...],
) -> np.ndarray:
    """Z-score the selected behavior channels of one trial into ``(n_behavior, T)``."""
    channels = list(behavior_channels)
    beh = behavior[channels, :].astype(np.float32)  # (n, T)
    return (beh - behavior_mean[channels][:, None]) / behavior_std[channels][:, None]


def _assemble_input_tensor(
    video_norm: np.ndarray,  # (1, T, H, W)
    behavior_norm: np.ndarray,  # (n_behavior, T)
) -> np.ndarray:
    """Broadcast normalized behavior across every pixel and stack it onto the video channel."""
    _, n_time, height, width = video_norm.shape
    beh = rearrange(behavior_norm, "c t -> c t 1 1")
    beh = np.broadcast_to(beh, (behavior_norm.shape[0], n_time, height, width))
    return np.concatenate([video_norm, beh], axis=0).astype(np.float32)



def load_stimuli_for_session(
    session_path: str | os.PathLike,
    *,
    behavior_channels: tuple[int, ...] = C.BEHAVIOR_CHANNELS,
    stimulus_type: str = "clip",
    video_deviation_tolerance: float = 0.1,
) -> MoviesTrainTestSplit:
    """Build a :class:`MoviesTrainTestSplit` for one qiu_2026 session.

    The continuous train movie concatenates all train+validation trials (each trimmed to its valid
    length) in file-index order. ``test_dict`` holds one clip per ``condition_hash``, averaged over the
    condition's repeats in both the video and the behavior channels.

    Averaging the video is near enough to a no-op: the repeats are the same clip, differing only by
    storage noise (see :func:`_warn_on_video_deviation`). Averaging the *behavior* is not — the animal
    behaves differently on every repeat, and the model is nonlinear in those channels. So
    ``test_behavior_by_repeat_dict`` additionally keeps each repeat's behavior as
    ``(repeats, n_behavior, T)``, letting the dataloader feed them through the model one at a time and
    average the *predictions* instead (see ``qiu_2026_dataloaders``).

    Only trials matching ``stimulus_type`` (see
    :func:`~openretina.data_io.qiu_2026.trials.read_stimulus_type`) are included; some sessions' ``test``
    tier additionally contains non-clip functional-characterization trials with per-repeat frame-count
    jitter that would otherwise break the repeat-stacking below.

    Args:
        video_deviation_tolerance: Warn when a condition's repeats differ by more than this mean
            absolute deviation, in units of the video std. See :func:`_warn_on_video_deviation`.
    """
    session_path = resolve_session_path(session_path)
    tiers = read_tiers(session_path)
    condition_hash = read_condition_hash(session_path)
    stimulus_types = read_stimulus_type(session_path)
    video_mean, video_std, behavior_mean, behavior_std = _read_norm_stats(session_path)

    def load_video(i: int) -> tuple[np.ndarray, int]:
        """Return the trial's normalized ``(1, T, H, W)`` video and its valid frame count."""
        video = load_trial_array(session_path, "videos", i)
        n_valid = valid_length(video, STREAM_TIME_AXIS["videos"])
        video = trim_time(video, STREAM_TIME_AXIS["videos"], n_valid)
        return _normalize_video(video, video_mean, video_std), n_valid

    def load_behavior(i: int, n_valid: int) -> np.ndarray:
        """Return the trial's normalized ``(n_behavior, T)`` behavior, trimmed to ``n_valid`` frames."""
        behavior = load_trial_array(session_path, "behavior", i)
        n_behavior_valid = valid_length(behavior, STREAM_TIME_AXIS["behavior"])
        if n_behavior_valid != n_valid:
            raise ValueError(
                f"Trial {i} of session {session_path.name}: behavior has {n_behavior_valid} valid frames "
                f"but the video has {n_valid}. The streams must be frame-aligned to stack repeats."
            )
        behavior = trim_time(behavior, STREAM_TIME_AXIS["behavior"], n_valid)
        return _normalize_behavior(behavior, behavior_mean, behavior_std, behavior_channels)

    def build_trial(i: int) -> np.ndarray:
        video_norm, n_valid = load_video(i)
        return _assemble_input_tensor(video_norm, load_behavior(i, n_valid))

    train = np.concatenate([build_trial(i) for i in train_val_indices(tiers, stimulus_types, stimulus_type)], axis=1)

    test_dict: dict[str, np.ndarray] = {}
    test_behavior_by_repeat_dict: dict[str, np.ndarray] = {}
    for condition, indices in test_conditions(tiers, condition_hash, stimulus_types, stimulus_type).items():
        videos: list[np.ndarray] = []
        behaviors: list[np.ndarray] = []
        for i in indices:
            video_norm, n_valid = load_video(i)
            if videos and video_norm.shape != videos[0].shape:
                raise ValueError(
                    f"Session {session_path.name}, condition {condition}: trial {i} has video shape "
                    f"{video_norm.shape} but trial {indices[0]} has {videos[0].shape}; repeats of one "
                    f"condition must share a frame count to be stacked."
                )
            videos.append(video_norm)
            behaviors.append(load_behavior(i, n_valid))

        behavior_by_repeat = np.stack(behaviors, axis=0)  # (R, n_behavior, T)
        _warn_on_video_deviation(session_path, condition, indices, videos, video_deviation_tolerance)
        test_dict[condition] = _assemble_input_tensor(np.mean(videos, axis=0), behavior_by_repeat.mean(axis=0))
        test_behavior_by_repeat_dict[condition] = behavior_by_repeat.astype(np.float32)

    return MoviesTrainTestSplit(
        train=train,
        test_dict=test_dict,
        stim_id="qiu_2026",
        norm_mean=video_mean,
        norm_std=video_std,
        test_behavior_by_repeat_dict=test_behavior_by_repeat_dict,
    )


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    *,
    behavior_channels: tuple[int, ...] = C.BEHAVIOR_CHANNELS,
    stimulus_type: str = "clip",
    sessions: list[str] | None = None,
    video_deviation_tolerance: float = 0.1,
) -> dict[str, MoviesTrainTestSplit]:
    """Load every discovered qiu_2026 session into a ``{session_key: MoviesTrainTestSplit}`` dict."""
    base_data_path = Path(get_local_file_path(str(base_data_path)))
    stimuli_all_sessions: dict[str, MoviesTrainTestSplit] = {}
    for path in tqdm(discover_sessions(base_data_path, sessions), desc="qiu_2026 stimuli"):
        stimuli_all_sessions[session_key(path)] = load_stimuli_for_session(
            path,
            behavior_channels=behavior_channels,
            stimulus_type=stimulus_type,
            video_deviation_tolerance=video_deviation_tolerance,
        )
    return stimuli_all_sessions
