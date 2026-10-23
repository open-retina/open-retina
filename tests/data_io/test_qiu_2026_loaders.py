"""Loader-level tests for qiu_2026, driven by a synthetic FileTree written to tmp_path.

The train-wiring tests construct `MoviesTrainTestSplit` / pupil dicts directly, so they never execute
`load_stimuli_for_session` or `load_pupil_for_session`. These do, on a miniature session laid out
exactly like the real one (`data/<stream>/{i}.npy` positionally aligned to `meta/trials/*.npy`, with
trailing whole-frame NaN padding), so the test-condition handling is covered without the multi-GB
dataset.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from openretina.data_io.qiu_2026 import constants as C
from openretina.data_io.qiu_2026.pupil import load_pupil_for_session
from openretina.data_io.qiu_2026.stimuli import (
    _assemble_input_tensor, _normalize_behavior, _normalize_video, load_stimuli_for_session
)
HEIGHT, WIDTH = 4, 6
CLIP_LENGTH = 10  # valid frames per trial
PAD = 3  # trailing NaN-padded frames, as the real FileTree ships
N_BEHAVIOR_STREAMS = 3  # pupil size, d(pupil size), locomotion; BEHAVIOR_CHANNELS selects 2 of them
N_TRAIN_TRIALS = 2
N_REPEATS = 4


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _pad(array: np.ndarray, time_axis: int) -> np.ndarray:
    """Append `PAD` whole-frame-NaN frames along `time_axis`, as the shipped trial arrays are stored."""
    shape = list(array.shape)
    shape[time_axis] = PAD
    return np.concatenate([array, np.full(shape, np.nan, dtype=np.float32)], axis=time_axis)


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """A miniature session: 2 train trials plus one test condition repeated `N_REPEATS` times.

    The test repeats share a video (as the real condition_hash guarantees) but carry per-repeat
    behavior and pupil traces.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "dynamic00000-0-0-Fluorescence-abc1234-v4a"
    n_trials = N_TRAIN_TRIALS + N_REPEATS

    tiers = np.array(["train"] * N_TRAIN_TRIALS + ["test"] * N_REPEATS)
    _write(root / "meta" / "trials" / "tiers.npy", tiers)
    _write(
        root / "meta" / "trials" / "condition_hash.npy",
        np.array([f"train{i}" for i in range(N_TRAIN_TRIALS)] + ["cond1"] * N_REPEATS),
    )
    _write(root / "meta" / "trials" / "clip_movie_name.npy", np.array(["movie"] * n_trials))

    for name, value in [("mean", 0.25), ("std", 2.0)]:
        _write(root / "meta" / "statistics" / "videos" / "all" / f"{name}.npy", np.array([value], dtype=np.float32))
        _write(
            root / "meta" / "statistics" / "behavior" / "all" / f"{name}.npy",
            np.full((N_BEHAVIOR_STREAMS, 1), value, dtype=np.float32),
        )
        _write(
            root / "meta" / "statistics" / "pupil_center" / "all" / f"{name}.npy",
            np.full((2, 1), value, dtype=np.float32),
        )

    shared_video = rng.standard_normal((HEIGHT, WIDTH, CLIP_LENGTH)).astype(np.float32)
    for i in range(n_trials):
        is_test = i >= N_TRAIN_TRIALS
        video = shared_video if is_test else rng.standard_normal((HEIGHT, WIDTH, CLIP_LENGTH)).astype(np.float32)
        _write(root / "data" / "videos" / f"{i}.npy", _pad(video, 2))
        # Behavior and eye position differ on every trial, test repeats included.
        _write(
            root / "data" / "behavior" / f"{i}.npy",
            _pad(rng.standard_normal((N_BEHAVIOR_STREAMS, CLIP_LENGTH)).astype(np.float32), 1),
        )
        _write(
            root / "data" / "pupil_center" / f"{i}.npy",
            _pad(rng.standard_normal((2, CLIP_LENGTH)).astype(np.float32), 1),
        )
    return root


def test_load_stimuli_keeps_behavior_per_test_repeat(session: Path) -> None:
    movies = load_stimuli_for_session(session)

    n_channels = 1 + len(C.BEHAVIOR_CHANNELS)
    assert movies.train.shape == (n_channels, CLIP_LENGTH * N_TRAIN_TRIALS, HEIGHT, WIDTH)
    assert np.isfinite(movies.train).all(), "NaN padding must be trimmed off"

    assert set(movies.test_dict) == {"cond1"} == set(movies.test_behavior_by_repeat_dict)
    assert movies.test_dict["cond1"].shape == (n_channels, CLIP_LENGTH, HEIGHT, WIDTH)

    behavior_by_repeat = movies.test_behavior_by_repeat_dict["cond1"]
    assert behavior_by_repeat.shape == (N_REPEATS, len(C.BEHAVIOR_CHANNELS), CLIP_LENGTH)
    assert np.isfinite(behavior_by_repeat).all()
    assert not np.array_equal(behavior_by_repeat[0], behavior_by_repeat[-1]), "repeats must differ"

    # test_dict still carries the across-repeat mean behavior, broadcast over every pixel.
    assert np.allclose(movies.test_dict["cond1"][1:, :, 0, 0], behavior_by_repeat.mean(axis=0), atol=1e-6)
    for channel in range(1, n_channels):
        plane = movies.test_dict["cond1"][channel]
        assert np.allclose(plane, plane[:, :1, :1], atol=1e-6), "behavior must be spatially constant"


def test_test_dict_matches_the_repeat_averaged_input_recipe(session: Path) -> None:
    """Reading the video once per condition must give what averaging full per-repeat tensors gave."""
    movies = load_stimuli_for_session(session)

    stat = session / "meta" / "statistics"
    video_mean = float(np.load(stat / "videos" / "all" / "mean.npy").flat[0])
    video_std = float(np.load(stat / "videos" / "all" / "std.npy").flat[0])
    behavior_mean = np.load(stat / "behavior" / "all" / "mean.npy")[..., 0].astype(np.float32)
    behavior_std = np.load(stat / "behavior" / "all" / "std.npy")[..., 0].astype(np.float32)

    repeats = []
    for i in range(N_TRAIN_TRIALS, N_TRAIN_TRIALS + N_REPEATS):
        video = np.load(session / "data" / "videos" / f"{i}.npy")[:, :, :CLIP_LENGTH]
        behavior = np.load(session / "data" / "behavior" / f"{i}.npy")[:, :CLIP_LENGTH]
        repeats.append(
            _assemble_input_tensor(
            _normalize_video(video, video_mean, video_std),
            _normalize_behavior(behavior, behavior_mean, behavior_std, C.BEHAVIOR_CHANNELS),
            )
        )
    assert np.allclose(movies.test_dict["cond1"], np.stack(repeats).mean(axis=0), atol=1e-6)


def test_load_pupil_keeps_every_repeat(session: Path) -> None:
    pupil = load_pupil_for_session(session)

    train_trace = pupil["train"]
    assert isinstance(train_trace, np.ndarray)
    assert train_trace.shape == (2, CLIP_LENGTH * N_TRAIN_TRIALS)
    by_repeat = pupil["test_by_repeat_dict"]["cond1"]
    assert by_repeat.shape == (N_REPEATS, 2, CLIP_LENGTH)
    assert np.isfinite(by_repeat).all()
    assert not np.array_equal(by_repeat[0], by_repeat[-1]), "eye position must differ across repeats"
    # The averaged trace is kept too, for the per_repeat_test_behavior=False path.
    assert np.allclose(pupil["test_dict"]["cond1"], by_repeat.mean(axis=0), atol=1e-6)


def test_small_video_deviation_between_repeats_is_tolerated(session: Path) -> None:
    """Real repeats are not bit-identical: on dynamic28188-16-3 they differ by ~0.01-0.02 video std.

    An exact-equality check would therefore reject every real session. Only a deviation large enough
    to mean "a different clip" should be reported.
    """
    last = N_TRAIN_TRIALS + N_REPEATS - 1
    video = np.load(session / "data" / "videos" / f"{last}.npy")
    rng = np.random.default_rng(1)
    video[:, :, :CLIP_LENGTH] += rng.normal(0, 0.02 * 2.0, (HEIGHT, WIDTH, CLIP_LENGTH))  # 0.02 * video std
    np.save(session / "data" / "videos" / f"{last}.npy", video)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        movies = load_stimuli_for_session(session)
    assert "cond1" in movies.test_dict


def test_videos_from_different_clips_within_a_condition_warn(session: Path) -> None:
    """A condition_hash is supposed to mean one stimulus; mixing two silently would be worse."""
    last = N_TRAIN_TRIALS + N_REPEATS - 1
    rng = np.random.default_rng(2)
    video = np.load(session / "data" / "videos" / f"{last}.npy")
    video[:, :, :CLIP_LENGTH] = rng.standard_normal((HEIGHT, WIDTH, CLIP_LENGTH)) * 2.0 * 3.0  # unrelated clip
    np.save(session / "data" / "videos" / f"{last}.npy", video)

    with pytest.warns(UserWarning, match="share a condition_hash"):
        load_stimuli_for_session(session)

    # The tolerance is tunable, and 0 disables the check entirely.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_stimuli_for_session(session, video_deviation_tolerance=0.0)


def test_behavior_not_frame_aligned_with_video_raises(session: Path) -> None:
    """The streams are padded identically; if they are not, repeats would stack misaligned."""
    trial = N_TRAIN_TRIALS
    behavior = np.load(session / "data" / "behavior" / f"{trial}.npy")
    behavior[:, CLIP_LENGTH - 2 :] = np.nan  # two fewer valid frames than the video
    np.save(session / "data" / "behavior" / f"{trial}.npy", behavior)

    with pytest.raises(ValueError, match="valid frames"):
        load_stimuli_for_session(session)
