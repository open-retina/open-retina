"""Tests for the qiu_2026 in-silico stimulus constants and the range-measurement helper."""

import numpy as np
import pytest

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.data_io.qiu_2026.constants import (
    BEHAVIOR_SWEEP_RANGE,
    STIMULUS_RANGE_CONSTRAINTS,
    VIDEO_SHAPE,
    video_range_and_norm,
)
from openretina.data_io.qiu_2026.stimuli import measure_video_range


def _split(video: np.ndarray, n_behavior_channels: int = 2) -> MoviesTrainTestSplit:
    """Wrap a (t, h, w) video as a qiu-shaped train movie with constant behavior channels."""
    behavior = np.zeros((n_behavior_channels, *video.shape), dtype=np.float32)
    train = np.concatenate([video[None].astype(np.float32), behavior], axis=0)
    # `MoviesTrainTestSplit` insists on exactly one of `test` / `test_dict`; the helper only ever
    # reads `.train`, so a one-frame stub keeps the container happy without affecting anything.
    return MoviesTrainTestSplit(train=train, test=train[:, :1], stim_id="test", norm_mean=0.0, norm_std=1.0)


def test_measured_constants_are_signed_correctly() -> None:
    """Guards against a transcription typo: a dropped minus sign here would be invisible downstream."""
    assert STIMULUS_RANGE_CONSTRAINTS["x_min_video"] < 0 < STIMULUS_RANGE_CONSTRAINTS["x_max_video"]
    assert STIMULUS_RANGE_CONSTRAINTS["rms_video"] > 0
    # The raw movie is 8-bit, so the z-scored video cannot exceed 255/std for any plausible std.
    assert -5 < STIMULUS_RANGE_CONSTRAINTS["x_min_video"] < -0.5
    assert 1.0 < STIMULUS_RANGE_CONSTRAINTS["x_max_video"] < 6.0
    # The video is z-scored, so its RMS must sit near 1.
    assert 0.5 < STIMULUS_RANGE_CONSTRAINTS["rms_video"] < 1.5
    assert BEHAVIOR_SWEEP_RANGE == (-2.0, 2.0)


def test_video_range_and_norm_derives_the_norm_from_the_stimulus_size() -> None:
    min_max_values, norm = video_range_and_norm(50)

    assert min_max_values == [(STIMULUS_RANGE_CONSTRAINTS["x_min_video"], STIMULUS_RANGE_CONSTRAINTS["x_max_video"])]
    assert len(min_max_values) == 1, "the optimized tensor is single-channel"
    expected = STIMULUS_RANGE_CONSTRAINTS["rms_video"] * np.sqrt(50 * VIDEO_SHAPE[1] * VIDEO_SHAPE[2])
    assert norm == pytest.approx(expected)
    assert norm == pytest.approx(338.05, abs=0.01)


def test_norm_scales_with_the_square_root_of_the_element_count() -> None:
    _, norm_50 = video_range_and_norm(50)
    _, norm_200 = video_range_and_norm(200)
    assert norm_200 / norm_50 == pytest.approx(2.0)

    _, norm_small = video_range_and_norm(50, height=18, width=32)
    assert norm_50 / norm_small == pytest.approx(2.0)


def test_rms_factor_scales_the_norm_linearly() -> None:
    _, baseline = video_range_and_norm(50)
    _, doubled = video_range_and_norm(50, rms_factor=2.0)
    assert doubled == pytest.approx(2.0 * baseline)


@pytest.mark.parametrize(
    "kwargs",
    [{"time_steps": 0}, {"time_steps": 10, "height": 0}, {"time_steps": 10, "width": -1}],
)
def test_video_range_and_norm_rejects_degenerate_shapes(kwargs) -> None:
    with pytest.raises(ValueError, match="must all be >= 1"):
        video_range_and_norm(**kwargs)


def test_measure_video_range_pools_across_sessions() -> None:
    """Lowest low percentile, highest high percentile, RMS over every pixel of every session."""
    rng = np.random.default_rng(0)
    movies = {
        "narrow": _split(rng.uniform(-1.0, 1.0, size=(40, 6, 8))),
        "wide": _split(rng.uniform(-3.0, 4.0, size=(40, 6, 8))),
    }

    measured = measure_video_range(movies, percentiles=(0.0, 100.0))

    assert measured["x_min_video"] == pytest.approx(movies["wide"].train[0].min())
    assert measured["x_max_video"] == pytest.approx(movies["wide"].train[0].max())
    pooled = np.concatenate([split.train[0].ravel() for split in movies.values()])
    assert measured["rms_video"] == pytest.approx(np.sqrt(np.mean(pooled**2)), rel=1e-6)


def test_measure_video_range_ignores_the_behavior_channels() -> None:
    """The behavior channels are huge constants; including them would wreck both range and RMS."""
    video = np.full((10, 4, 5), 0.5, dtype=np.float32)
    split = _split(video)
    split.train[1:] = 100.0

    measured = measure_video_range({"only": split}, percentiles=(0.0, 100.0))

    assert measured["x_max_video"] == pytest.approx(0.5)
    assert measured["rms_video"] == pytest.approx(0.5)


def test_measure_video_range_percentiles_exclude_outliers() -> None:
    video = np.zeros((100, 10, 10), dtype=np.float32)
    video[0, 0, 0] = 1000.0  # one pixel in 10000 -> above the 99.9th percentile

    measured = measure_video_range({"only": _split(video)}, percentiles=(0.1, 99.9))

    assert measured["x_max_video"] < 1.0, "a single saturated pixel must not set the range"


def test_measure_video_range_subsampling_is_reported_and_bounded(caplog) -> None:
    rng = np.random.default_rng(1)
    movies = {"big": _split(rng.normal(size=(200, 10, 10)).astype(np.float32))}

    with caplog.at_level("INFO", logger="openretina.data_io.qiu_2026.stimuli"):
        subsampled = measure_video_range(movies, max_elements_per_session=1000)
    full = measure_video_range(movies, max_elements_per_session=None)

    assert any("subsampled" in record.getMessage() for record in caplog.records), (
        "dropping pixels must be logged, never silent"
    )
    # The RMS always uses every pixel, so it must be identical either way.
    assert subsampled["rms_video"] == pytest.approx(full["rms_video"])
    assert subsampled["x_max_video"] == pytest.approx(full["x_max_video"], rel=0.2)


def test_measure_video_range_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="nothing to measure"):
        measure_video_range({})
