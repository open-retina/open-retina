import numpy as np
import pytest

from openretina.utils.sta_processing import extract_filters_from_sta


@pytest.mark.parametrize(("num_frames", "temporal_crop_frames"), [(1, 30), (10, 30), (10, 4), (10, None)])
def test_nan_sta_gives_zero_filters_of_the_usual_length(num_frames: int, temporal_crop_frames: int | None) -> None:
    """Zero filters must match the length a valid STA would get, so a session's filters can be stacked."""
    sta = np.random.default_rng(0).normal(size=(num_frames, 12, 10))
    _, valid_temporal, _ = extract_filters_from_sta(sta, temporal_crop_frames=temporal_crop_frames)

    sta[0, 0, 0] = np.nan
    spatial, temporal, params = extract_filters_from_sta(sta, temporal_crop_frames=temporal_crop_frames)

    assert temporal.shape == valid_temporal.shape
    assert not temporal.any() and not spatial.any()
    assert params["success"] is False


def test_signed_amplitude_keeps_the_polarity() -> None:
    sta = np.zeros((3, 12, 10))
    sta[1, 5, 4] = -1.0
    sta[1, 5, 5] = -0.5
    _, temporal, params = extract_filters_from_sta(sta)

    assert params["peak_frame"] == 1
    assert params["amplitude"] > 0
    assert params["signed_amplitude"] < 0
    assert np.sign(temporal[params["peak_frame"]]) == np.sign(params["signed_amplitude"])
