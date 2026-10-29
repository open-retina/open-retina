"""Tests for ``openretina.data_io.hoefling_2024.responses``."""

import numpy as np
import pytest

from openretina.data_io.hoefling_2024.responses import upsample_all_responses, upsample_traces

MOVIE_STIM_ID = 5
FRAMES_PER_CLIP = 150


def _recording(n_clips: int = 3, n_rois: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip triggers ~5 s apart, and random traces sampled at the imaging rate (128 ms) with a per-ROI phase."""
    triggertimes = 10.0 + 4.9666667 * np.arange(n_clips)
    tracestimes = np.stack([np.arange(0.0, 30.0, 0.128) + 0.002 * (roi + 1) for roi in range(n_rois)])
    traces = np.asarray(np.random.default_rng(0).random(size=tracestimes.shape))
    return triggertimes, traces, tracestimes


def test_zero_delay_keeps_the_frame_times() -> None:
    triggertimes, traces, tracestimes = _recording()
    frame_times = np.concatenate([np.linspace(t, t + 4.9666667, FRAMES_PER_CLIP) for t in triggertimes])
    expected = np.stack([np.interp(frame_times, times, trace) for times, trace in zip(tracestimes, traces)])

    upsampled = upsample_traces(triggertimes, traces, tracestimes, stim_id=MOVIE_STIM_ID, norm_by_std=False)

    np.testing.assert_array_equal(upsampled, expected)


@pytest.mark.parametrize("delay", [1, 3, 6])
def test_delay_shifts_responses_by_whole_frames(delay: int) -> None:
    triggertimes, traces, tracestimes = _recording()
    undelayed = upsample_traces(triggertimes, traces, tracestimes, stim_id=MOVIE_STIM_ID, norm_by_std=False)
    delayed = upsample_traces(
        triggertimes, traces, tracestimes, stim_id=MOVIE_STIM_ID, norm_by_std=False, response_delay_frames=delay
    )

    # Frame t of the delayed responses is frame t - delay of the undelayed ones
    for clip in range(len(triggertimes)):
        clip_delayed = delayed[:, clip * FRAMES_PER_CLIP : (clip + 1) * FRAMES_PER_CLIP]
        clip_undelayed = undelayed[:, clip * FRAMES_PER_CLIP : (clip + 1) * FRAMES_PER_CLIP]
        np.testing.assert_allclose(clip_delayed[:, delay:], clip_undelayed[:, :-delay], atol=1e-6)


def test_upsample_all_responses_passes_the_delay_on() -> None:
    triggertimes, traces, tracestimes = _recording()
    data_dict = {
        "session": {
            "natural_spikes": traces,
            "natural_trigger_times": triggertimes[None],
            "natural_traces_times": tracestimes,
        }
    }

    result = upsample_all_responses(data_dict, norm_by_std=False, response_delay_frames=3)

    expected = upsample_traces(
        triggertimes, traces, tracestimes, stim_id=MOVIE_STIM_ID, norm_by_std=False, response_delay_frames=3
    )
    np.testing.assert_array_equal(result["session"]["responses_final"], expected)
