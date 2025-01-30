import os.path
import tempfile

import numpy as np
import pytest

from openretina.utils.plotting import numpy_to_mp4_video, save_stimulus_to_mp4_video


@pytest.mark.parametrize(
    "stimulus_shape",
    [
        (1, 5, 7, 11),
        (2, 5, 7, 11),
        (3, 5, 7, 11),
    ],
)
def test_save_stimulus_to_mp4_video(stimulus_shape: tuple[int, ...]) -> None:
    stimulus = np.random.rand(*stimulus_shape)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
        save_stimulus_to_mp4_video(stimulus, filepath=temp_file.name)
        assert os.path.exists(temp_file.name)


@pytest.mark.parametrize(
    "stimulus_shape",
    [
        (11, 7, 5, 3),
    ],
)
def test_numpy_to_mp4_video(stimulus_shape: tuple[int, ...]) -> None:
    video = np.random.rand(*stimulus_shape)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
        numpy_to_mp4_video(video, temp_file.name, display_video=False)
        assert os.path.exists(temp_file.name)
