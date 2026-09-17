import os.path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from openretina.utils.plotting import (
    numpy_to_mp4_video,
    plot_stimulus_composition,
    plot_vector_field_resp_iso,
    save_stimulus_to_mp4_video,
)


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


def _vector_field_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients and responses shaped for ``plot_vector_field_resp_iso`` on the grid (x, y)."""
    gradient_dict = np.ones((2, len(x), len(y)))
    resp_dict = np.outer(np.arange(len(x), dtype=float), np.arange(len(y), dtype=float))
    return gradient_dict, resp_dict


def test_plot_vector_field_resp_iso_uses_the_y_axis_for_y() -> None:
    """`np.meshgrid(x, x)` drew the response surface on the x range in BOTH directions.

    `y` was a declared parameter the body never read, so any grid whose two axes differ -- the
    normal case once the function is used for anything but the symmetric [-1, 1] chromatic
    contrast grid -- put the contours at the wrong coordinates while the arrows, which do read
    `y`, stayed correct. Nothing raised; the figure was simply wrong.
    """
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(10.0, 12.0, 5)
    gradient_dict, resp_dict = _vector_field_inputs(x, y)

    fig = plot_vector_field_resp_iso(x, y, gradient_dict, resp_dict)
    ax = fig.gca()

    # The drawn data must lie within the y range, not the x range. Under the bug the contour
    # surface spanned [-1, 1] vertically, dragging the lower bound far below y.min().
    assert ax.dataLim.y0 >= y.min() - 0.5, f"y data starts at {ax.dataLim.y0}, below the y axis {y.min()}"
    assert ax.dataLim.y1 <= y.max() + 0.5, f"y data ends at {ax.dataLim.y1}, above the y axis {y.max()}"
    plt.close(fig)


def test_plot_vector_field_resp_iso_rejects_a_mismatched_grid() -> None:
    """A gradient grid that does not match the axes is a caller error, not a silent mis-plot."""
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    gradient_dict, resp_dict = _vector_field_inputs(x, np.linspace(-1.0, 1.0, 4))

    with pytest.raises(ValueError, match="grid shape"):
        plot_vector_field_resp_iso(x, y, gradient_dict, resp_dict)


@pytest.mark.parametrize("num_channels", [1, 2, 3, 4, 6])
def test_plot_stimulus_composition_handles_any_channel_count(num_channels: int) -> None:
    """Channels are not necessarily colours, so >3 of them must plot rather than raise.

    The colour maps only key 1/2/3; indexing them directly raised `KeyError: 4` for a 4-channel
    model, and `visualize_model_neurons` hits this call outside its try/except -- i.e. after the
    full MEI optimisation has already been paid for.
    """
    # >18 frames: the frequency panel lowpass-filters the temporal trace and needs the padlen.
    stimulus = np.random.rand(num_channels, 50, 18, 16)
    fig, axes = plt.subplots(2, 2)

    plot_stimulus_composition(
        stimulus=stimulus,
        temporal_trace_ax=axes[0, 0],
        freq_ax=axes[0, 1],
        spatial_ax=axes[1, 0],
    )

    # One temporal trace per channel, each in a distinct colour.
    lines = axes[0, 0].get_lines()
    assert len(lines) == num_channels
    assert len({line.get_color() for line in lines}) == num_channels
    plt.close(fig)
