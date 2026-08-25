"""Tests for the behavior-state modulation sweeps.

Correctness (not just shapes) is checked against `LinearBehaviorStub`, whose response is an exactly
known linear function of the behavior channels -- the only way to validate the numbers without a
trained network.
"""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from openretina.insilico.stimulus_optimization.objective import MeanReducer  # noqa: E402
from openretina.insilico.tuning_analyses.behavior_modulation import (  # noqa: E402
    DEFAULT_SWEEP_VALUES,
    ResponseGrid,
    behavior_response_grid,
    pupil_center_response_grid,
    shift_to_core_pixels,
    shifter_shift_grid,
)

IN_SHAPE = (3, 20, 16, 18)  # channels, time, height, width
SMALL_GRID = np.array([-1.0, 0.0, 1.0, 2.0])


def _video() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(IN_SHAPE[1], IN_SHAPE[2], IN_SHAPE[3])


def test_grid_shapes_and_axis_bookkeeping(linear_behavior_stub) -> None:
    grid = behavior_response_grid(
        linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
    )

    n_neurons = linear_behavior_stub.n_neurons_dict["sess_a"]
    assert grid.responses.shape == (len(SMALL_GRID), len(SMALL_GRID), n_neurons)
    assert grid.gradients is not None
    assert grid.gradients.shape == (2, len(SMALL_GRID), len(SMALL_GRID), n_neurons)
    assert grid.n_neurons == n_neurons
    assert np.array_equal(grid.axis_values[0], SMALL_GRID)
    assert grid.axis_names == ("pupil size [z]", "locomotion [z]")


def test_default_grid_is_plus_minus_two_standard_deviations(linear_behavior_stub) -> None:
    grid = behavior_response_grid(linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0))

    assert grid.responses.shape[:2] == (17, 17)
    assert np.array_equal(grid.axis_values[0], DEFAULT_SWEEP_VALUES)
    assert (grid.axis_values[0][0], grid.axis_values[0][-1]) == (-2.0, 2.0)


def test_responses_match_the_stub_analytically(linear_behavior_stub) -> None:
    video = _video()
    grid = behavior_response_grid(
        linear_behavior_stub, "sess_a", video, MeanReducer(axis=0), behavior_values=SMALL_GRID
    )

    bias = linear_behavior_stub.bias["sess_a"].detach().numpy()
    video_weight = linear_behavior_stub.video_weight["sess_a"].detach().numpy()
    behavior_weight = linear_behavior_stub.behavior_weight["sess_a"].detach().numpy()
    video_mean = float(video.mean())

    for i, value_0 in enumerate(SMALL_GRID):
        for j, value_1 in enumerate(SMALL_GRID):
            expected = (
                bias + video_weight * video_mean + behavior_weight[:, 0] * value_0 + behavior_weight[:, 1] * value_1
            )
            np.testing.assert_allclose(grid.responses[i, j], expected, rtol=1e-5, atol=1e-5)


def test_finite_difference_gradients_recover_the_analytic_slope(linear_behavior_stub) -> None:
    """The stub is linear in the behavior values, so central differences must be exact."""
    grid = behavior_response_grid(
        linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
    )

    behavior_weight = linear_behavior_stub.behavior_weight["sess_a"].detach().numpy()
    assert grid.gradients is not None
    for axis in (0, 1):
        expected = np.broadcast_to(behavior_weight[:, axis], grid.gradients[axis].shape)
        np.testing.assert_allclose(grid.gradients[axis], expected, rtol=1e-4, atol=1e-4)


def test_behavior_channels_are_actually_written_into_the_stimulus(linear_behavior_stub) -> None:
    """Spy on the assembled input: the swept values must land on the declared channels only."""
    video = _video()
    behavior_values = np.array([-1.5, 0.75])

    behavior_response_grid(
        linear_behavior_stub,
        "sess_a",
        video,
        MeanReducer(axis=0),
        behavior_values=behavior_values,
        grid_batch_size=1,
    )

    # One grid point per forward, iterating axis 0 outer and axis 1 inner.
    inputs = linear_behavior_stub.inputs
    assert len(inputs) == behavior_values.size**2
    expected_points = [(-1.5, -1.5), (-1.5, 0.75), (0.75, -1.5), (0.75, 0.75)]
    for stimulus, (value_0, value_1) in zip(inputs, expected_points, strict=True):
        assert torch.equal(stimulus[0, 0], video), "the video channel must be untouched"
        assert torch.equal(stimulus[0, 1], torch.full_like(stimulus[0, 1], value_0))
        assert torch.equal(stimulus[0, 2], torch.full_like(stimulus[0, 2], value_1))


def test_neuron_indices_subset_the_output(linear_behavior_stub) -> None:
    full = behavior_response_grid(
        linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
    )
    subset = behavior_response_grid(
        linear_behavior_stub,
        "sess_a",
        _video(),
        MeanReducer(axis=0),
        behavior_values=SMALL_GRID,
        neuron_indices=[2, 0],
    )

    assert subset.responses.shape[-1] == 2
    np.testing.assert_allclose(subset.responses[..., 0], full.responses[..., 2], rtol=1e-6)
    np.testing.assert_allclose(subset.responses[..., 1], full.responses[..., 0], rtol=1e-6)


def test_pupil_center_is_relayed_to_the_model(linear_behavior_stub) -> None:
    behavior_response_grid(
        linear_behavior_stub,
        "sess_a",
        _video(),
        MeanReducer(axis=0),
        behavior_values=SMALL_GRID,
        pupil_center=(0.25, -0.5),
    )

    pupil = linear_behavior_stub.last_pupil_center
    assert pupil is not None
    assert pupil.shape[1:] == (2, IN_SHAPE[1])
    assert pupil[0, :, 0].tolist() == [0.25, -0.5]


def test_pupil_center_is_omitted_when_not_requested(linear_behavior_stub) -> None:
    behavior_response_grid(linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID)
    assert linear_behavior_stub.last_pupil_center is None


@pytest.mark.parametrize("grid_batch_size", [1, 3, 64])
def test_chunking_does_not_change_the_result(linear_behavior_stub, grid_batch_size) -> None:
    reference = behavior_response_grid(
        linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
    )
    chunked = behavior_response_grid(
        linear_behavior_stub,
        "sess_a",
        _video(),
        MeanReducer(axis=0),
        behavior_values=SMALL_GRID,
        grid_batch_size=grid_batch_size,
    )
    np.testing.assert_allclose(chunked.responses, reference.responses, rtol=1e-6)


def test_training_mode_is_refused(linear_behavior_stub) -> None:
    linear_behavior_stub.train()
    with pytest.raises(ValueError, match="training mode"):
        behavior_response_grid(
            linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
        )


def test_invalid_behavior_channels_are_rejected(linear_behavior_stub) -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        behavior_response_grid(linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_channels=(1, 1))
    with pytest.raises(ValueError, match="outside the model's channel range"):
        behavior_response_grid(linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_channels=(1, 7))


def test_single_valued_axis_is_rejected(linear_behavior_stub) -> None:
    with pytest.raises(ValueError, match="at least 2 values"):
        behavior_response_grid(linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=[0.0])


def test_grid_on_the_real_model_is_deterministic_and_non_degenerate(tiny_qiu_model) -> None:
    first = behavior_response_grid(tiny_qiu_model, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID)
    second = behavior_response_grid(tiny_qiu_model, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID)

    assert np.array_equal(first.responses, second.responses)
    assert first.responses.std() > 0, "the behavior channels must actually change the response"


def test_for_neuron_layout_feeds_the_vector_field_plot(linear_behavior_stub) -> None:
    from openretina.utils.plotting import plot_vector_field_resp_iso

    grid = behavior_response_grid(
        linear_behavior_stub, "sess_a", _video(), MeanReducer(axis=0), behavior_values=SMALL_GRID
    )
    responses, gradients = grid.for_neuron(1)

    assert responses.shape == (len(SMALL_GRID), len(SMALL_GRID))
    assert gradients is not None and gradients.shape == (2, len(SMALL_GRID), len(SMALL_GRID))
    np.testing.assert_allclose(responses, grid.responses[:, :, 1])

    figure = plot_vector_field_resp_iso(
        grid.axis_values[0],
        grid.axis_values[1],
        gradients,
        responses,
        xlabel=grid.axis_names[0],
        ylabel=grid.axis_names[1],
        tick_locations=None,
    )
    assert figure.axes[0].get_xlabel() == "pupil size [z]"


def test_compute_gradients_can_be_switched_off(linear_behavior_stub) -> None:
    grid = behavior_response_grid(
        linear_behavior_stub,
        "sess_a",
        _video(),
        MeanReducer(axis=0),
        behavior_values=SMALL_GRID,
        compute_gradients=False,
    )
    assert grid.gradients is None
    assert grid.for_neuron(0)[1] is None


def test_modulation_index() -> None:
    axis = np.array([0.0, 1.0])
    responses = np.zeros((2, 2, 3))
    responses[..., 0] = [[1.0, 1.0], [1.0, 1.0]]  # flat -> 0
    responses[..., 1] = [[1.0, 3.0], [1.0, 3.0]]  # (3-1)/(3+1) = 0.5
    responses[..., 2] = [[-1.0, 1.0], [-1.0, 1.0]]  # max + min == 0 -> nan, not inf
    grid = ResponseGrid(axis_values=(axis, axis), axis_names=("a", "b"), responses=responses)

    index = grid.modulation_index()

    assert index[0] == pytest.approx(0.0)
    assert index[1] == pytest.approx(0.5)
    assert np.isnan(index[2])


def test_pupil_center_grid_requires_a_shifter(tiny_qiu_model_factory) -> None:
    model = tiny_qiu_model_factory(with_shifter=False)
    model.eval()
    with pytest.raises(ValueError, match="no shifter"):
        pupil_center_response_grid(
            model, "sess_a", torch.randn(*IN_SHAPE), MeanReducer(axis=0), pupil_values=SMALL_GRID
        )


def test_pupil_center_grid_varies_with_eye_position(tiny_qiu_model) -> None:
    torch.manual_seed(5)
    stimulus = torch.randn(*IN_SHAPE)

    grid = pupil_center_response_grid(tiny_qiu_model, "sess_a", stimulus, MeanReducer(axis=0), pupil_values=SMALL_GRID)

    assert grid.responses.shape == (len(SMALL_GRID), len(SMALL_GRID), 4)
    assert grid.axis_names == ("pupil x [z]", "pupil y [z]")
    assert grid.responses.std() > 0, "moving the eye must move the readout"


def test_shifter_shift_grid_is_zero_at_the_session_mean(tiny_qiu_model) -> None:
    """`MLPShifter.initialize` zeroes every bias, so an untrained shifter maps (0, 0) to exactly 0."""
    shifts = shifter_shift_grid(tiny_qiu_model)

    assert set(shifts) == set(tiny_qiu_model.shifter.keys())
    zero_index = int(np.argmin(np.abs(DEFAULT_SWEEP_VALUES)))
    assert DEFAULT_SWEEP_VALUES[zero_index] == 0.0
    for session, shift in shifts.items():
        assert shift.shape == (len(DEFAULT_SWEEP_VALUES), len(DEFAULT_SWEEP_VALUES), 2), session
        np.testing.assert_allclose(shift[zero_index, zero_index], 0.0, atol=1e-7)
    assert np.abs(shifts["sess_a"]).max() > 0, "a non-zero eye position must produce a non-zero shift"


def test_shifter_shift_grid_can_select_sessions(tiny_qiu_model) -> None:
    shifts = shifter_shift_grid(tiny_qiu_model, data_keys=["sess_b"], pupil_values=SMALL_GRID)
    assert list(shifts) == ["sess_b"]
    assert shifts["sess_b"].shape == (len(SMALL_GRID), len(SMALL_GRID), 2)


def test_shifter_shift_grid_requires_a_shifter(tiny_qiu_model_factory) -> None:
    model = tiny_qiu_model_factory(with_shifter=False)
    with pytest.raises(ValueError, match="no shifter"):
        shifter_shift_grid(model)


@pytest.mark.parametrize(
    "align_corners, expected",
    [
        # (height, width) = (10, 20); width span 19 or 20, height span 9 or 10.
        (True, [0.5 * 19 / 2, -1.0 * 9 / 2]),
        (False, [0.5 * 20 / 2, -1.0 * 10 / 2]),
    ],
)
def test_shift_to_core_pixels(align_corners, expected) -> None:
    pixels = shift_to_core_pixels(np.array([0.5, -1.0]), (10, 20), align_corners=align_corners)
    np.testing.assert_allclose(pixels, expected)


def test_shift_to_core_pixels_accepts_a_full_core_shape_and_broadcasts() -> None:
    shift = np.zeros((4, 5, 2))
    shift[..., 0] = 1.0  # full-width displacement
    pixels = shift_to_core_pixels(shift, (8, 3, 10, 20))  # (c, t, h, w) -> uses (10, 20)

    assert pixels.shape == (4, 5, 2)
    np.testing.assert_allclose(pixels[..., 0], (20 - 1) / 2)
    np.testing.assert_allclose(pixels[..., 1], 0.0)


def test_shift_to_core_pixels_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="last axis"):
        shift_to_core_pixels(np.zeros((4, 3)), (10, 20))
    with pytest.raises(ValueError, match="at least 2 entries"):
        shift_to_core_pixels(np.zeros(2), (20,))
