"""Tests for `FixedChannelStimulusModel`, the channel-freezing wrapper used for qiu_2026 MEIs."""

from functools import partial

import pytest
import torch
import torch.nn as nn

from openretina.insilico.stimulus_optimization.fixed_channel_model import FixedChannelStimulusModel
from openretina.insilico.stimulus_optimization.objective import IncreaseObjective, SliceMeanReducer
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)

IN_SHAPE = (3, 20, 16, 18)  # channels, time, height, width
BEHAVIOR_CONSTANTS = {1: 0.0, 2: 1.5}


class _EchoModel(nn.Module):
    """Returns the input unchanged, recording the kwargs it was called with."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        # A parameter so `next(model.parameters())` works, as several helpers assume.
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x, data_key=None, pupil_center=None):
        self.calls.append({"data_key": data_key, "pupil_center": pupil_center})
        return x * self.scale


def _wrapped_echo(constant_channels=None) -> FixedChannelStimulusModel:
    return FixedChannelStimulusModel(
        _EchoModel(),
        BEHAVIOR_CONSTANTS if constant_channels is None else constant_channels,
        num_channels=IN_SHAPE[0],
    )


def test_assemble_places_constants_exactly() -> None:
    wrapper = _wrapped_echo()
    video = torch.randn(2, 1, 5, 4, 6)

    assembled = wrapper.assemble(video)

    assert assembled.shape == (2, 3, 5, 4, 6)
    assert torch.equal(assembled[:, 0], video[:, 0])
    assert torch.equal(assembled[:, 1], torch.zeros_like(assembled[:, 1]))
    assert torch.equal(assembled[:, 2], torch.full_like(assembled[:, 2], 1.5))


def test_assemble_interleaves_non_contiguous_optimized_channels() -> None:
    """Constant channels need not be trailing: channel order must be preserved exactly."""
    wrapper = FixedChannelStimulusModel(_EchoModel(), {0: -1.0, 2: 2.0}, num_channels=4)
    assert wrapper.optimized_channels == (1, 3)

    x = torch.randn(1, 2, 3, 2, 2)
    assembled = wrapper.assemble(x)

    assert torch.equal(assembled[:, 0], torch.full_like(assembled[:, 0], -1.0))
    assert torch.equal(assembled[:, 1], x[:, 0])
    assert torch.equal(assembled[:, 2], torch.full_like(assembled[:, 2], 2.0))
    assert torch.equal(assembled[:, 3], x[:, 1])


def test_assemble_is_differentiable_only_through_optimized_channels() -> None:
    wrapper = _wrapped_echo()
    video = torch.randn(1, 1, 5, 4, 6, requires_grad=True)

    wrapper.assemble(video).sum().backward()

    assert video.grad is not None
    assert torch.equal(video.grad, torch.ones_like(video))
    # The constants live in a buffer, so no optimizer can reach them.
    assert list(wrapper.named_buffers()) != []
    assert all(not name.endswith("constant_values") for name, _ in wrapper.named_parameters())


def test_forward_matches_base_model_on_the_assembled_input(tiny_qiu_model) -> None:
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)
    video = torch.randn(1, 1, *IN_SHAPE[1:])

    wrapped_output = wrapper(video, data_key="sess_a")
    direct_output = tiny_qiu_model(wrapper.assemble(video), data_key="sess_a")

    assert torch.equal(wrapped_output, direct_output)


def test_num_channels_inferred_from_data_info(tiny_qiu_model) -> None:
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)
    assert wrapper.num_channels == IN_SHAPE[0]
    assert wrapper.optimized_channels == (0,)


def test_stimulus_shape_drops_the_constant_channels(tiny_qiu_model) -> None:
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)

    assert tiny_qiu_model.stimulus_shape(50) == (1, 3, 50, IN_SHAPE[2], IN_SHAPE[3])
    assert wrapper.stimulus_shape(50) == (1, 1, 50, IN_SHAPE[2], IN_SHAPE[3])
    assert wrapper.stimulus_shape(50, num_batches=4)[0] == 4


def test_readout_keys_pass_through(tiny_qiu_model) -> None:
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)
    assert wrapper.readout_keys() == tiny_qiu_model.readout.readout_keys()


def test_pupil_center_is_relayed_only_when_given() -> None:
    wrapper = _wrapped_echo()
    video = torch.randn(1, 1, 5, 4, 6)
    pupil = torch.zeros(1, 2, 5)

    wrapper(video, data_key="sess_a")
    wrapper(video, data_key="sess_a", pupil_center=pupil)

    echo = wrapper.model
    assert isinstance(echo, _EchoModel)
    assert echo.calls[0]["pupil_center"] is None
    assert echo.calls[1]["pupil_center"] is pupil


@pytest.mark.parametrize(
    "constant_channels, num_channels, message",
    [
        ({3: 0.0}, 3, "outside the model's channel range"),
        ({-1: 0.0}, 3, "outside the model's channel range"),
        ({0: 0.0, 1: 0.0, 2: 0.0}, 3, "nothing left to optimize"),
    ],
)
def test_invalid_channel_specifications_raise(constant_channels, num_channels, message) -> None:
    with pytest.raises(ValueError, match=message):
        FixedChannelStimulusModel(_EchoModel(), constant_channels, num_channels=num_channels)


def test_num_channels_cannot_be_inferred_without_data_info() -> None:
    with pytest.raises(ValueError, match="Pass num_channels explicitly"):
        FixedChannelStimulusModel(_EchoModel(), BEHAVIOR_CONSTANTS)


def test_assemble_rejects_a_wrong_channel_count() -> None:
    wrapper = _wrapped_echo()
    with pytest.raises(ValueError, match="Expected 1 channels in dim 1"):
        wrapper.assemble(torch.randn(1, 3, 5, 4, 6))


def test_optimization_increases_the_objective_and_leaves_constants_untouched(tiny_qiu_model) -> None:
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)
    objective = IncreaseObjective(
        wrapper,
        neuron_indices=0,
        data_key="sess_a",
        response_reducer=SliceMeanReducer(axis=0, start=0, length=4),
    )
    stimulus = torch.randn(wrapper.stimulus_shape(time_steps=IN_SHAPE[1]), requires_grad=True)
    initial_score = float(objective.forward(stimulus))

    optimize_stimulus(
        stimulus,
        optimizer_init_fn=partial(torch.optim.SGD, lr=10.0),
        objective_object=objective,
        optimization_stopper=OptimizationStopper(max_iterations=10),
    )

    assert float(objective.forward(stimulus)) > initial_score
    assert stimulus.shape == (1, 1, *IN_SHAPE[1:]), "the optimized tensor stays single-channel"
    assembled = wrapper.assemble(stimulus)
    assert torch.equal(assembled[:, 1], torch.zeros_like(assembled[:, 1]))
    assert torch.equal(assembled[:, 2], torch.full_like(assembled[:, 2], 1.5))


def test_norm_postprocessor_composes_without_shrinking_the_video(tiny_qiu_model) -> None:
    """The point of the wrapper: the joint norm becomes the norm of the video alone."""
    wrapper = FixedChannelStimulusModel(tiny_qiu_model, BEHAVIOR_CONSTANTS)
    target_norm = 30.0
    postprocessor = ChangeNormJointlyClipRangeSeparately(min_max_values=[(-3.0, 3.0)], norm=target_norm)
    objective = IncreaseObjective(
        wrapper,
        neuron_indices=0,
        data_key="sess_a",
        response_reducer=SliceMeanReducer(axis=0, start=0, length=4),
    )
    stimulus = torch.randn(wrapper.stimulus_shape(time_steps=IN_SHAPE[1]), requires_grad=True)

    optimize_stimulus(
        stimulus,
        optimizer_init_fn=partial(torch.optim.SGD, lr=10.0),
        objective_object=objective,
        optimization_stopper=OptimizationStopper(max_iterations=20),
        stimulus_regularization_loss=RangeRegularizationLoss(
            min_max_values=[(-3.0, 3.0)], max_norm=target_norm, factor=0.1
        ),
        stimulus_postprocessor=postprocessor,
    )

    assert float(torch.linalg.vector_norm(stimulus)) == pytest.approx(target_norm, rel=0.2)
    assert float(stimulus.abs().max()) <= 3.0 + 1e-6


def test_freeze_after_renorm_would_shrink_the_video_geometrically() -> None:
    """Regression guard for the design this wrapper exists to avoid.

    Renormalizing a full 3-channel stimulus and *then* resetting the behavior channels to their
    constants leaves the behavior block at full size while the video is rescaled by the same
    factor every iteration -- so the video decays geometrically to zero. Nobody should have to
    rediscover that by watching an MEI fade out.
    """
    channels, time_steps, height, width = 3, 20, 16, 18
    behavior_value = 1.0
    video_norm = 30.0

    stimulus = torch.randn(1, channels, time_steps, height, width)
    stimulus[:, 0] *= video_norm / torch.linalg.vector_norm(stimulus[:, 0])
    stimulus[:, 1:] = behavior_value
    postprocessor = ChangeNormJointlyClipRangeSeparately(min_max_values=[(None, None)] * channels, norm=video_norm)

    norms = []
    for _ in range(5):
        stimulus = postprocessor.process(stimulus)
        stimulus[:, 1:] = behavior_value  # the "freeze" step
        norms.append(float(torch.linalg.vector_norm(stimulus[:, 0])))

    assert norms[0] < 0.9 * video_norm, "the very first joint renormalization already shrinks the video"
    assert all(later < earlier for earlier, later in zip(norms, norms[1:])), f"expected monotonic decay, got {norms}"
    assert norms[-1] < 0.05 * video_norm, f"expected near-total collapse after 5 steps, got {norms}"

    # The wrapper's route is immune: the postprocessor never sees the behavior channels at all.
    video_only = torch.randn(1, 1, time_steps, height, width)
    video_postprocessor = ChangeNormJointlyClipRangeSeparately(min_max_values=[(None, None)], norm=video_norm)
    for _ in range(5):
        video_only = video_postprocessor.process(video_only)
    assert float(torch.linalg.vector_norm(video_only)) == pytest.approx(video_norm, rel=1e-5)
