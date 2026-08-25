"""Tests for the optional `pupil_center` argument on the in-silico objectives.

Two properties matter. First, that leaving it out changes nothing at all: every existing
(shifter-free) caller must produce a byte-identical model call. Second, that supplying it
actually reaches the shifter and moves the readout.
"""

import pytest
import torch
import torch.nn as nn

from openretina.insilico.stimulus_optimization.objective import (
    ContrastiveNeuronObjective,
    IncreaseObjective,
    InnerNeuronVisualizationObjective,
    MeanReducer,
    SliceMeanReducer,
)

IN_SHAPE = (3, 20, 16, 18)  # channels, time, height, width
N_NEURONS = 3


class _RecordingModel(nn.Module):
    """Records the exact keyword arguments of every call; returns a (batch, t, neurons) response."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x, **kwargs):
        self.calls.append(kwargs)
        per_frame = x.mean(dim=(1, 3, 4)) * self.weight  # (batch, t)
        return per_frame.unsqueeze(-1).expand(-1, -1, N_NEURONS)


def _reducer() -> SliceMeanReducer:
    return SliceMeanReducer(axis=0, start=0, length=4)


def _stimulus(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, *IN_SHAPE)


def test_omitting_pupil_center_leaves_the_model_call_untouched() -> None:
    model = _RecordingModel()

    IncreaseObjective(model, 0, None, _reducer()).forward(_stimulus())
    IncreaseObjective(model, 0, "sess_a", _reducer()).forward(_stimulus())

    assert model.calls[0] == {}, "single-session objectives must call the model with no kwargs at all"
    assert model.calls[1] == {"data_key": "sess_a"}


def test_pupil_center_is_broadcast_over_batch_and_input_time() -> None:
    model = _RecordingModel()
    batch = 2

    IncreaseObjective(model, 0, "sess_a", _reducer(), pupil_center=(0.5, -1.25)).forward(_stimulus(batch))

    pupil = model.calls[0]["pupil_center"]
    # Aligned with the *input* time dimension: BaseCoreReadout.forward drops the leading frames
    # the core consumes itself.
    assert pupil.shape == (batch, 2, IN_SHAPE[1])
    assert torch.equal(pupil[:, 0], torch.full((batch, IN_SHAPE[1]), 0.5))
    assert torch.equal(pupil[:, 1], torch.full((batch, IN_SHAPE[1]), -1.25))


@pytest.mark.parametrize(
    "pupil_center",
    [(0.5, -1.25), [0.5, -1.25], torch.tensor([0.5, -1.25]), torch.tensor([[0.5, -1.25]])],
)
def test_pupil_center_accepts_sequences_and_tensors(pupil_center) -> None:
    model = _RecordingModel()

    IncreaseObjective(model, 0, "sess_a", _reducer(), pupil_center=pupil_center).forward(_stimulus())

    assert model.calls[0]["pupil_center"][0, :, 0].tolist() == [0.5, -1.25]


def test_pupil_center_without_data_key_raises() -> None:
    with pytest.raises(ValueError, match="pupil_center requires a data_key"):
        IncreaseObjective(_RecordingModel(), 0, None, _reducer(), pupil_center=(0.0, 0.0))


@pytest.mark.parametrize(
    "make_objective",
    [
        lambda model, pupil: IncreaseObjective(model, 0, "sess_a", _reducer(), pupil_center=pupil),
        lambda model, pupil: ContrastiveNeuronObjective(model, [0], [[1, 2]], "sess_a", MeanReducer(0), 1.6, pupil),
    ],
    ids=["increase", "contrastive"],
)
def test_every_objective_subclass_forwards_pupil_center(make_objective) -> None:
    model = _RecordingModel()

    make_objective(model, (0.5, -1.25)).forward(_stimulus())

    assert model.calls[-1]["pupil_center"][0, :, 0].tolist() == [0.5, -1.25]


def test_inner_neuron_objective_forwards_pupil_center() -> None:
    """Checked through `model_forward` directly: its `forward` needs a layer/channel selection."""
    model = _RecordingModel()
    objective = InnerNeuronVisualizationObjective(model, "sess_a", _reducer(), pupil_center=(0.5, -1.25))

    objective.model_forward(_stimulus())

    assert model.calls[-1]["pupil_center"][0, :, 0].tolist() == [0.5, -1.25]


def test_zero_pupil_center_matches_none_for_an_untrained_shifter(tiny_qiu_model) -> None:
    """`MLPShifter.initialize` zeroes every bias, so an untrained shifter maps (0, 0) to exactly 0.

    That makes this the *wrong* input to test "pupil changes the output" with -- and the right one
    to test the opposite. The companion test below uses a non-zero position.
    """
    stimulus = _stimulus()
    without = IncreaseObjective(tiny_qiu_model, 0, "sess_a", _reducer())
    centered = IncreaseObjective(tiny_qiu_model, 0, "sess_a", _reducer(), pupil_center=(0.0, 0.0))

    assert torch.equal(without.forward(stimulus), centered.forward(stimulus))


def test_nonzero_pupil_center_moves_the_response(tiny_qiu_model) -> None:
    stimulus = _stimulus()
    centered = IncreaseObjective(tiny_qiu_model, 0, "sess_a", _reducer(), pupil_center=(0.0, 0.0))
    shifted = IncreaseObjective(tiny_qiu_model, 0, "sess_a", _reducer(), pupil_center=(1.0, -1.0))

    assert not torch.allclose(centered.forward(stimulus), shifted.forward(stimulus))


def test_pupil_center_is_ignored_by_a_model_without_a_shifter(tiny_qiu_model_factory) -> None:
    """Regression guard mirroring `tests/models/test_core_readout_shifter.py`."""
    model = tiny_qiu_model_factory(with_shifter=False)
    model.eval()
    stimulus = _stimulus()

    without = IncreaseObjective(model, 0, "sess_a", _reducer())
    with_pupil = IncreaseObjective(model, 0, "sess_a", _reducer(), pupil_center=(1.0, -1.0))

    assert torch.equal(without.forward(stimulus), with_pupil.forward(stimulus))
