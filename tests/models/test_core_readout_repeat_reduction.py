"""Tests for averaging predictions over a batch of stimulus repeats in `test_step`.

A repeated test stimulus is one clip shown many times. The movie is identical on every repeat, but the
animal's behavior is not, and the model is nonlinear in behavior -- so `mean_r f(video, b_r)` is not
`f(video, mean_r b_r)`. These tests pin that the model reduces predictions (not inputs) when a batch is
marked, and that unmarked batches from every other dataset are untouched.

Dataset-agnostic on purpose: `test_step` reads the marker via `getattr`, so these build their own
duck-typed datapoints rather than importing the qiu_2026 ones.
"""

from collections import namedtuple

import pytest
import torch

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import SimpleCoreWrapper
from openretina.modules.readout.multi_readout import MultiSampledGaussianReadout
from openretina.modules.shifters.mlp_shifter import MultiSessionMLPShifter

N_NEURONS_DICT = {"sess1": 5}
IN_SHAPE = (3, 20, 36, 64)  # channels, time, height, width
N_REPEATS = 6

_RepeatDataPoint = namedtuple("_RepeatDataPoint", ["inputs", "targets", "pupil_center", "reduce_over_batch"])


def _amplify(model: BaseCoreReadout) -> None:
    """Scale the weights up so the untrained model is not degenerate.

    At init the core's conv weights are ~0.008, so after two layers its output is ~1e-5 and every
    readout saturates at `softplus(bias)`: a constant, for which averaging inputs and averaging
    predictions trivially agree. Scaling the weights puts the model in a regime where its
    nonlinearities actually bite, which is what these tests are about. The factors are arbitrary; they
    only need to produce a non-constant output (asserted below).
    """
    with torch.no_grad():
        for name, parameter in model.core.named_parameters():
            if "weight" in name and "norm" not in name:
                parameter.mul_(20.0)
        if model.shifter is not None:
            for name, parameter in model.shifter.named_parameters():
                if "weight" in name:
                    parameter.mul_(5.0)


def _build_model(shifter: bool = True, test_repeat_chunk_size: int = 4) -> BaseCoreReadout:
    core = SimpleCoreWrapper(
        channels=(IN_SHAPE[0], 8, 8),
        temporal_kernel_sizes=(5, 5),
        spatial_kernel_sizes=(5, 5),
        gamma_input=0.0,
        gamma_temporal=0.0,
        gamma_in_sparse=0.0,
        gamma_hidden=0.0,
        input_padding=False,
        hidden_padding=True,
        cut_first_n_frames=0,
    )
    readout_in_shape = BaseCoreReadout.compute_readout_input_shape(
        BaseCoreReadout.__new__(BaseCoreReadout), IN_SHAPE, core
    )
    readout = MultiSampledGaussianReadout(
        in_shape=readout_in_shape,
        n_neurons_dict=N_NEURONS_DICT,
        bias=True,
        init_mu_range=0.1,
        init_sigma_range=0.3,
        gauss_type="full",
        grid_mean_predictor=None,
        gamma=1.0,
    )
    shifter_module = (
        MultiSessionMLPShifter(n_neurons_dict=N_NEURONS_DICT, input_channels=2, hidden_channels=5, n_layers=3)
        if shifter
        else None
    )
    model = BaseCoreReadout(
        core=core,
        readout=readout,
        learning_rate=1e-3,
        shifter=shifter_module,
        test_repeat_chunk_size=test_repeat_chunk_size,
    )
    _amplify(model)
    # eval(), as Lightning's test loop runs it: the core's norm layers then use running statistics, so
    # each sample is independent of the rest of its batch and chunking the forward cannot change it.
    model.eval()
    return model


def _t_out(model: BaseCoreReadout) -> int:
    return model.core(torch.zeros(1, *IN_SHAPE)).size(2)


def _make_repeat_batch(model: BaseCoreReadout, seed: int = 0) -> _RepeatDataPoint:
    """One video repeated N_REPEATS times, with per-repeat behavior channels and pupil traces."""
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 1, *IN_SHAPE[1:], generator=generator).expand(N_REPEATS, -1, -1, -1, -1)
    behavior = torch.randn(N_REPEATS, IN_SHAPE[0] - 1, IN_SHAPE[1], 1, 1, generator=generator).expand(
        -1, -1, -1, *IN_SHAPE[2:]
    )
    targets = torch.rand(1, _t_out(model), N_NEURONS_DICT["sess1"], generator=generator).expand(N_REPEATS, -1, -1)
    return _RepeatDataPoint(
        inputs=torch.cat([video, behavior], dim=1).contiguous(),
        targets=targets.contiguous(),
        pupil_center=torch.randn(N_REPEATS, 2, IN_SHAPE[1], generator=generator),
        reduce_over_batch=torch.ones(N_REPEATS, dtype=torch.bool),
    )


def test_test_step_averages_predictions_over_the_repeat_batch() -> None:
    """The scored prediction must be the mean of the per-repeat forwards, not a per-repeat score."""
    model = _build_model()
    batch = _make_repeat_batch(model)

    captured: list[torch.Tensor] = []
    original_evaluation_loss_forward = model.evaluation_loss.forward

    def capture(output, target):
        captured.append(output)
        return original_evaluation_loss_forward(output, target)

    model.evaluation_loss.forward = capture  # type: ignore[method-assign]
    with torch.no_grad():
        model.test_step(("sess1", batch), 0)  # type: ignore[arg-type]

        expected = torch.stack(
            [
                model.forward(batch.inputs[r : r + 1], "sess1", pupil_center=batch.pupil_center[r : r + 1])
                for r in range(N_REPEATS)
            ]
        ).mean(dim=0)

    assert len(captured) == 1
    scored = captured[0]
    assert scored.shape == (1, _t_out(model), N_NEURONS_DICT["sess1"])
    assert torch.allclose(scored, expected, atol=1e-5)


def test_averaging_predictions_differs_from_averaging_inputs() -> None:
    """The reason this path exists: the model is nonlinear in the behavior channels and the pupil."""
    model = _build_model()
    batch = _make_repeat_batch(model)

    with torch.no_grad():
        averaged_predictions = torch.stack(
            [
                model.forward(batch.inputs[r : r + 1], "sess1", pupil_center=batch.pupil_center[r : r + 1])
                for r in range(N_REPEATS)
            ]
        ).mean(dim=0)
        prediction_from_averaged_input = model.forward(
            batch.inputs.mean(dim=0, keepdim=True),
            "sess1",
            pupil_center=batch.pupil_center.mean(dim=0, keepdim=True),
        )

    # Guard against passing for the wrong reason: a saturated model outputs a constant, for which the
    # two agree trivially.
    assert averaged_predictions.std() > 1e-3, "model is degenerate; the comparison below means nothing"
    assert not torch.allclose(averaged_predictions, prediction_from_averaged_input, atol=1e-3)


@pytest.mark.parametrize("chunk_size", [0, 1, 4, N_REPEATS, N_REPEATS + 10])
def test_chunked_forward_does_not_change_the_reduced_prediction(chunk_size: int) -> None:
    """`test_repeat_chunk_size` is a memory knob only: every value must give the same average."""
    model = _build_model(test_repeat_chunk_size=chunk_size)
    batch = _make_repeat_batch(model)

    with torch.no_grad():
        chunked = model._forward_in_chunks(batch.inputs, "sess1", batch.pupil_center, chunk_size)
        unchunked = model.forward(batch.inputs, "sess1", pupil_center=batch.pupil_center)

    assert chunked.shape == unchunked.shape
    assert torch.allclose(chunked, unchunked, atol=1e-5)


def test_test_step_leaves_unmarked_batches_alone() -> None:
    """Regression guard for every other dataset: a plain 2-field DataPoint must score per batch item."""
    model = _build_model(shifter=False)
    batch_size = 3
    data_point = DataPoint(
        inputs=torch.randn(batch_size, *IN_SHAPE),
        targets=torch.rand(batch_size, _t_out(model), N_NEURONS_DICT["sess1"]),
    )

    captured: list[torch.Tensor] = []
    original_evaluation_loss_forward = model.evaluation_loss.forward

    def capture(output, target):
        captured.append(output)
        return original_evaluation_loss_forward(output, target)

    model.evaluation_loss.forward = capture  # type: ignore[method-assign]
    with torch.no_grad():
        loss = model.test_step(("sess1", data_point), 0)

    assert torch.isfinite(loss)
    assert captured[0].shape[0] == batch_size, "an unmarked batch must not be reduced"
