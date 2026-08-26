"""Tests for the stimulus regularization losses.

The smoothness loss exists because the shipped MEI recipe had no working regularizer at all:
`RangeRegularizationLoss` is provably inert whenever `ChangeNormJointlyClipRangeSeparately` runs as
the postprocessor, since the projection enforces range and norm *before* every forward pass. The
first test below pins that, so the reason this module gained a second loss stays on the record.
"""

import torch

from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
    SmoothnessRegularizationLoss,
    ZeroOutsideMaskProcessor,
)

MIN_MAX = [(-1.29, 2.96)]
NORM = 338.05360994966463
SHAPE = (1, 1, 12, 16, 20)


def _white(shape=SHAPE, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator)


def test_range_loss_is_inert_under_the_norm_projection() -> None:
    """The postprocessor already guarantees range and norm, so the range loss can never fire."""
    postprocessor = ChangeNormJointlyClipRangeSeparately(min_max_values=MIN_MAX, norm=NORM)
    loss = RangeRegularizationLoss(min_max_values=MIN_MAX, max_norm=NORM, factor=0.1)

    stimulus = postprocessor.process(_white() * 0.1)
    assert float(loss.forward(stimulus)) == 0.0

    # ... and it stays zero after any number of projected steps.
    for _ in range(5):
        stimulus = postprocessor.process(stimulus + _white(seed=1) * 0.5)
        assert float(loss.forward(stimulus)) == 0.0


def test_smoothness_loss_is_inert_by_default() -> None:
    """Both weights default to 0.0, so adding it to a call site changes nothing until asked."""
    loss = SmoothnessRegularizationLoss()
    assert float(loss.forward(_white())) == 0.0


def test_white_noise_hits_the_analytic_ceiling() -> None:
    """Normalized, each axis' term is 2 * (1 - lag-1 autocorrelation); white noise gives 2 per axis."""
    loss = SmoothnessRegularizationLoss(factor_spatial=1.0)
    value = float(loss.forward(_white(shape=(1, 1, 8, 64, 64))))
    assert abs(value - 4.0) < 0.1, value


def test_spatially_constant_stimulus_has_no_spatial_penalty() -> None:
    stimulus = _white(shape=(1, 1, 12, 1, 1)).expand(1, 1, 12, 16, 20).contiguous()
    loss = SmoothnessRegularizationLoss(factor_spatial=1.0)
    assert float(loss.forward(stimulus)) < 1e-6


def test_smooth_stimulus_is_penalized_less_than_white_noise() -> None:
    loss = SmoothnessRegularizationLoss(factor_spatial=1.0)
    white = _white(shape=(1, 1, 8, 32, 32))
    kernel = torch.ones(1, 1, 1, 5, 5) / 25.0
    smooth = torch.nn.functional.conv3d(white, kernel, padding=(0, 2, 2))
    assert float(loss.forward(smooth)) < 0.5 * float(loss.forward(white))


def test_loss_is_invariant_to_the_contrast_target() -> None:
    """The point of dividing by the mean square: `factor` means the same at every `rms_factor` rung."""
    loss = SmoothnessRegularizationLoss(factor_spatial=1.0, factor_temporal=1.0)
    stimulus = _white()
    assert abs(float(loss.forward(stimulus * 10.0)) - float(loss.forward(stimulus))) < 1e-4


def test_temporal_and_spatial_terms_are_independent() -> None:
    """A stimulus that is smooth in time but white in space is penalized only spatially."""
    stimulus = _white(shape=(1, 1, 1, 32, 32)).expand(1, 1, 8, 32, 32).contiguous()
    assert float(SmoothnessRegularizationLoss(factor_temporal=1.0).forward(stimulus)) < 1e-6
    assert float(SmoothnessRegularizationLoss(factor_spatial=1.0).forward(stimulus)) > 1.0


def test_loss_is_differentiable_and_has_a_nonzero_gradient() -> None:
    """Unlike the range loss under projection, this one actually reaches the stimulus."""
    stimulus = _white().requires_grad_(True)
    SmoothnessRegularizationLoss(factor_spatial=1.0, factor_temporal=1.0).forward(stimulus).backward()
    assert stimulus.grad is not None
    assert float(stimulus.grad.norm()) > 0.0


def test_mask_processor_zeroes_the_surround() -> None:
    mask = torch.zeros(16, 20, dtype=torch.bool)
    mask[4:8, 5:11] = True
    processor = ZeroOutsideMaskProcessor(mask)

    out = processor.process(_white())
    assert float(out[..., ~mask].abs().max()) == 0.0
    assert float(out[..., mask].abs().max()) > 0.0


def test_mask_before_norm_puts_the_whole_norm_on_the_mask() -> None:
    """The documented chaining order: mask, then renormalize over what is left."""
    mask = torch.zeros(16, 20, dtype=torch.bool)
    mask[4:8, 5:11] = True
    target = 50.0
    stimulus = _white()
    for processor in (ZeroOutsideMaskProcessor(mask), ChangeNormJointlyClipRangeSeparately([(None, None)], target)):
        stimulus = processor.process(stimulus)

    assert abs(float(torch.linalg.vector_norm(stimulus)) - target) < 1e-3
    assert float(stimulus[..., ~mask].abs().max()) == 0.0


def test_smoothness_loss_reaches_outside_the_mask() -> None:
    """Why the mask has to be re-applied every step, not just at init."""
    mask = torch.zeros(16, 20, dtype=torch.bool)
    mask[4:8, 5:11] = True
    stimulus = (_white() * mask).requires_grad_(True)
    SmoothnessRegularizationLoss(factor_spatial=1.0).forward(stimulus).backward()
    assert stimulus.grad is not None
    assert float(stimulus.grad[..., ~mask].abs().max()) > 0.0
