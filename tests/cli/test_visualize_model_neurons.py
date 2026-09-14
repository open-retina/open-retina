"""Tests for ``openretina.cli.visualize_model_neurons``."""

import pytest
import torch

from openretina.cli.visualize_model_neurons import _get_min_max_values_and_norm
from openretina.insilico.stimulus_optimization.regularizer import ChangeNormJointlyClipRangeSeparately


@pytest.mark.parametrize("num_channels", [1, 3, 4, 5])
def test_one_range_pair_per_channel(num_channels: int) -> None:
    """The list is indexed by channel downstream, so it needs one entry per channel.

    It used to return a single `(None, None)` regardless of `num_channels`, which happened to be
    right only for 1-channel models.
    """
    min_max_values, norm = _get_min_max_values_and_norm(num_channels)

    assert len(min_max_values) == num_channels
    assert all(pair == (None, None) for pair in min_max_values)
    assert norm is None


def test_two_channel_path_is_unchanged() -> None:
    """The hoefling_2024 green/UV case keeps its measured constraints and its norm."""
    min_max_values, norm = _get_min_max_values_and_norm(2)

    assert len(min_max_values) == 2
    assert norm is not None
    assert all(low is not None and high is not None for low, high in min_max_values)


@pytest.mark.parametrize("num_channels", [1, 3, 4])
def test_result_composes_with_the_norm_postprocessor(num_channels: int) -> None:
    """`ChangeNormJointlyClipRangeSeparately` asserts the list length against the stimulus.

    This is the assert the old return value tripped for any model that was neither 1- nor
    2-channel -- qiu_2026's 3 input channels being the case that surfaced it.
    """
    min_max_values, norm = _get_min_max_values_and_norm(num_channels)
    postprocessor = ChangeNormJointlyClipRangeSeparately(min_max_values, norm)

    stimulus = torch.randn(1, num_channels, 5, 4, 4)
    processed = postprocessor.process(stimulus)

    assert processed.shape == stimulus.shape
