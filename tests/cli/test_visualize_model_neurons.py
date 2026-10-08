"""Tests for ``openretina.cli.visualize_model_neurons``."""

import pytest
import torch

from openretina.cli.visualize_model_neurons import _get_min_max_values_and_norm
from openretina.insilico.stimulus_optimization.regularizer import ChangeNormJointlyClipRangeSeparately


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
