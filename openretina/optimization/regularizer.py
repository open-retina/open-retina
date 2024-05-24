from typing import List, Tuple, Optional, Iterable
from collections.abc import Iterable
import abc

import torch


def no_op_regularizer(stimulus: torch.Tensor) -> torch.Tensor:
    return torch.zeros([1])


"""
So far use the defaults for the mouse retina model
postprocessing_config = {
    "norm": 30,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}
"""
def range_regularizer_fn(
        stimulus: torch.Tensor,
        min_max_values: List[Tuple[float, float]] = [(-0.654, 6.269), (-0.913, 6.269)],
        max_norm: float = 30.0,
        factor: float = 0.1,
) -> torch.Tensor:
    """ Penalizes the stimulus if it is outside the range defined by min_max_values. """
    penalty = torch.zeros([1])
    for i, (min_val, max_val) in enumerate(min_max_values):
        stimulus_i = stimulus[:, i]
        min_penalty = torch.sum(torch.relu(min_val - stimulus_i))
        max_penalty = torch.sum(torch.relu(stimulus_i - max_val))
        penalty += min_penalty + max_penalty

    # Add a penalty such that the norm of the stimulus is lower than max_norm
    norm_penalty = torch.relu(torch.norm(stimulus) - max_norm)
    penalty += norm_penalty

    penalty *= factor
    return penalty

class StimulusPostprocessor(abc.ABC):
    """ Base class for stimulus clippers. """
    @abc.abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor:
        """ x.shape: batch x channels x time x n_rows x n_cols """
        pass

class NoOpStimulusPostprocessor(StimulusPostprocessor):
    def process(self, x: torch.Tensor) -> torch.Tensor:
        return x

class ChangeNormJointlyClipRangeSeparately(StimulusPostprocessor):
    """ First change the norm and afterward clip the value of x to some specified range """
    def __init__(
            self,
            norm: float = 30.0,
            min_max_values: Iterable[Tuple[Optional[float], Optional[float]]] = ((-0.654, 6.269), (-0.913, 6.269)),
    ):
        self._norm = norm
        self._min_max_values = list(min_max_values)

    def process(self, x):
        assert x.shape[1] == len(self._min_max_values), \
            f"Expected {len(self._min_max_values)} channels in dim 1, got {x.shape=}"

        # Re-normalize
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        renorm = x * (self._norm / x_norm).view(len(x), *[1] * (x.dim() - 1))

        # Clip
        clipped_array = []
        for i, (min_val, max_val) in enumerate(self._min_max_values):
            clipped = torch.clamp(renorm[:, i], min=min_val, max=max_val)
            clipped_array.append(clipped)
        result = torch.stack(clipped_array, dim=1)

        return result
