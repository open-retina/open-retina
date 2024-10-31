from typing import Tuple, Optional
from collections.abc import Iterable

import torch


class StimulusRegularizationLoss:
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        return 0.0  # type: ignore


class RangeRegularizationLoss(StimulusRegularizationLoss):
    def __init__(
        self,
        min_max_values: Iterable[tuple[float, float]],
        max_norm: Optional[float],
        factor: float = 1.0,
    ):
        self._min_max_values = list(min_max_values)
        self._max_norm = max_norm
        self._factor = factor

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        """Penalizes the stimulus if it is outside the range defined by min_max_values."""
        loss: torch.Tensor = 0.0  # type: ignore
        for i, (min_val, max_val) in enumerate(self._min_max_values):
            stimulus_i = stimulus[:, i]
            min_penalty = torch.sum(torch.relu(min_val - stimulus_i))
            max_penalty = torch.sum(torch.relu(stimulus_i - max_val))
            loss += min_penalty + max_penalty

        if self._max_norm is not None:
            # Add a loss such that the norm of the stimulus is lower than max_norm
            norm_penalty = torch.relu(torch.norm(stimulus) - self._max_norm)
            loss += norm_penalty

        loss *= self._factor
        return loss


class StimulusPostprocessor:
    """Base class for stimulus clippers."""

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """x.shape: batch x channels x time x n_rows x n_cols"""
        return x


class ChangeNormJointlyClipRangeSeparately(StimulusPostprocessor):
    """First change the norm and afterward clip the value of x to some specified range"""

    def __init__(
        self,
        min_max_values: Iterable[Tuple[Optional[float], Optional[float]]],
        norm: Optional[float],
    ):
        self._norm = norm
        self._min_max_values = list(min_max_values)

    def process(self, x):
        assert x.shape[1] == len(
            self._min_max_values
        ), f"Expected {len(self._min_max_values)} channels in dim 1, got {x.shape=}"

        if self._norm is not None:
            # Re-normalize
            x_norm = torch.linalg.vector_norm(x.view(len(x), -1), dim=-1)
            renorm = x * (self._norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        else:
            renorm = x

        # Clip
        clipped_array = []
        for i, (min_val, max_val) in enumerate(self._min_max_values):
            clipped = torch.clamp(renorm[:, i], min=min_val, max=max_val)
            clipped_array.append(clipped)
        result = torch.stack(clipped_array, dim=1)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._norm=}, {self._min_max_values=})"
