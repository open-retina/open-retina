from collections.abc import Iterable
from typing import Optional

from jaxtyping import Float
import torch
import torch.nn.functional as F


def _gaussian_1d_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize to ensure the sum is 1
    return kernel


class StimulusRegularizationLoss:
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        return 0.0  # type: ignore


class RangeRegularizationLoss(StimulusRegularizationLoss):
    def __init__(
        self,
        min_max_values: Iterable[tuple[float | None, float | None]],
        max_norm: float | None,
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
            if min_val is not None:
                loss += torch.sum(torch.relu(min_val - stimulus_i))
            if max_val is not None:
                loss += torch.sum(torch.relu(stimulus_i - max_val))

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
        min_max_values: Iterable[tuple[Optional[float], Optional[float]]],
        norm: float | None,
    ):
        self._norm = norm
        self._min_max_values = list(min_max_values)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == len(self._min_max_values), (
            f"Expected {len(self._min_max_values)} channels in dim 1, got {x.shape=}"
        )

        if self._norm is not None:
            # Re-normalize
            x_norm = torch.linalg.vector_norm(x.view(len(x), -1), dim=-1)
            renorm = x * (self._norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        else:
            renorm = x

        # Clip
        clipped_array = []
        for i, (min_val, max_val) in enumerate(self._min_max_values):
            clipped = renorm[:, i]
            if min_val is not None or max_val is not None:
                clipped = torch.clamp(clipped, min=min_val, max=max_val)
            clipped_array.append(clipped)
        result = torch.stack(clipped_array, dim=1)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._norm=}, {self._min_max_values=})"


class TemporalGaussianLowPassFilterProcessor(StimulusPostprocessor):
    """ Uses a 1d Gaussian filter to convolve the stimulus over the temporal dimension.
        This acts as a low pass filter. """

    def __init__(
            self,
            sigma: float,
            kernel_size: int,
            device: str = "cpu",
    ):
        kernel = _gaussian_1d_kernel(sigma, kernel_size)
        self._kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        self._kernel_size = kernel_size

    def process(self, x: Float[torch.Tensor, "batch_dim channels time height width"]) -> torch.Tensor:
        """
        Apply a Gaussian low-pass filter to the stimulus tensor along the temporal dimension.

        Arguments:
            x (Tensor): Tensor of shape (batch_dim, channels, time_dim, height, width)
        Returns:
            Tensor: The filtered stimulus tensor.
        """
        # Create the Gaussian kernel in the temporal dimension
        kernel = self._kernel.repeat(x.shape[1], 1, 1, 1, 1).to(x.device)

        # Apply convolution in the temporal dimension (axis 2)
        # We need to ensure that the kernel is convolved only along the time dimension.
        filtered_stimulus = F.conv3d(x, kernel, padding="same", groups=x.shape[1])

        return filtered_stimulus

