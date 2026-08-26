from collections.abc import Iterable
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float


def _gaussian_1d_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
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


class SmoothnessRegularizationLoss(StimulusRegularizationLoss):
    """Penalizes high spatial and/or temporal frequencies in the stimulus.

    A squared first-difference (total-variation-style) penalty on neighbouring elements, divided by
    the stimulus' mean square. The normalization matters: `ChangeNormJointlyClipRangeSeparately`
    pins the norm, so an unnormalized penalty would scale with the contrast target and `factor`
    would mean something different at every `rms_factor` rung.

    Normalized this way, each term equals `2 * (1 - r)` with `r` the lag-1 autocorrelation along
    that axis, so the knob acts directly on the quantity you measure on the result: pushing the
    spatial term down is the same as pushing the MEI's lag-1 spatial autocorrelation up.

    Note on scale: `optimize_stimulus` minimizes `-objective + sum(regularizers)`, and each term
    here is dimensionless and O(1). The factors therefore have to be comparable to the *objective's*
    magnitude to bite at all -- for a model whose responses are O(1e3), a factor of O(1e2..1e3) is
    the interesting range, not O(0.1).

    Both weights default to 0.0, so the loss is inert until one is set and adding it to an existing
    call site changes nothing until you ask it to.

    Args:
        factor_spatial: weight on the height/width first differences.
        factor_temporal: weight on the time first differences.
        eps: guard for the division by the mean square.
    """

    def __init__(
        self,
        factor_spatial: float = 0.0,
        factor_temporal: float = 0.0,
        eps: float = 1e-8,
    ):
        self._factor_spatial = factor_spatial
        self._factor_temporal = factor_temporal
        self._eps = eps

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=stimulus.device, dtype=stimulus.dtype)
        if self._factor_spatial == 0.0 and self._factor_temporal == 0.0:
            return loss

        mean_square = stimulus.pow(2).mean() + self._eps
        if self._factor_spatial != 0.0:
            spatial = stimulus.diff(dim=-2).pow(2).mean() + stimulus.diff(dim=-1).pow(2).mean()
            loss = loss + self._factor_spatial * spatial / mean_square
        if self._factor_temporal != 0.0:
            temporal = stimulus.diff(dim=-3).pow(2).mean()
            loss = loss + self._factor_temporal * temporal / mean_square
        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(factor_spatial={self._factor_spatial}, factor_temporal={self._factor_temporal})"
        )


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


class ZeroOutsideMaskProcessor(StimulusPostprocessor):
    """Holds the stimulus at zero outside a spatial mask, after every step.

    Masking the *initial* stimulus is not enough once any regularizer is active. The objective's
    gradient is confined to what the model can see -- for an unpadded core with a point readout, a
    ~20x20 px footprint of the frame -- but a smoothness or range loss is computed over the whole
    tensor, so its gradient reaches the surround too and pushes it back off zero. Measured on
    `qiu_2026`, a spatial smoothness weight of 1e3 leaked 5% of the squared norm outside the support
    and 1e4 leaked 9%.

    Chain this *before* the norm postprocessor, so the norm is renormalized over the masked tensor
    and therefore actually hits its target on the region that matters:

        stimulus_postprocessor=[ZeroOutsideMaskProcessor(mask), ChangeNormJointlyClipRangeSeparately(...)]

    Args:
        mask: boolean or 0/1 tensor broadcastable against the stimulus' trailing dimensions -- an
            ``(height, width)`` mask is the usual case.
    """

    def __init__(self, mask: torch.Tensor):
        self._mask = mask

    def process(self, x: Float[torch.Tensor, "batch_dim channels time height width"]) -> torch.Tensor:
        return x * self._mask.to(device=x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mask_shape={tuple(self._mask.shape)}, kept={int(self._mask.sum())})"


class TemporalGaussianLowPassFilterProcessor(StimulusPostprocessor):
    """Uses a 1d Gaussian filter to convolve the stimulus over the temporal dimension.
    This acts as a low pass filter."""

    def __init__(
        self,
        sigma: float,
        kernel_size: int,
        device: str = "cpu",
    ):
        kernel = _gaussian_1d_kernel(sigma, kernel_size)
        self._kernel = kernel.to(device)

    def process(self, x: Float[torch.Tensor, "batch_dim channels time height width"]) -> torch.Tensor:
        """
        Apply a Gaussian low-pass filter to the stimulus tensor along the temporal dimension.

        Arguments:
            x (Tensor): Tensor of shape (batch_dim, channels, time_dim, height, width)
        Returns:
            Tensor: The filtered stimulus tensor.
        """
        # Create the Gaussian kernel in the temporal dimension
        kernel = einops.repeat(self._kernel.to(x.device), "s -> c 1 s 1 1", c=x.shape[1])

        # Apply convolution in the temporal dimension (axis 2)
        # We need to ensure that the kernel is convolved only along the time dimension.
        filtered_stimulus = F.conv3d(x, kernel, padding="same", groups=x.shape[1])

        return filtered_stimulus
