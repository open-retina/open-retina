from math import ceil, sqrt
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from matplotlib.colors import Normalize

from openretina.modules.layers import regularizers
from openretina.modules.readout.base import Readout


class LNPReadout(Readout):
    """Linear Nonlinear Poisson Readout (LNP)
    For use as an LNP Model use this readout with a DummyCore that passes the input through.
    """

    # Linear nonlinear Poisson
    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        mean_activity: Float[torch.Tensor, " outdims"] | None = None,
        smooth_weight: float = 0.0,
        sparse_weight: float = 0.0,
        smooth_regularizer: str = "LaplaceL2norm",
        laplace_padding=None,
        nonlinearity: str = "exp",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight
        self.kernel_size = tuple(in_shape[2:])
        self.in_channels = in_shape[0]
        self.n_neurons = outdims
        self.nonlinearity = torch.exp if nonlinearity == "exp" else F.__dict__[nonlinearity]

        self.inner_product_kernel = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_neurons,  # Each neuron gets its own kernel
            kernel_size=(1, *self.kernel_size),  # Not using time
            bias=bias,
            stride=1,
        )

        nn.init.xavier_normal_(self.inner_product_kernel.weight.data)

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )

        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)
        self.initialize(mean_activity)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        out = self.inner_product_kernel(x)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons")
        return out

    def weights_l1(self, average: bool = True):
        """Returns l1 regularization across all weight dimensions

        Args:
            average (bool, optional): use mean of weights instead of sum. Defaults to True.
        """
        if average:
            return self.inner_product_kernel.weight.abs().mean()
        else:
            return self.inner_product_kernel.weight.abs().sum()

    def laplace(self):
        # Squeezing out the empty time dimension, so we can use 2D regularizers
        return self._smooth_reg_fn(self.inner_product_kernel.weight.squeeze(2))

    def regularizer(self, **kwargs):
        return self.smooth_weight * self.laplace() + self.sparse_weight * self.weights_l1()

    @staticmethod
    def _build_channel_montage(channel_weights: np.ndarray, padding: int = 1) -> np.ndarray:
        n_channels, height, width = channel_weights.shape
        n_cols = max(1, ceil(sqrt(n_channels)))
        n_rows = ceil(n_channels / n_cols)
        montage_height = n_rows * height + padding * (n_rows - 1)
        montage_width = n_cols * width + padding * (n_cols - 1)
        montage = np.zeros((montage_height, montage_width), dtype=channel_weights.dtype)

        for idx, kernel in enumerate(channel_weights):
            row, col = divmod(idx, n_cols)
            row_start = row * (height + padding)
            col_start = col * (width + padding)
            montage[row_start : row_start + height, col_start : col_start + width] = kernel

        return montage

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        weights = self.inner_product_kernel.weight.detach().cpu().numpy()[neuron_id, :, 0, :, :]
        weight_abs_max = float(np.abs(weights).max()) or 1.0
        montage = self._build_channel_montage(weights)
        channel_norms = np.linalg.norm(weights.reshape(weights.shape[0], -1), axis=1)

        ax_readout.imshow(
            montage,
            interpolation="none",
            cmap="RdBu_r",
            norm=Normalize(-weight_abs_max, weight_abs_max),
        )
        ax_features.bar(range(channel_norms.shape[0]), channel_norms)

        if add_titles:
            ax_readout.set_title("Per-Channel Spatial Kernels")
            ax_features.set_title("Kernel Norm per Channel")

    def number_of_neurons(self) -> int:
        return self.n_neurons

    def spatiotemporal_filters(self) -> Float[torch.Tensor, "neurons channels time height width"]:
        """The linear filter of each neuron. Without a temporal kernel, time has length 1 (the current frame)."""
        return self.inner_product_kernel.weight

    def initialize(self, mean_activity: Float[torch.Tensor, " n_neurons"] | None = None) -> None:
        if self.inner_product_kernel.bias is not None and mean_activity is not None:
            with torch.no_grad():
                self.inner_product_kernel.bias.copy_(mean_activity)


class SeparableLNPReadout(Readout):
    """Space-time separable (rank-1) Linear Nonlinear Poisson readout.

    Each neuron's linear filter is the product of one spatial map and one temporal kernel:
    a full-field spatial projection of every frame, followed by a per-neuron temporal convolution over the
    last ``temporal_kernel_size`` frames. Like `LNPReadout`, use it with a DummyCore that passes the input through.

    The temporal convolution is valid in time and, like every torch convolution, a cross-correlation: kernel
    index -1 weighs the current frame and index 0 the furthest past. That is the same frame order the Spatial
    Contrast readout expects from its temporal filters, so `spatiotemporal_filters()` can be saved as an STA as-is.
    The output is ``temporal_kernel_size - 1`` frames shorter than the input.

    By default (``temporal_init="current_frame"``) the temporal kernel starts as a delta on the current frame, so at
    initialization the model is the lag-0 `LNPReadout`. Natural movies change little from one frame to the next, so
    they constrain the sum of neighbouring kernel weights much better than their difference: part of the delta
    survives training as a large weight on the current frame with an opposite-sign weight on the frame before, which
    does not change the predictions. ``temporal_init="uniform"`` starts every frame at ``1 / temporal_kernel_size``
    (the same total gain) and leaves no such trace, but on hoefling_2024 it trained to a lower test correlation.
    ``temporal_init="random"`` draws every tap from U(-a, a) with ``a = sqrt(3 / temporal_kernel_size)``, so the
    kernel starts with the delta's expected squared norm of 1 but without a preferred frame.

    With ``temporal_kernel_per_channel=True`` each input channel (e.g. green and UV) gets its own temporal kernel:
    the filter is ``sum_c temporal[c, t] * spatial[c, h, w]``, rank 1 per channel rather than overall. The bias then
    sits in ``output_bias`` instead of the temporal convolution, which carries one kernel per neuron and channel.
    With a single channel both modes compute the same filter.
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        temporal_kernel_size: int = 10,
        mean_activity: Float[torch.Tensor, " outdims"] | None = None,
        smooth_weight: float = 0.0,
        smooth_weight_temporal: float = 0.0,
        sparse_weight: float = 0.0,
        smooth_regularizer: str = "LaplaceL2norm",
        laplace_padding=None,
        nonlinearity: str = "exp",
        bias: bool = False,
        temporal_init: Literal["current_frame", "uniform", "random"] = "current_frame",
        temporal_kernel_per_channel: bool = False,
        **kwargs,
    ):
        super().__init__()
        if temporal_kernel_size > in_shape[1]:
            raise ValueError(f"{temporal_kernel_size=} is longer than the {in_shape[1]} frames the core outputs.")
        self.smooth_weight = smooth_weight
        self.smooth_weight_temporal = smooth_weight_temporal
        self.sparse_weight = sparse_weight
        self.kernel_size = tuple(in_shape[2:])
        self.temporal_kernel_size = temporal_kernel_size
        self.in_channels = in_shape[0]
        self.n_neurons = outdims
        self.nonlinearity = torch.exp if nonlinearity == "exp" else F.__dict__[nonlinearity]

        self.spatial_kernel = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_neurons,
            kernel_size=(1, *self.kernel_size),
            bias=False,
            stride=1,
        )
        self.temporal_kernel_per_channel = temporal_kernel_per_channel
        # One temporal kernel per neuron (and channel): a grouped convolution over the spatial projections
        n_temporal_kernels = self.n_neurons * (self.in_channels if temporal_kernel_per_channel else 1)
        self.temporal_kernel = nn.Conv3d(
            in_channels=n_temporal_kernels,
            out_channels=n_temporal_kernels,
            kernel_size=(temporal_kernel_size, 1, 1),
            groups=n_temporal_kernels,
            bias=bias and not temporal_kernel_per_channel,
            stride=1,
        )
        # Per channel, the bias is added once after summing the channels
        self.output_bias = nn.Parameter(torch.zeros(self.n_neurons)) if bias and temporal_kernel_per_channel else None

        nn.init.xavier_normal_(self.spatial_kernel.weight.data)
        with torch.no_grad():
            if temporal_init == "current_frame":
                self.temporal_kernel.weight.zero_()
                self.temporal_kernel.weight[:, :, -1] = 1.0
            elif temporal_init == "uniform":
                self.temporal_kernel.weight.fill_(1.0 / temporal_kernel_size)
            elif temporal_init == "random":
                bound = (3.0 / temporal_kernel_size) ** 0.5
                self.temporal_kernel.weight.uniform_(-bound, bound)
            else:
                raise ValueError(f"Unknown {temporal_init=}, expected 'current_frame', 'uniform' or 'random'.")

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)
        self._temporal_smooth_reg_fn = regularizers.TimeLaplaceL23dnorm(padding=laplace_padding)
        self.initialize(mean_activity)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        if self.temporal_kernel_per_channel:
            # Project each channel on its own spatial map, so each channel can be filtered in time separately
            out = torch.einsum("bcthw,nchw->bnct", x, self.spatial_kernel.weight[:, :, 0])
            out = rearrange(out, "batch neurons channels t -> batch (neurons channels) t 1 1")
            out = self.temporal_kernel(out)
            out = rearrange(
                out, "batch (neurons channels) t 1 1 -> batch neurons channels t", channels=self.in_channels
            )
            out = out.sum(dim=2)
            if self.output_bias is not None:
                out = out + self.output_bias[None, :, None]
            out = rearrange(out, "batch neurons t -> batch neurons t 1 1")
        else:
            out = self.spatial_kernel(x)  # (batch, neurons, t, 1, 1)
            out = self.temporal_kernel(out)  # (batch, neurons, t - temporal_kernel_size + 1, 1, 1)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons")
        return out

    def temporal_kernels(self) -> Float[torch.Tensor, "neurons kernels time"]:
        """The temporal kernels, one per neuron, or one per neuron and channel with ``temporal_kernel_per_channel``."""
        return self.temporal_kernel.weight[:, 0, :, 0, 0].reshape(self.n_neurons, -1, self.temporal_kernel_size)

    def spatiotemporal_filters(self) -> Float[torch.Tensor, "neurons channels time height width"]:
        """The separable linear filter of each neuron, ``temporal[c, t] * spatial[c, h, w]``.

        Without ``temporal_kernel_per_channel`` all channels share one temporal kernel, so the filter is rank 1.
        The product is free of the scale and sign ambiguity between the two factors.
        """
        spatial = self.spatial_kernel.weight[:, :, 0]  # (neurons, channels, h, w)
        temporal = self.temporal_kernels().expand(-1, self.in_channels, -1)  # (neurons, channels, t)
        return torch.einsum("nct,nchw->ncthw", temporal, spatial)

    def weights_l1(self, average: bool = True):
        """L1 of the full spatiotemporal filter.

        Taken on the product rather than the spatial factor alone: the latter could be shrunk at no cost by
        moving its scale into the temporal kernel.
        """
        filters = self.spatiotemporal_filters()
        return filters.abs().mean() if average else filters.abs().sum()

    def laplace(self):
        return self._smooth_reg_fn(self.spatial_kernel.weight.squeeze(2))

    def temporal_laplace(self):
        return self._temporal_smooth_reg_fn(self.temporal_kernel.weight)

    def regularizer(self, **kwargs):
        return (
            self.smooth_weight * self.laplace()
            + self.smooth_weight_temporal * self.temporal_laplace()
            + self.sparse_weight * self.weights_l1()
        )

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        spatial = self.spatial_kernel.weight.detach().cpu().numpy()[neuron_id, :, 0, :, :]
        temporal = self.temporal_kernels().detach().cpu().numpy()[neuron_id]  # (kernels, t)
        weight_abs_max = float(np.abs(spatial).max()) or 1.0

        ax_readout.imshow(
            LNPReadout._build_channel_montage(spatial),
            interpolation="none",
            cmap="RdBu_r",
            norm=Normalize(-weight_abs_max, weight_abs_max),
        )
        # Plot time relative to the current frame, which is the last kernel entry
        for channel, kernel in enumerate(temporal):
            label = f"channel {channel}" if len(temporal) > 1 else None
            ax_features.plot(np.arange(-len(kernel) + 1, 1), kernel, marker="o", label=label)
        if len(temporal) > 1:
            ax_features.legend()
        ax_features.axhline(0.0, color="black", linewidth=0.8)

        if add_titles:
            ax_readout.set_title("Per-Channel Spatial Kernels")
            ax_features.set_title("Temporal Kernel (frames before current)")

    def number_of_neurons(self) -> int:
        return self.n_neurons

    def initialize(self, mean_activity: Float[torch.Tensor, " n_neurons"] | None = None) -> None:
        bias = self.output_bias if self.output_bias is not None else self.temporal_kernel.bias
        if bias is not None and mean_activity is not None:
            with torch.no_grad():
                bias.copy_(mean_activity)
