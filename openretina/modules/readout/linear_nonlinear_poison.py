from math import ceil, sqrt

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

    The temporal kernel starts as a delta on the current frame, so at initialization the model is the
    lag-0 `LNPReadout`.
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
        # One temporal kernel per neuron: a grouped convolution over the neurons' spatial projections
        self.temporal_kernel = nn.Conv3d(
            in_channels=self.n_neurons,
            out_channels=self.n_neurons,
            kernel_size=(temporal_kernel_size, 1, 1),
            groups=self.n_neurons,
            bias=bias,
            stride=1,
        )

        nn.init.xavier_normal_(self.spatial_kernel.weight.data)
        with torch.no_grad():
            self.temporal_kernel.weight.zero_()
            self.temporal_kernel.weight[:, :, -1] = 1.0

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)
        self._temporal_smooth_reg_fn = regularizers.TimeLaplaceL23dnorm(padding=laplace_padding)
        self.initialize(mean_activity)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        out = self.spatial_kernel(x)  # (batch, neurons, t, 1, 1)
        out = self.temporal_kernel(out)  # (batch, neurons, t - temporal_kernel_size + 1, 1, 1)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons")
        return out

    def spatiotemporal_filters(self) -> Float[torch.Tensor, "neurons channels time height width"]:
        """The rank-1 linear filter of each neuron, ``temporal[t] * spatial[c, h, w]``.

        The product is free of the scale and sign ambiguity between the two factors.
        """
        spatial = self.spatial_kernel.weight[:, :, 0]  # (neurons, channels, h, w)
        temporal = self.temporal_kernel.weight[:, 0, :, 0, 0]  # (neurons, t)
        return torch.einsum("nt,nchw->ncthw", temporal, spatial)

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
        temporal = self.temporal_kernel.weight.detach().cpu().numpy()[neuron_id, 0, :, 0, 0]
        weight_abs_max = float(np.abs(spatial).max()) or 1.0

        ax_readout.imshow(
            LNPReadout._build_channel_montage(spatial),
            interpolation="none",
            cmap="RdBu_r",
            norm=Normalize(-weight_abs_max, weight_abs_max),
        )
        # Plot time relative to the current frame, which is the last kernel entry
        ax_features.plot(np.arange(-len(temporal) + 1, 1), temporal, marker="o")
        ax_features.axhline(0.0, color="black", linewidth=0.8)

        if add_titles:
            ax_readout.set_title("Per-Channel Spatial Kernels")
            ax_features.set_title("Temporal Kernel (frames before current)")

    def number_of_neurons(self) -> int:
        return self.n_neurons

    def initialize(self, mean_activity: Float[torch.Tensor, " n_neurons"] | None = None) -> None:
        if self.temporal_kernel.bias is not None and mean_activity is not None:
            with torch.no_grad():
                self.temporal_kernel.bias.copy_(mean_activity)
