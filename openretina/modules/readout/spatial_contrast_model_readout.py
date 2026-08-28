"""
Multi-neuron Spatial Contrast readout following the core-readout pattern.

SpatialContrastReadout handles N neurons in a single session with frozen
spatial/temporal filters and per-neuron learnable parameters (w, a, b, c).

MultiSpatialContrastReadout wraps multiple sessions, each with its own
SpatialContrastReadout, following the MultiReadoutBase interface.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int

from openretina.modules.readout.base import Readout
from openretina.modules.readout.multi_readout import MultiReadoutBase


class SpatialContrastReadout(Readout):
    """Single-session, multi-neuron Spatial Contrast readout.

    Each neuron has frozen spatial and temporal filters (from pre-computed STAs)
    and 4 learnable scalar parameters: w (contrast weight) and a, b, c
    (nonlinearity shape).

    The forward pass processes neurons in chunks to limit GPU memory:
    for each chunk of K neurons, temporal filtering (conv1d) and spatial
    contraction (einsum) are done together, keeping the peak 5D tensor
    at [batch, K, t_out, H, W] instead of [batch, N, t_out, H, W].

    Args:
        in_shape: (channels, time, height, width) of core output.
        outdims: Number of neurons in this session.
        spatial_filters: Frozen spatial filters, shape [N, H, W].
        temporal_filters: Frozen temporal filters, shape [N, T_filter].
        w_init: Initial contrast weight.
        a_init: Initial output scaling (nl_a).
        b_init: Initial input gain (nl_b).
        c_init: Initial offset (nl_c).
        mean_activity: Optional mean activity for bias initialization.
        neuron_chunk_size: Max neurons processed at once in forward pass.
            Controls the memory/speed trade-off. Default 32 works on 8 GB GPUs.
    """

    # Registered buffers (declared for type checking)
    spatial_filters: torch.Tensor
    temporal_filters: torch.Tensor
    sf_sums: torch.Tensor

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        spatial_filters: Float[torch.Tensor, "neurons height width"],
        temporal_filters: Float[torch.Tensor, "neurons t_filter"],
        w_init: float = 0.0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        mean_activity: Float[torch.Tensor, " neurons"] | None = None,
        neuron_chunk_size: int = 32,
    ):
        super().__init__()
        assert spatial_filters.shape[0] == outdims
        assert temporal_filters.shape[0] == outdims

        self.outdims = outdims
        self.in_shape = in_shape
        self.neuron_chunk_size = neuron_chunk_size

        # Frozen filters as buffers
        self.register_buffer("spatial_filters", spatial_filters)
        self.register_buffer("temporal_filters", temporal_filters)
        self.register_buffer("sf_sums", spatial_filters.sum(dim=(-2, -1)))

        # Per-neuron learnable parameters
        self.w = nn.Parameter(torch.full((outdims,), w_init))
        self.nl_a = nn.Parameter(torch.full((outdims,), a_init))
        self.nl_b = nn.Parameter(torch.full((outdims,), b_init))
        self.nl_c = nn.Parameter(torch.full((outdims,), c_init))

        # Bias for Readout base class compatibility (unused but required by interface)
        self.bias = nn.Parameter(torch.zeros(outdims), requires_grad=False)

        self.initialize(mean_activity)

    @property
    def features(self) -> nn.Parameter:  # type: ignore[override]
        """Satisfies the Readout base class interface."""
        return self.w

    def initialize(self, mean_activity: Float[torch.Tensor, " n_neurons"] | None = None) -> None:
        self.initialize_bias(mean_activity)

    def regularizer(self, reduction: Literal["sum", "mean", None] = "sum") -> torch.Tensor:
        """No regularization needed for 4 params per neuron."""
        return torch.tensor(0.0, device=self.w.device)

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        spatial_filter = self.spatial_filters[neuron_id].detach().cpu().numpy()
        spatial_abs_max = np.abs(spatial_filter).max()
        if spatial_abs_max == 0:
            spatial_abs_max = 1.0

        ax_readout.imshow(
            spatial_filter,
            interpolation="none",
            cmap="RdBu_r",
            vmin=-spatial_abs_max,
            vmax=spatial_abs_max,
        )

        parameters = torch.stack((self.w[neuron_id], self.nl_a[neuron_id], self.nl_b[neuron_id], self.nl_c[neuron_id]))
        parameter_values = parameters.detach().cpu().numpy()
        parameter_names = ["w", "a", "b", "c"]

        ax_features.bar(parameter_names, parameter_values)
        parameter_abs_max = np.abs(parameter_values).max()
        if parameter_abs_max == 0:
            parameter_abs_max = 1.0
        ax_features.set_ylim(-parameter_abs_max * 1.1, parameter_abs_max * 1.1)
        ax_features.axhline(0.0, color="black", linewidth=0.8)

        if add_titles:
            ax_readout.set_title("Spatial Filter")
            ax_features.set_title("Neuron Parameters")

    def number_of_neurons(self) -> int:
        return self.outdims

    def _forward_chunk(
        self,
        x_flat: Float[torch.Tensor, "bhw 1 time"],
        chunk_start: int,
        chunk_end: int,
        batch: int,
        h: int,
        w: int,
    ) -> Float[torch.Tensor, "batch chunk time_out"]:
        """Process a chunk of neurons: temporal filtering + spatial contraction + nonlinearity."""
        # Temporal filtering: standard conv1d with K output channels, 1 input channel
        kernels = self.temporal_filters[chunk_start:chunk_end, None, :]  # [K, 1, t_filter]
        tf_chunk = F.conv1d(x_flat, kernels)  # [(b*h*w), K, t_out]
        tf_chunk = rearrange(tf_chunk, "(b h w) k t -> b k t h w", b=batch, h=h, w=w)

        # Spatially-weighted mean
        sf = self.spatial_filters[chunk_start:chunk_end]  # [K, H, W]
        sf_sums = self.sf_sums[chunk_start:chunk_end] + 1e-8  # [K]
        imean = einsum(tf_chunk, sf, "b k t h w, k h w -> b k t") / sf_sums[None, :, None]

        # Local spatial contrast via variance decomposition: Var = E[x^2] - E[x]^2
        mean_sq = einsum(tf_chunk**2, sf, "b k t h w, k h w -> b k t") / sf_sums[None, :, None]
        variance = mean_sq - imean**2
        lsc = torch.sqrt(torch.clamp(variance, min=1e-6))

        # Per-neuron nonlinearity
        w_chunk = self.w[chunk_start:chunk_end][None, :, None]
        combined = imean + w_chunk * lsc

        nl_a = self.nl_a[chunk_start:chunk_end][None, :, None]
        nl_b = self.nl_b[chunk_start:chunk_end][None, :, None]
        nl_c = self.nl_c[chunk_start:chunk_end][None, :, None]
        return nl_a * F.softplus(nl_b * combined + nl_c)  # [b, K, t_out]

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time height width"],
        data_key: str | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch time_out neurons"]:
        batch, channels, time, h, w = x.shape
        n = self.outdims

        # Flatten spatial dims for conv1d: each pixel is an independent "batch" element
        x_flat = rearrange(x, "b 1 t h w -> (b h w) 1 t")

        # Process neurons in chunks to limit peak memory
        # Peak 5D tensor per chunk: [batch, chunk_size, t_out, H, W]
        chunk_outputs: list[torch.Tensor] = []
        for chunk_start in range(0, n, self.neuron_chunk_size):
            chunk_end = min(chunk_start + self.neuron_chunk_size, n)
            chunk_out = self._forward_chunk(x_flat, chunk_start, chunk_end, batch, h, w)
            chunk_outputs.append(chunk_out)

        output = torch.cat(chunk_outputs, dim=1)  # [b, N, t_out]
        return rearrange(output, "b n t -> b t n")


class MultiSpatialContrastReadout(MultiReadoutBase):
    """Multi-session wrapper for SpatialContrastReadout.

    Unlike other MultiReadoutBase subclasses, this one overrides __init__
    because each session needs its own pre-computed filter tensors passed
    to SpatialContrastReadout, rather than sharing the same kwargs.

    Args:
        in_shape: (channels, time, height, width) of core output.
        n_neurons_dict: Mapping from session key to number of neurons.
        spatial_filters_dict: Per-session spatial filter tensors [N, H, W].
        temporal_filters_dict: Per-session temporal filter tensors [N, T_filter].
        w_init: Initial contrast weight.
        a_init: Initial output scaling.
        b_init: Initial input gain.
        c_init: Initial offset.
        mean_activity_dict: Optional per-session mean activities.
        readout_reg_avg: Whether to average regularizer across sessions.
    """

    _base_readout_cls = SpatialContrastReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        spatial_filters_dict: dict[str, Float[torch.Tensor, "neurons height width"]],
        temporal_filters_dict: dict[str, Float[torch.Tensor, "neurons t_filter"]],
        w_init: float = 0.0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
        readout_reg_avg: bool = False,
        neuron_chunk_size: int = 32,
    ):
        # Bypass MultiReadoutBase.__init__ since we need per-session filter tensors
        nn.ModuleDict.__init__(self)

        self._base_readout_cls = SpatialContrastReadout
        self._in_shape = in_shape
        self._readout_kwargs = {
            "w_init": w_init,
            "a_init": a_init,
            "b_init": b_init,
            "c_init": c_init,
            "neuron_chunk_size": neuron_chunk_size,
        }
        self.readout_reg_avg = readout_reg_avg
        self.readout_reg_reduction: Literal["mean", "sum"] = "mean" if readout_reg_avg else "sum"

        for data_key in n_neurons_dict:
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None
            self.add_module(
                data_key,
                SpatialContrastReadout(
                    in_shape=in_shape,
                    outdims=n_neurons_dict[data_key],
                    spatial_filters=spatial_filters_dict[data_key],
                    temporal_filters=temporal_filters_dict[data_key],
                    w_init=w_init,
                    a_init=a_init,
                    b_init=b_init,
                    c_init=c_init,
                    mean_activity=mean_activity,
                    neuron_chunk_size=neuron_chunk_size,
                ),
            )
