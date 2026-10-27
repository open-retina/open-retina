"""
Multi-neuron colour LN readout with one learnable Gaussian spatial filter per colour channel.

ColourGaussianLNReadout gives each neuron, for every input channel (e.g. green and UV), its own elliptical Gaussian
spatial filter and its own temporal filter. The channels' filtered signals are summed and passed through the
softplus nonlinearity of the Spatial Contrast readouts. It is the colour counterpart of the LN model, i.e. of
LearnableFilterSpatialContrastReadout with the contrast weight w fixed at 0, as the separable colour LNP is of the
greyscale one.

MultiColourGaussianLNReadout wraps multiple sessions, following the MultiReadoutBase interface.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int

from openretina.modules.layers.regularizers import TimeLaplaceL23dnorm
from openretina.modules.readout.base import Readout
from openretina.modules.readout.multi_readout import MultiReadoutBase
from openretina.modules.readout.spatial_contrast_model_readout import gaussian_2d_torch


class ColourGaussianLNReadout(Readout):
    """Single-session, multi-neuron LN readout with an independent Gaussian spatial filter per channel.

    For neuron n, the drive is ``sum_c (temporal[n, c] * p[n, c])(t)``, where ``p[n, c](t)`` is the mean of channel c
    weighted by the neuron's Gaussian for that channel, and ``*`` is a temporal convolution. The output is
    ``a * softplus(b * drive + c)``.

    The spatial filters are parametrised as in LearnableFilterSpatialContrastReadout: each channel's Gaussian has its
    own centre, widths (softplus above ``sigma_floor``) and orientation, and is masked at ``sigma_contour`` times its
    current widths. Each spatial filter is applied as a weighted mean, so it carries no gain. The temporal filters
    of a neuron are L2-normalised jointly over channels and time: their relative norms set the balance between the
    channels, and the overall gain sits in ``nl_b``.

    The forward pass projects the stimulus onto the spatial filters before filtering in time, which is exact for a
    linear filter and costs one pass over the pixels per neuron and channel.

    Args:
        in_shape: (channels, time, height, width) of the core output.
        outdims: Number of neurons in this session.
        temporal_filter_length: Number of frames of the temporal filters; the last one weighs the current frame.
        sigma_contour: Each spatial filter is masked to this many standard deviations of its current Gaussian.
        sigma_floor: Lower bound on the Gaussian widths, in pixels. Must keep the mask wider than one pixel.
        gaussian_init_sigma: Initial width of every Gaussian, in pixels. All Gaussians start at the frame centre.
        smooth_weight_temporal: Weight of the temporal smoothness penalty (``TimeLaplaceL23dnorm`` on the
            normalised temporal filters), relative to a Poisson loss summed over the batch, as in the LNP readouts.
        a_init, b_init, c_init: Initial values of the nonlinearity parameters.
        mean_activity: Optional mean activity of each neuron.
        init_offset_from_mean_activity: If True (and mean_activity is given), initialise each neuron's offset c so
            that a * softplus(c) equals its mean activity, instead of using c_init.

    The temporal filters start random, each tap drawn from U(-a, a) with ``a = sqrt(3 / temporal_filter_length)``
    from torch's global RNG, which Lightning seeds with the run's seed.
    """

    x_grid: torch.Tensor
    y_grid: torch.Tensor

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        temporal_filter_length: int,
        sigma_contour: float,
        sigma_floor: float = 1.0,
        gaussian_init_sigma: float = 1.5,
        smooth_weight_temporal: float = 0.0,
        a_init: float = 1.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        mean_activity: Float[torch.Tensor, " neurons"] | None = None,
        init_offset_from_mean_activity: bool = False,
    ):
        super().__init__()
        if sigma_floor * sigma_contour < 1.0:
            raise ValueError(
                f"{sigma_floor=} with {sigma_contour=} lets the mask shrink to a single pixel, where the spatial "
                "filter's parameters get no gradient. Raise sigma_floor to at least 1 / sigma_contour."
            )
        if temporal_filter_length > in_shape[1]:
            raise ValueError(f"{temporal_filter_length=} is longer than the {in_shape[1]} frames the core outputs.")

        self.in_shape = in_shape
        self.outdims = outdims
        self.n_channels = in_shape[0]
        self.temporal_filter_length = temporal_filter_length
        self.sigma_contour = sigma_contour
        self.sigma_floor = sigma_floor
        self.smooth_weight_temporal = smooth_weight_temporal

        height, width = in_shape[-2:]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
        )
        self.register_buffer("x_grid", x_grid, persistent=False)
        self.register_buffer("y_grid", y_grid, persistent=False)

        shape = (outdims, self.n_channels)
        self.center_x = nn.Parameter(torch.full(shape, (width - 1) / 2))
        self.center_y = nn.Parameter(torch.full(shape, (height - 1) / 2))
        # Widths are softplus(raw) + floor. Start a little above the floor, where softplus is not flat.
        sigma_excess = torch.clamp(torch.tensor(gaussian_init_sigma - sigma_floor), min=0.05)
        sigma_raw = sigma_excess + torch.log(-torch.expm1(-sigma_excess))  # inverse softplus
        self.sigma_x_raw = nn.Parameter(torch.full(shape, float(sigma_raw)))
        self.sigma_y_raw = nn.Parameter(torch.full(shape, float(sigma_raw)))
        self.theta = nn.Parameter(torch.zeros(shape))

        bound = (3.0 / temporal_filter_length) ** 0.5
        self.temporal_weight = nn.Parameter(
            torch.empty(outdims, self.n_channels, temporal_filter_length).uniform_(-bound, bound)
        )

        self.nl_a = nn.Parameter(torch.full((outdims,), a_init))
        self.nl_b = nn.Parameter(torch.full((outdims,), b_init))
        self.nl_c = nn.Parameter(torch.full((outdims,), c_init))
        if init_offset_from_mean_activity and mean_activity is not None:
            # As in SpatialContrastReadout: start each neuron at its own mean rate, a * softplus(c) = mean activity
            if a_init <= 0:
                raise ValueError(f"init_offset_from_mean_activity needs a positive a_init, got {a_init}.")
            target = torch.clamp(torch.as_tensor(mean_activity, dtype=torch.float32) / a_init, min=1e-3)
            self.nl_c.data.copy_(target + torch.log(-torch.expm1(-target)))  # inverse softplus

        # Bias for Readout base class compatibility (unused but required by interface)
        self.bias = nn.Parameter(torch.zeros(outdims), requires_grad=False)
        self._temporal_smooth_reg_fn = TimeLaplaceL23dnorm(persistent_buffer=False)

        self.initialize(mean_activity)

    @property
    def features(self) -> nn.Parameter:  # type: ignore[override]
        """Satisfies the Readout base class interface."""
        return self.temporal_weight

    def initialize(self, mean_activity: Float[torch.Tensor, " n_neurons"] | None = None) -> None:
        self.initialize_bias(mean_activity)

    def sigmas(self) -> tuple[Float[torch.Tensor, "neurons channels"], Float[torch.Tensor, "neurons channels"]]:
        return F.softplus(self.sigma_x_raw) + self.sigma_floor, F.softplus(self.sigma_y_raw) + self.sigma_floor

    def _filters(
        self,
    ) -> tuple[
        Float[torch.Tensor, "neurons channels height width"],
        Float[torch.Tensor, "neurons channels t_filter"],
        Float[torch.Tensor, "neurons channels"],
    ]:
        """The spatial filters (unit L2 norm per channel), the temporal filters (unit L2 norm per neuron) and the
        spatial filter sums the forward pass uses."""
        sigma_x, sigma_y = self.sigmas()
        gaussian, normalized_distance = gaussian_2d_torch(
            self.x_grid,
            self.y_grid,
            self.center_x.flatten(),
            self.center_y.flatten(),
            sigma_x.flatten(),
            sigma_y.flatten(),
            self.theta.flatten(),
        )
        # Hard mask as in LearnableFilterSpatialContrastReadout; gradients reach the parameters inside it
        mask = (normalized_distance <= self.sigma_contour).to(gaussian.dtype).detach()
        spatial = gaussian * mask
        spatial = spatial / spatial.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)[:, None, None]
        spatial = rearrange(spatial, "(n c) h w -> n c h w", n=self.outdims)

        temporal = F.normalize(self.temporal_weight.flatten(start_dim=1), dim=-1, eps=1e-8)
        temporal = temporal.view_as(self.temporal_weight)
        return spatial, temporal, spatial.sum(dim=(-2, -1))

    def regularizer(self, reduction: Literal["sum", "mean", None] = "sum") -> torch.Tensor:
        if self.smooth_weight_temporal == 0:
            return torch.tensor(0.0, device=self.nl_a.device)
        _, temporal, _ = self._filters()
        return self.smooth_weight_temporal * self._temporal_smooth_reg_fn(temporal[..., None, None])

    def filter_parameters(self) -> list[nn.Parameter]:
        return [self.center_x, self.center_y, self.sigma_x_raw, self.sigma_y_raw, self.theta, self.temporal_weight]

    def apply_constraints(self) -> None:
        """Called after every optimizer step: keep the output gain non-negative and the centres on the stimulus."""
        self.nl_a.data.clamp_(min=0.0)
        self.center_x.data.clamp_(min=0.0, max=float(self.x_grid.shape[-1] - 1))
        self.center_y.data.clamp_(min=0.0, max=float(self.y_grid.shape[-2] - 1))

    def number_of_neurons(self) -> int:
        return self.outdims

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes
        with torch.no_grad():
            spatial, temporal, _ = self._filters()
        # Channels side by side, separated by a blank column
        spatial_filters = spatial[neuron_id].cpu().numpy()
        gap = np.full((spatial_filters.shape[1], 1), np.nan)
        panels = [part for channel in spatial_filters for part in (channel, gap)][:-1]
        ax_readout.imshow(np.concatenate(panels, axis=1), interpolation="none", cmap="Greys", vmin=0)
        for channel, kernel in enumerate(temporal[neuron_id].cpu().numpy()):
            ax_features.plot(np.arange(-len(kernel) + 1, 1), kernel, label=f"channel {channel}")
        ax_features.axhline(0.0, color="black", linewidth=0.8)
        ax_features.legend(fontsize="small")
        if add_titles:
            ax_readout.set_title("Spatial filters per channel")
            ax_features.set_title("Temporal filters")

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time height width"],
        data_key: str | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch time_out neurons"]:
        spatial, temporal, spatial_sums = self._filters()
        # Weighted mean of every channel under each neuron's Gaussian for that channel
        projected = einsum(x, spatial, "b c t h w, n c h w -> b n c t") / (spatial_sums[None, :, :, None] + 1e-8)
        # One group per neuron: its C channels are filtered in time and summed
        drive = F.conv1d(rearrange(projected, "b n c t -> b (n c) t"), temporal, groups=self.outdims)
        output = self.nl_a[None, :, None] * F.softplus(self.nl_b[None, :, None] * drive + self.nl_c[None, :, None])
        return rearrange(output, "b n t -> b t n")


class MultiColourGaussianLNReadout(MultiReadoutBase):
    """Multi-session wrapper for ColourGaussianLNReadout.

    Args:
        in_shape: (channels, time, height, width) of the core output.
        n_neurons_dict: Mapping from session key to number of neurons.
        mean_activity_dict: Optional per-session mean activities.
        readout_reg_avg: Whether to average the regularizer across sessions.
        Remaining arguments: see ColourGaussianLNReadout.
    """

    _base_readout_cls = ColourGaussianLNReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        temporal_filter_length: int,
        sigma_contour: float,
        sigma_floor: float = 1.0,
        gaussian_init_sigma: float = 1.5,
        smooth_weight_temporal: float = 0.0,
        a_init: float = 1.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
        init_offset_from_mean_activity: bool = False,
        readout_reg_avg: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            mean_activity_dict=mean_activity_dict,
            readout_reg_avg=readout_reg_avg,
            temporal_filter_length=temporal_filter_length,
            sigma_contour=sigma_contour,
            sigma_floor=sigma_floor,
            gaussian_init_sigma=gaussian_init_sigma,
            smooth_weight_temporal=smooth_weight_temporal,
            a_init=a_init,
            b_init=b_init,
            c_init=c_init,
            init_offset_from_mean_activity=init_offset_from_mean_activity,
            **kwargs,
        )
