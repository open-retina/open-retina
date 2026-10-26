"""
Multi-neuron Spatial Contrast readout following the core-readout pattern.

SpatialContrastReadout handles N neurons in a single session with frozen
spatial/temporal filters and per-neuron learnable parameters (w, a, b, c).

LearnableFilterSpatialContrastReadout additionally learns the filters: the spatial
filter stays an elliptical Gaussian whose centre, widths and orientation are trained,
and the temporal filter is trained freely.

MultiSpatialContrastReadout and MultiLearnableFilterSpatialContrastReadout wrap
multiple sessions, each with its own readout, following the MultiReadoutBase interface.
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
        init_offset_from_mean_activity: If True (and mean_activity is given), initialise each neuron's
            offset c so that a * softplus(c) equals its mean activity, instead of using c_init.
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
        init_offset_from_mean_activity: bool = False,
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
        if init_offset_from_mean_activity and mean_activity is not None:
            # Start each neuron at its own mean rate, a * softplus(c) = mean activity, rather than at a shared
            # offset: per-neuron parameters receive few updates per epoch, so a shared start is slow to calibrate.
            if a_init <= 0:
                raise ValueError(f"init_offset_from_mean_activity needs a positive a_init, got {a_init}.")
            target = torch.clamp(torch.as_tensor(mean_activity, dtype=torch.float32) / a_init, min=1e-3)
            self.nl_c.data.copy_(target + torch.log(-torch.expm1(-target)))  # inverse softplus

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

    def _filters(
        self,
    ) -> tuple[
        Float[torch.Tensor, "neurons height width"],
        Float[torch.Tensor, "neurons t_filter"],
        Float[torch.Tensor, " neurons"],
    ]:
        """The spatial filters, temporal filters and spatial filter sums the forward pass uses."""
        return self.spatial_filters, self.temporal_filters, self.sf_sums

    def filter_parameters(self) -> list[nn.Parameter]:
        """Trainable filter parameters. The filters are frozen here; see LearnableFilterSpatialContrastReadout."""
        return []

    def apply_constraints(self) -> None:
        """Called after every optimizer step: keep the output gain non-negative to avoid negative predictions."""
        self.nl_a.data.clamp_(min=0.0)

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        with torch.no_grad():
            spatial_filter = self._filters()[0][neuron_id].cpu().numpy()
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
        spatial_filters: Float[torch.Tensor, "neurons height width"],
        temporal_filters: Float[torch.Tensor, "neurons t_filter"],
        spatial_filter_sums: Float[torch.Tensor, " neurons"],
        chunk_start: int,
        chunk_end: int,
        batch: int,
        h: int,
        w: int,
    ) -> Float[torch.Tensor, "batch chunk time_out"]:
        """Process a chunk of neurons: temporal filtering + spatial contraction + nonlinearity."""
        # Temporal filtering: standard conv1d with K output channels, 1 input channel
        kernels = temporal_filters[chunk_start:chunk_end, None, :]  # [K, 1, t_filter]
        tf_chunk = F.conv1d(x_flat, kernels)  # [(b*h*w), K, t_out]
        tf_chunk = rearrange(tf_chunk, "(b h w) k t -> b k t h w", b=batch, h=h, w=w)

        # Spatially-weighted mean
        sf = spatial_filters[chunk_start:chunk_end]  # [K, H, W]
        sf_sums = spatial_filter_sums[chunk_start:chunk_end] + 1e-8  # [K]
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

        # Computed once per forward pass; for learnable filters this is where they are built from their parameters
        spatial_filters, temporal_filters, spatial_filter_sums = self._filters()

        # Process neurons in chunks to limit peak memory
        # Peak 5D tensor per chunk: [batch, chunk_size, t_out, H, W]
        chunk_outputs: list[torch.Tensor] = []
        for chunk_start in range(0, n, self.neuron_chunk_size):
            chunk_end = min(chunk_start + self.neuron_chunk_size, n)
            chunk_out = self._forward_chunk(
                x_flat, spatial_filters, temporal_filters, spatial_filter_sums, chunk_start, chunk_end, batch, h, w
            )
            chunk_outputs.append(chunk_out)

        output = torch.cat(chunk_outputs, dim=1)  # [b, N, t_out]
        return rearrange(output, "b n t -> b t n")


def gaussian_2d_torch(
    x_grid: Float[torch.Tensor, "height width"],
    y_grid: Float[torch.Tensor, "height width"],
    center_x: Float[torch.Tensor, " neurons"],
    center_y: Float[torch.Tensor, " neurons"],
    sigma_x: Float[torch.Tensor, " neurons"],
    sigma_y: Float[torch.Tensor, " neurons"],
    theta: Float[torch.Tensor, " neurons"],
) -> tuple[Float[torch.Tensor, "neurons height width"], Float[torch.Tensor, "neurons height width"]]:
    """Batched, differentiable unit-amplitude elliptical Gaussians.

    Torch counterpart of `openretina.utils.sta_processing.gaussian_2d` (same rotation convention), so that
    Gaussians fitted to STAs can be trained further. Also returns each pixel's distance from the centre in units
    of sigma, the quantity `create_gaussian_mask` thresholds.
    """
    x_diff = x_grid[None] - center_x[:, None, None]
    y_diff = y_grid[None] - center_y[:, None, None]
    cos_theta = torch.cos(theta)[:, None, None]
    sin_theta = torch.sin(theta)[:, None, None]
    # Coordinates along the Gaussian's principal axes
    x_rot = cos_theta * x_diff + sin_theta * y_diff
    y_rot = -sin_theta * x_diff + cos_theta * y_diff
    normalized_distance_sq = (x_rot / sigma_x[:, None, None]) ** 2 + (y_rot / sigma_y[:, None, None]) ** 2
    return torch.exp(-0.5 * normalized_distance_sq), torch.sqrt(normalized_distance_sq)


class LearnableFilterSpatialContrastReadout(SpatialContrastReadout):
    """Spatial Contrast readout whose spatial and temporal filters are learned.

    The spatial filter of each neuron stays an elliptical Gaussian, masked at ``sigma_contour`` standard
    deviations and L2-normalized exactly like the frozen filters from `extract_filters_from_sta`; its centre,
    widths and orientation are trained. The temporal filter is trained freely and normalized to unit L2 norm in
    every forward pass, since its scale is redundant with the nonlinearity's input gain ``nl_b``.

    The filters are initialised from the frozen filters' fits, so at initialization the readout reproduces
    `SpatialContrastReadout` (up to float precision, and for widths above ``sigma_floor``).

    Args:
        gaussian_params: Initial Gaussian of each neuron, columns (center_x, center_y, sigma_x, sigma_y, theta)
            in pixels and radians.
        sigma_contour: The spatial filter is masked to this many standard deviations.
        sigma_floor: Lower bound on both widths, in pixels. Must keep the mask wider than one pixel: once it covers
            only the centre pixel, the filter's parameters get no gradient and can never recover.
        smooth_weight_temporal: Weight of the temporal smoothness penalty, the normalised Laplacian of the LNP
            readouts (``TimeLaplaceL23dnorm``) taken on the unit-norm temporal filters, so that every neuron counts
            the same whatever the norm of its stored weights. As in the LNP readouts, the weight is relative to a
            Poisson loss summed over the batch.
        Remaining arguments: see SpatialContrastReadout. The frozen ``spatial_filters`` and ``temporal_filters``
            are kept as buffers, as a record of the initialization.
    """

    x_grid: torch.Tensor
    y_grid: torch.Tensor
    active: torch.Tensor

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        spatial_filters: Float[torch.Tensor, "neurons height width"],
        temporal_filters: Float[torch.Tensor, "neurons t_filter"],
        gaussian_params: Float[torch.Tensor, "neurons 5"],
        sigma_contour: float,
        sigma_floor: float = 1.0,
        smooth_weight_temporal: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            in_shape=in_shape,
            outdims=outdims,
            spatial_filters=spatial_filters,
            temporal_filters=temporal_filters,
            **kwargs,
        )
        if sigma_floor * sigma_contour < 1.0:
            raise ValueError(
                f"{sigma_floor=} with {sigma_contour=} lets the mask shrink to a single pixel, where the spatial "
                "filter's parameters get no gradient. Raise sigma_floor to at least 1 / sigma_contour."
            )
        assert gaussian_params.shape == (outdims, 5)

        self.sigma_contour = sigma_contour
        self.sigma_floor = sigma_floor
        height, width = spatial_filters.shape[-2:]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
        )
        self.register_buffer("x_grid", x_grid, persistent=False)
        self.register_buffer("y_grid", y_grid, persistent=False)

        params = gaussian_params.to(torch.float32)
        self.center_x = nn.Parameter(params[:, 0].clone())
        self.center_y = nn.Parameter(params[:, 1].clone())
        # Widths are softplus(raw) + floor. Start a little above the floor, where softplus is not flat.
        sigma_excess = torch.clamp(params[:, 2:4] - sigma_floor, min=0.05)
        sigma_raw = sigma_excess + torch.log(-torch.expm1(-sigma_excess))  # inverse softplus
        self.sigma_x_raw = nn.Parameter(sigma_raw[:, 0].clone())
        self.sigma_y_raw = nn.Parameter(sigma_raw[:, 1].clone())
        self.theta = nn.Parameter(params[:, 4].clone())

        # Named apart from the `temporal_filters` buffer, which keeps the initial filters
        self.temporal_weight = nn.Parameter(temporal_filters.to(torch.float32).clone())
        # Neurons without a usable filter (e.g. a NaN STA) start all-zero. Normalizing a zero vector is
        # ill-conditioned, so they stay switched off instead of feeding huge gradients into the global clipping.
        self.register_buffer("active", (temporal_filters.abs().sum(dim=-1) > 0).to(torch.float32))

        self.smooth_weight_temporal = smooth_weight_temporal
        # Not persistent, so checkpoints from before the penalty existed still load
        self._temporal_smooth_reg_fn = TimeLaplaceL23dnorm(persistent_buffer=False)

    def sigmas(self) -> tuple[Float[torch.Tensor, " neurons"], Float[torch.Tensor, " neurons"]]:
        return F.softplus(self.sigma_x_raw) + self.sigma_floor, F.softplus(self.sigma_y_raw) + self.sigma_floor

    def _filters(
        self,
    ) -> tuple[
        Float[torch.Tensor, "neurons height width"],
        Float[torch.Tensor, "neurons t_filter"],
        Float[torch.Tensor, " neurons"],
    ]:
        sigma_x, sigma_y = self.sigmas()
        gaussian, normalized_distance = gaussian_2d_torch(
            self.x_grid, self.y_grid, self.center_x, self.center_y, sigma_x, sigma_y, self.theta
        )
        # Hard mask as in create_gaussian_mask; gradients reach the parameters through the Gaussian inside it
        mask = (normalized_distance <= self.sigma_contour).to(gaussian.dtype).detach()
        spatial = gaussian * mask
        spatial = spatial / spatial.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)[:, None, None]

        temporal = F.normalize(self.temporal_weight, dim=-1, eps=1e-8) * self.active[:, None]
        return spatial, temporal, spatial.sum(dim=(-2, -1))

    def regularizer(self, reduction: Literal["sum", "mean", None] = "sum") -> torch.Tensor:
        if self.smooth_weight_temporal == 0:
            return torch.tensor(0.0, device=self.w.device)
        _, temporal, _ = self._filters()
        return self.smooth_weight_temporal * self._temporal_smooth_reg_fn(temporal[:, None, :, None, None])

    def filter_parameters(self) -> list[nn.Parameter]:
        return [self.center_x, self.center_y, self.sigma_x_raw, self.sigma_y_raw, self.theta, self.temporal_weight]

    def apply_constraints(self) -> None:
        super().apply_constraints()
        # Keep the receptive field centres on the stimulus
        self.center_x.data.clamp_(min=0.0, max=float(self.x_grid.shape[-1] - 1))
        self.center_y.data.clamp_(min=0.0, max=float(self.y_grid.shape[-2] - 1))


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

    _base_readout_cls: type[Readout] | None = SpatialContrastReadout
    # Typed separately from `_base_readout_cls` so the filter arguments type-check
    _session_readout_cls: type[SpatialContrastReadout] = SpatialContrastReadout

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
        session_readout_kwargs: dict[str, dict[str, Any]] | None = None,
        **readout_kwargs: Any,
    ):
        # Bypass MultiReadoutBase.__init__ since we need per-session filter tensors
        nn.ModuleDict.__init__(self)

        self._in_shape = in_shape
        self._readout_kwargs = {
            "w_init": w_init,
            "a_init": a_init,
            "b_init": b_init,
            "c_init": c_init,
            "neuron_chunk_size": neuron_chunk_size,
            **readout_kwargs,
        }
        self.readout_reg_avg = readout_reg_avg
        self.readout_reg_reduction: Literal["mean", "sum"] = "mean" if readout_reg_avg else "sum"

        for data_key in n_neurons_dict:
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None
            per_session = session_readout_kwargs.get(data_key, {}) if session_readout_kwargs is not None else {}
            self.add_module(
                data_key,
                self._session_readout_cls(
                    in_shape=in_shape,
                    outdims=n_neurons_dict[data_key],
                    spatial_filters=spatial_filters_dict[data_key],
                    temporal_filters=temporal_filters_dict[data_key],
                    mean_activity=mean_activity,
                    **self._readout_kwargs,
                    **per_session,
                ),
            )


class MultiLearnableFilterSpatialContrastReadout(MultiSpatialContrastReadout):
    """Multi-session wrapper for LearnableFilterSpatialContrastReadout.

    Args:
        gaussian_params_dict: Per-session initial Gaussians [N, 5], columns (center_x, center_y, sigma_x,
            sigma_y, theta).
        sigma_contour: Spatial filter mask contour in sigmas.
        sigma_floor: Lower bound on the Gaussian widths, in pixels.
        smooth_weight_temporal: Weight of the temporal smoothness penalty, see LearnableFilterSpatialContrastReadout.
        Remaining arguments: see MultiSpatialContrastReadout.
    """

    _base_readout_cls = LearnableFilterSpatialContrastReadout
    _session_readout_cls: type[SpatialContrastReadout] = LearnableFilterSpatialContrastReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        spatial_filters_dict: dict[str, Float[torch.Tensor, "neurons height width"]],
        temporal_filters_dict: dict[str, Float[torch.Tensor, "neurons t_filter"]],
        gaussian_params_dict: dict[str, Float[torch.Tensor, "neurons 5"]],
        sigma_contour: float,
        sigma_floor: float = 1.0,
        smooth_weight_temporal: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            spatial_filters_dict=spatial_filters_dict,
            temporal_filters_dict=temporal_filters_dict,
            session_readout_kwargs={key: {"gaussian_params": params} for key, params in gaussian_params_dict.items()},
            sigma_contour=sigma_contour,
            sigma_floor=sigma_floor,
            smooth_weight_temporal=smooth_weight_temporal,
            **kwargs,
        )
