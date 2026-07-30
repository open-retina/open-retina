"""
Multi-neuron Spatial Contrast readout following the core-readout pattern.

SpatialContrastReadout handles N neurons in a single session with fixed or
learnable spatial/temporal filters and per-neuron parameters (w, a, b, c).

MultiSpatialContrastReadout wraps multiple sessions, each with its own
SpatialContrastReadout, following the MultiReadoutBase interface.
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int

from openretina.modules.readout.base import Readout
from openretina.modules.readout.multi_readout import MultiReadoutBase
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.sta_processing import load_sta_and_extract_filters


def _matching_filter_rows(available_ids: Any, requested_ids: Any, session: str) -> list[int]:
    available = np.asarray(available_ids)
    requested = np.asarray(requested_ids)
    if available.ndim != 1 or requested.ndim != 1:
        raise ValueError(f"Neuron IDs for session {session!r} must be one-dimensional")

    rows = []
    for neuron_id in requested:
        matches = np.flatnonzero(available == neuron_id)
        if len(matches) != 1:
            raise ValueError(f"Filter bank has no unique row for neuron {neuron_id!r} in session {session!r}")
        rows.append(int(matches[0]))
    return rows


class SpatialContrastReadout(Readout):
    """Single-session, multi-neuron Spatial Contrast readout.

    Each neuron has provided spatial and temporal filters and 4 learnable scalar
    parameters: w (contrast weight) and a, b, c
    (nonlinearity shape). Optionally, the temporal filter and a Gaussian
    parameterization of the spatial filter are learned as well.

    The forward pass processes neurons in chunks to limit GPU memory:
    for each chunk of K neurons, temporal filtering (conv1d) and spatial
    contraction (einsum) are done together, keeping the peak 5D tensor
    at [batch, K, t_out, H, W] instead of [batch, N, t_out, H, W].

    Args:
        in_shape: (channels, time, height, width) of core output.
        outdims: Number of neurons in this session.
        spatial_filters: Initial spatial filters, shape [N, H, W].
        temporal_filters: Initial temporal filters, shape [N, T_filter] or
            [N, channels, T_filter].
        w_init: Initial contrast weight.
        a_init: Initial output scaling (nl_a).
        b_init: Initial input gain (nl_b).
        c_init: Initial offset (nl_c).
        mean_activity: Optional mean activity for bias initialization.
        neuron_chunk_size: Max neurons processed at once in forward pass.
            Controls the memory/speed trade-off. Default 32 works on 8 GB GPUs.
        learnable_filters: Learn temporal filters and Gaussian RF center/size.
        temporal_smoothness_reg: Weight on temporal second differences.
        spatial_center_reg: Weight on RF center displacement from initialization.
        spatial_scale_reg: Weight on log RF scale displacement from initialization.
    """

    # Registered buffers (declared for type checking)
    spatial_filters: torch.Tensor
    temporal_filters: torch.Tensor
    sf_sums: torch.Tensor
    spatial_indices: torch.Tensor
    spatial_weights: torch.Tensor
    spatial_grid_y: torch.Tensor
    spatial_grid_x: torch.Tensor
    initial_spatial_mu: torch.Tensor
    initial_spatial_log_sigma: torch.Tensor

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
        learnable_filters: bool = False,
        temporal_smoothness_reg: float = 0.0,
        spatial_center_reg: float = 0.0,
        spatial_scale_reg: float = 0.0,
    ):
        super().__init__()
        if spatial_filters.shape != (outdims, *in_shape[-2:]):
            raise ValueError(f"Expected spatial filters {(outdims, *in_shape[-2:])}, got {spatial_filters.shape}")
        if temporal_filters.ndim == 2:
            valid_temporal_shape = in_shape[0] == 1 and temporal_filters.shape[0] == outdims
        else:
            valid_temporal_shape = temporal_filters.ndim == 3 and temporal_filters.shape[:2] == (
                outdims,
                in_shape[0],
            )
        if not valid_temporal_shape:
            raise ValueError(
                f"Expected temporal filters [neurons, time] for one channel or "
                f"[neurons, {in_shape[0]}, time], got {temporal_filters.shape}"
            )
        if temporal_filters.shape[-1] > in_shape[1]:
            raise ValueError("Temporal filters cannot be longer than the input")
        if neuron_chunk_size < 1:
            raise ValueError("neuron_chunk_size must be positive")

        self.outdims = outdims
        self.in_shape = in_shape
        self.neuron_chunk_size = neuron_chunk_size
        self.learnable_filters = learnable_filters
        self.temporal_smoothness_reg = temporal_smoothness_reg
        self.spatial_center_reg = spatial_center_reg
        self.spatial_scale_reg = spatial_scale_reg

        self.register_buffer("spatial_filters", spatial_filters)
        if learnable_filters:
            self.temporal_filters = nn.Parameter(temporal_filters.clone())
        else:
            self.register_buffer("temporal_filters", temporal_filters)
        self.register_buffer("sf_sums", spatial_filters.sum(dim=(-2, -1)))

        height, width = in_shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=spatial_filters.dtype),
            torch.arange(width, dtype=spatial_filters.dtype),
            indexing="ij",
        )
        weights = spatial_filters.abs()
        weight_sums = weights.sum(dim=(-2, -1)).clamp_min(1e-8)
        initial_mu = torch.stack(
            (
                einsum(weights, grid_y, "n h w, h w -> n") / weight_sums,
                einsum(weights, grid_x, "n h w, h w -> n") / weight_sums,
            ),
            dim=-1,
        )
        initial_variance = torch.stack(
            (
                einsum(weights, (grid_y[None] - initial_mu[:, 0, None, None]) ** 2, "n h w, n h w -> n") / weight_sums,
                einsum(weights, (grid_x[None] - initial_mu[:, 1, None, None]) ** 2, "n h w, n h w -> n") / weight_sums,
            ),
            dim=-1,
        )
        initial_log_sigma = initial_variance.clamp_min(0.25).sqrt().log()
        self.register_buffer("spatial_grid_y", grid_y, persistent=False)
        self.register_buffer("spatial_grid_x", grid_x, persistent=False)
        self.register_buffer("initial_spatial_mu", initial_mu, persistent=False)
        self.register_buffer("initial_spatial_log_sigma", initial_log_sigma, persistent=False)
        if learnable_filters:
            self.spatial_mu = nn.Parameter(initial_mu.clone())
            self.spatial_log_sigma = nn.Parameter(initial_log_sigma.clone())
        else:
            self.register_parameter("spatial_mu", None)
            self.register_parameter("spatial_log_sigma", None)

        # Sparse support keeps the fixed-filter model fast.
        support_sizes = spatial_filters.ne(0).sum(dim=(-2, -1))
        max_support = int(support_sizes.max())
        spatial_indices = torch.zeros((outdims, max_support), dtype=torch.long)
        spatial_weights = torch.zeros((outdims, max_support), dtype=spatial_filters.dtype)
        flat_filters = spatial_filters.flatten(1)
        for neuron, support_size in enumerate(support_sizes.tolist()):
            indices = flat_filters[neuron].nonzero().flatten()
            spatial_indices[neuron, :support_size] = indices
            spatial_weights[neuron, :support_size] = flat_filters[neuron, indices]
        self.register_buffer("spatial_indices", spatial_indices, persistent=False)
        self.register_buffer("spatial_weights", spatial_weights, persistent=False)

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
        if not self.learnable_filters:
            return torch.tensor(0.0, device=self.w.device)

        kernels = self._normalized_temporal_filters()
        temporal_penalty = kernels.diff(n=2, dim=-1).square().mean(dim=(-2, -1))
        assert self.spatial_mu is not None and self.spatial_log_sigma is not None
        center_penalty = (self.spatial_mu - self.initial_spatial_mu).square().mean(dim=-1)
        scale_penalty = (self.spatial_log_sigma - self.initial_spatial_log_sigma).square().mean(dim=-1)
        penalty = (
            self.temporal_smoothness_reg * temporal_penalty
            + self.spatial_center_reg * center_penalty
            + self.spatial_scale_reg * scale_penalty
        )
        return self.apply_reduction(penalty, reduction)

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        spatial_filter = self._current_spatial_filters()[neuron_id].detach().cpu().numpy()
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

    def apply_constraints(self) -> None:
        with torch.no_grad():
            self.w.clamp_(min=0)
            self.nl_a.clamp_(min=0)
            if self.learnable_filters:
                assert self.spatial_mu is not None and self.spatial_log_sigma is not None
                self.spatial_mu[:, 0].clamp_(0, self.in_shape[-2] - 1)
                self.spatial_mu[:, 1].clamp_(0, self.in_shape[-1] - 1)
                self.spatial_log_sigma.clamp_(
                    min=float(np.log(0.5)),
                    max=float(np.log(max(self.in_shape[-2:]))),
                )

    def _normalized_temporal_filters(self) -> torch.Tensor:
        kernels = self.temporal_filters
        if kernels.ndim == 2:
            kernels = kernels[:, None, :]
        if self.learnable_filters:
            kernels = kernels / kernels.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
        return kernels

    def _current_spatial_filters(self) -> torch.Tensor:
        if not self.learnable_filters:
            return self.spatial_filters
        assert self.spatial_mu is not None and self.spatial_log_sigma is not None
        sigma = self.spatial_log_sigma.exp()
        distance = (self.spatial_grid_y[None] - self.spatial_mu[:, 0, None, None]) ** 2 / sigma[
            :, 0, None, None
        ] ** 2 + (self.spatial_grid_x[None] - self.spatial_mu[:, 1, None, None]) ** 2 / sigma[:, 1, None, None] ** 2
        filters = torch.exp(-0.5 * distance)
        return filters / filters.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def _nonlinearity(
        self,
        imean: torch.Tensor,
        lsc: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
    ) -> torch.Tensor:
        combined = imean + self.w[chunk_start:chunk_end][None, :, None] * lsc
        return self.nl_a[chunk_start:chunk_end][None, :, None] * F.softplus(
            self.nl_b[chunk_start:chunk_end][None, :, None] * combined + self.nl_c[chunk_start:chunk_end][None, :, None]
        )

    def _forward_chunk(
        self,
        x: Float[torch.Tensor, "batch channels time pixels"],
        chunk_start: int,
        chunk_end: int,
        batch: int,
    ) -> Float[torch.Tensor, "batch chunk time_out"]:
        """Process a chunk of neurons: temporal filtering + spatial contraction + nonlinearity."""
        # Select only nonzero RF pixels and apply each neuron's temporal filter
        # with grouped convolution.
        indices = self.spatial_indices[chunk_start:chunk_end]
        weights = self.spatial_weights[chunk_start:chunk_end]
        selected = x[..., indices]
        selected = rearrange(selected, "b c t k p -> (b p) (k c) t")
        kernels = self._normalized_temporal_filters()[chunk_start:chunk_end]
        tf_chunk = F.conv1d(selected, kernels, groups=len(kernels))
        tf_chunk = rearrange(tf_chunk, "(b p) k t -> b k t p", b=batch, p=indices.shape[1])

        # Spatially-weighted mean
        sf_sums = self.sf_sums[chunk_start:chunk_end] + 1e-8  # [K]
        imean = einsum(tf_chunk, weights, "b k t p, k p -> b k t") / sf_sums[None, :, None]

        # Local spatial contrast via variance decomposition: Var = E[x^2] - E[x]^2
        mean_sq = einsum(tf_chunk**2, weights, "b k t p, k p -> b k t") / sf_sums[None, :, None]
        variance = mean_sq - imean**2
        lsc = torch.sqrt(torch.clamp(variance, min=1e-6))

        return self._nonlinearity(imean, lsc, chunk_start, chunk_end)

    def _forward_learnable_chunk(
        self,
        x: Float[torch.Tensor, "batch_pixels channels time"],
        spatial_filters: Float[torch.Tensor, "neurons height width"],
        chunk_start: int,
        chunk_end: int,
        batch: int,
    ) -> Float[torch.Tensor, "batch chunk time_out"]:
        kernels = self._normalized_temporal_filters()[chunk_start:chunk_end]
        filtered = F.conv1d(x, kernels)
        filtered = rearrange(
            filtered,
            "(b h w) k t -> b k t h w",
            b=batch,
            h=self.in_shape[-2],
            w=self.in_shape[-1],
        )
        weights = spatial_filters[chunk_start:chunk_end]
        imean = einsum(filtered, weights, "b k t h w, k h w -> b k t")
        mean_sq = einsum(filtered.square(), weights, "b k t h w, k h w -> b k t")
        lsc = torch.sqrt((mean_sq - imean.square()).clamp_min(1e-6))
        return self._nonlinearity(imean, lsc, chunk_start, chunk_end)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time height width"],
        data_key: str | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch time_out neurons"]:
        self.apply_constraints()
        batch, channels, time, h, w = x.shape
        if channels != self.in_shape[0] or (h, w) != tuple(self.in_shape[-2:]):
            expected = f"(*, {self.in_shape[0]}, time, {self.in_shape[-2]}, {self.in_shape[-1]})"
            raise ValueError(f"Expected input shape {expected}, got {x.shape}")
        n = self.outdims

        if self.learnable_filters:
            x_for_readout = rearrange(x, "b c t h w -> (b h w) c t")
            current_spatial_filters = self._current_spatial_filters()
        else:
            x_for_readout = x.flatten(-2)
            current_spatial_filters = self.spatial_filters

        chunk_outputs: list[torch.Tensor] = []
        for chunk_start in range(0, n, self.neuron_chunk_size):
            chunk_end = min(chunk_start + self.neuron_chunk_size, n)
            if self.learnable_filters:
                chunk_out = self._forward_learnable_chunk(
                    x_for_readout,
                    current_spatial_filters,
                    chunk_start,
                    chunk_end,
                    batch,
                )
            else:
                chunk_out = self._forward_chunk(x_for_readout, chunk_start, chunk_end, batch)
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
        filter_bank_path: Torch file containing the two filter dictionaries and,
            optionally, ``neuron_ids_dict`` for stable-ID-based row selection.
        sta_dir: Alternative source directory for per-neuron STA files.
        data_info: Runtime session metadata injected by ``UnifiedCoreReadout``.
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
        spatial_filters_dict: dict[str, Float[torch.Tensor, "neurons height width"]] | None = None,
        temporal_filters_dict: dict[str, Float[torch.Tensor, "neurons t_filter"]] | None = None,
        filter_bank_path: str | Path | None = None,
        sta_dir: str | Path | None = None,
        sta_file_pattern: str = "cell_data_{retina_index}_WN_stas_cell_{cell_index}.npy",
        flip_sta: bool = False,
        temporal_crop_frames: int | None = 30,
        sigma_contour: float = 3.0,
        data_info: dict[str, Any] | None = None,
        w_init: float = 0.0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
        readout_reg_avg: bool = False,
        neuron_chunk_size: int = 32,
        learnable_filters: bool = False,
        temporal_smoothness_reg: float = 0.0,
        spatial_center_reg: float = 0.0,
        spatial_scale_reg: float = 0.0,
    ):
        direct_filters = spatial_filters_dict is not None or temporal_filters_dict is not None
        if (spatial_filters_dict is None) != (temporal_filters_dict is None):
            raise ValueError("spatial_filters_dict and temporal_filters_dict must be provided together")
        if sum((direct_filters, filter_bank_path is not None, sta_dir is not None)) != 1:
            raise ValueError("Provide exactly one SC filter source: dictionaries, filter_bank_path, or sta_dir")

        available_neuron_ids = None
        if filter_bank_path is not None:
            filter_bank = torch.load(
                get_local_file_path(Path(filter_bank_path).as_posix()), map_location="cpu", weights_only=True
            )
            spatial_filters_dict = filter_bank["spatial_filters_dict"]
            temporal_filters_dict = filter_bank["temporal_filters_dict"]
            available_neuron_ids = filter_bank.get("neuron_ids_dict")
        elif sta_dir is not None:
            if data_info is None:
                raise ValueError("data_info is required when loading SC filters from STAs")
            spatial_filters_dict, temporal_filters_dict = {}, {}
            local_sta_dir = get_local_file_path(Path(sta_dir).as_posix())
            for session_key in n_neurons_dict:
                cell_indices = data_info.get("sessions_kwargs", {}).get(session_key, {}).get("cell_indices")
                if cell_indices is None:
                    cell_indices = data_info.get("neuron_ids_dict", {}).get(session_key)
                if cell_indices is None:
                    raise ValueError(f"Neuron IDs are required to load STA filters for session {session_key!r}")
                filters = [
                    load_sta_and_extract_filters(
                        sta_dir=local_sta_dir,
                        file_name=sta_file_pattern.format(retina_index=session_key, cell_index=cell_index),
                        flip_sta=flip_sta,
                        target_spatial_shape=in_shape[-2:],
                        temporal_crop_frames=temporal_crop_frames,
                        sigma_contour=sigma_contour,
                    )[:2]
                    for cell_index in cell_indices
                ]
                spatial_filters_dict[session_key] = torch.stack([torch.from_numpy(x[0]) for x in filters])
                temporal_filters_dict[session_key] = torch.stack([torch.from_numpy(x[1]) for x in filters])

        if spatial_filters_dict is None or temporal_filters_dict is None:
            raise RuntimeError("SC filter source resolution failed")
        missing_sessions = set(n_neurons_dict) - spatial_filters_dict.keys() | (
            set(n_neurons_dict) - temporal_filters_dict.keys()
        )
        if missing_sessions:
            raise ValueError(f"SC filters are missing sessions: {sorted(missing_sessions)}")

        selected_neuron_ids = {} if data_info is None else data_info.get("neuron_ids_dict", {})
        resolved_spatial_filters = {}
        resolved_temporal_filters = {}
        for session_key, n_neurons in n_neurons_dict.items():
            spatial_filters = spatial_filters_dict[session_key]
            temporal_filters = temporal_filters_dict[session_key]
            if available_neuron_ids is not None and session_key in selected_neuron_ids:
                if session_key not in available_neuron_ids:
                    raise ValueError(f"Filter bank neuron IDs are missing session {session_key!r}")
                rows = _matching_filter_rows(
                    available_neuron_ids[session_key],
                    selected_neuron_ids[session_key],
                    session_key,
                )
                spatial_filters = spatial_filters[rows]
                temporal_filters = temporal_filters[rows]
            if spatial_filters.shape[0] != n_neurons or temporal_filters.shape[0] != n_neurons:
                raise ValueError(
                    f"Session {session_key!r} has {n_neurons} neurons but "
                    f"{spatial_filters.shape[0]} spatial and {temporal_filters.shape[0]} temporal filters. "
                    "Store neuron_ids_dict in the filter bank to support neuron selection."
                )
            resolved_spatial_filters[session_key] = spatial_filters
            resolved_temporal_filters[session_key] = temporal_filters

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
            "learnable_filters": learnable_filters,
            "temporal_smoothness_reg": temporal_smoothness_reg,
            "spatial_center_reg": spatial_center_reg,
            "spatial_scale_reg": spatial_scale_reg,
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
                    spatial_filters=resolved_spatial_filters[data_key],
                    temporal_filters=resolved_temporal_filters[data_key],
                    w_init=w_init,
                    a_init=a_init,
                    b_init=b_init,
                    c_init=c_init,
                    mean_activity=mean_activity,
                    neuron_chunk_size=neuron_chunk_size,
                    learnable_filters=learnable_filters,
                    temporal_smoothness_reg=temporal_smoothness_reg,
                    spatial_center_reg=spatial_center_reg,
                    spatial_scale_reg=spatial_scale_reg,
                ),
            )

    def add_sessions(
        self,
        n_neurons_dict: dict[str, int],
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ) -> None:
        raise NotImplementedError(
            "Adding SC sessions requires session-specific filters; construct a new readout instead"
        )
