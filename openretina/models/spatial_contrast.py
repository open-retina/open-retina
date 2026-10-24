"""
Spatial Contrast (SC) models for neural response prediction.

Extends the Linear-Nonlinear (LN) model by adding a local spatial contrast term.
The spatial and temporal filters are frozen from pre-computed STAs; only 4 parameters
per neuron are learned: the contrast weight (w) and three nonlinearity parameters (a, b, c).

Provides two variants:
- SingleCellSpatialContrast: standalone LightningModule for one neuron at a time
- SpatialContrastCoreReadout: multi-cell core-readout model using DummyCore + MultiSpatialContrastReadout,
  optionally learning the filters too (MultiLearnableFilterSpatialContrastReadout)

Reference: Sridhar S, Vystrčilová M, Khani MH, Karamanlis D, Schreyer HM, Ramakrishna V,
Krüppel S, Zapp SJ, Mietsch M, Ecker AS, Gollisch T (2025): Modeling spatial contrast
sensitivity in responses of primate retinal ganglion cells to natural movies.
"""

import json
import logging
from pathlib import Path
from typing import IO, Any, Callable, Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from lightning import LightningModule
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import DummyCore
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.readout.spatial_contrast_model_readout import (
    MultiLearnableFilterSpatialContrastReadout,
    MultiSpatialContrastReadout,
    SpatialContrastReadout,
    gaussian_2d_torch,
)
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.sta_processing import load_sta_and_extract_filters

LOGGER = logging.getLogger(__name__)


class SingleCellSpatialContrast(LightningModule):
    """
    Single-cell Spatial Contrast model.

    Predicts the response of a single neuron as:

        y = a * log(1 + exp(b * (imean + w * lsc) + c))

    where imean is the spatially-weighted temporal filter output and lsc is the
    local spatial contrast (weighted standard deviation of the filtered stimulus).

    The spatial filter (a 2D Gaussian fit to the STA) and temporal filter (the STA
    time course at the RF center) are frozen. Only 4 scalar parameters are learned:
    w (contrast weight) and a, b, c (nonlinearity shape).

    The model works with a single channel in the input; multiple channels will raise
    an error.

    Args:
        in_shape: (channels, time, height, width) of the input stimulus (no batch dim).
        sta_dir: Directory containing STA .npy files.
        sta_file_name: STA filename for this cell (e.g. 'cell_data_04_WN_stas_cell_2.npy').
        flip_sta: Whether to flip the STA horizontally. Set via data_io config; needed
            when the stimulus and STA coordinate systems are mirrored (e.g. NM dataset).
        temporal_crop_frames: How many of the most recent STA frames to keep. None = all.
        sigma_contour: Spatial filter is masked to this many standard deviations of
            the fitted Gaussian.
        spat_crop_size: (height, width) patch to crop around the RF center. Reduces
            computation by discarding stimulus pixels far from the receptive field.
            None = use full stimulus.
        learning_rate: Learning rate for AdamW optimizer.
        w_init: Initial contrast weight. 0 = start as a pure LN model.
        a_init, b_init, c_init: Initial nonlinearity parameters.
        loss: Training loss. Defaults to PoissonLoss3d.
        validation_loss: Validation metric. Defaults to CorrelationLoss3d.
    """

    # Registered buffers
    spatial_filter: torch.Tensor
    temporal_filter: torch.Tensor
    spatial_filter_sum: torch.Tensor

    _crop_bounds: tuple[int, int, int, int] | None

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        sta_dir: Union[Path, str],
        sta_file_name: str,
        flip_sta: bool = False,
        temporal_crop_frames: int | None = None,
        sigma_contour: float = 3.0,
        spat_crop_size: tuple[int, int] | None = None,
        learning_rate: float = 1e-3,
        w_init: float = 0.0,
        a_init: float = 1.0,
        b_init: float = 1.0,
        c_init: float = 0.0,
        loss=None,
        validation_loss=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "validation_loss"])

        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d(per_neuron=True, avg=True)
        self.validation_loss = validation_loss if validation_loss is not None else CorrelationLoss3d(avg=True)

        # Extract frozen filters from STA
        # target_spatial_shape ensures the STA is cropped to match the stimulus dimensions
        target_spatial_shape = (in_shape[-2], in_shape[-1])
        local_sta_dir = get_local_file_path(Path(sta_dir).as_posix())
        spatial_filter, temporal_filter, gaussian_params = load_sta_and_extract_filters(
            sta_dir=local_sta_dir,
            file_name=sta_file_name,
            flip_sta=flip_sta,
            target_spatial_shape=target_spatial_shape,
            temporal_crop_frames=temporal_crop_frames,
            sigma_contour=sigma_contour,
        )

        self.gaussian_params = gaussian_params

        # RF center in the (possibly cropped) STA coordinate system
        full_rf_location = (int(gaussian_params["center_y"]), int(gaussian_params["center_x"]))

        self.in_shape = in_shape

        self.in_channels = in_shape[0]
        if self.in_channels > 1:
            raise ValueError(
                "The spatial contrast model only supports a single channel as input. "
                "If the data has multiple channels, consider providing an average across channels "
                "as input."
            )

        # Optionally crop a small patch around the RF to speed up computation
        self.spat_crop_size = spat_crop_size
        self.do_crop = spat_crop_size is not None and (
            in_shape[-1] != spat_crop_size[1] or in_shape[-2] != spat_crop_size[0]
        )

        if self.do_crop and spat_crop_size is not None:
            crop_h, crop_w = spat_crop_size
            h, w = spatial_filter.shape
            center_y, center_x = full_rf_location

            # Boundary-safe crop
            y_start = max(0, center_y - crop_h // 2)
            y_end = min(h, y_start + crop_h)
            y_start = max(0, y_end - crop_h)

            x_start = max(0, center_x - crop_w // 2)
            x_end = min(w, x_start + crop_w)
            x_start = max(0, x_end - crop_w)

            spatial_filter = spatial_filter[y_start:y_end, x_start:x_end]

            self.rf_location = full_rf_location
            self._crop_bounds = (y_start, y_end, x_start, x_end)
        else:
            self.rf_location = full_rf_location
            self._crop_bounds = None

        # Filters are registered as buffers: non-trainable but saved in state_dict
        self.register_buffer("spatial_filter", torch.from_numpy(spatial_filter))
        self.register_buffer("temporal_filter", torch.from_numpy(temporal_filter))
        self.register_buffer("spatial_filter_sum", torch.tensor(spatial_filter.sum()))

        # The 4 learnable parameters
        self.nl_a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32), requires_grad=True)
        self.nl_b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32), requires_grad=True)
        self.nl_c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32), requires_grad=True)
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32), requires_grad=True)

    def crop_input(self, x: Float[torch.Tensor, "batch channels time height width"]) -> torch.Tensor:
        """Crop input stimulus to a patch around the RF center."""
        if self.spat_crop_size is None:
            return x

        _, _, _, h, w = x.shape
        crop_h, crop_w = self.spat_crop_size
        center_y, center_x = self.rf_location

        y_start = max(0, center_y - crop_h // 2)
        y_end = min(h, y_start + crop_h)
        y_start = max(0, y_end - crop_h)

        x_start = max(0, center_x - crop_w // 2)
        x_end = min(w, x_start + crop_w)
        x_start = max(0, x_end - crop_w)

        return x[:, :, :, y_start:y_end, x_start:x_end]

    def temporal_filter_stimulus(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch time_out height width"]:
        """Apply the temporal filter via 1D convolution (valid mode).

        Output time dimension is reduced by (temporal_filter_length - 1).
        """
        batch, channels, time, h, w = x.shape

        x_reshaped = rearrange(x, "b c t h w -> (b h w) c t")

        # conv1d cross-correlates, which matches STA temporal order directly
        # (index 0 = furthest past, index -1 = closest to spike)
        kernel = self.temporal_filter.view(1, 1, -1)
        filtered = F.conv1d(x_reshaped, kernel, padding=0)

        return rearrange(filtered, "(b h w) 1 t -> b t h w", b=batch, h=h, w=w)

    def compute_imean(
        self, temp_filtered: Float[torch.Tensor, "batch time height width"]
    ) -> Float[torch.Tensor, "batch time"]:
        """Spatially-weighted mean: sum(spatial_filter * stimulus) / spatial_filter_sum over space."""
        weighted = self.spatial_filter * temp_filtered
        return weighted.sum(dim=(-2, -1)) / (self.spatial_filter_sum + 1e-8)

    def compute_lsc(
        self,
        temp_filtered: Float[torch.Tensor, "batch time height width"],
        imean: Float[torch.Tensor, "batch time"],
    ) -> Float[torch.Tensor, "batch time"]:
        """Local spatial contrast: weighted spatial standard deviation.

        lsc = sqrt( sum(spatial_filter * (stimulus - imean)^2) / sum(spatial_filter) )
        """
        imean_broadcast = imean[:, :, None, None]
        deviation_sq = (temp_filtered - imean_broadcast) ** 2
        weighted_var = (self.spatial_filter * deviation_sq).sum(dim=(-2, -1))
        return torch.sqrt(weighted_var / (self.spatial_filter_sum + 1e-8))

    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"], data_key=None, **kwargs
    ) -> Float[torch.Tensor, "batch time neurons"]:
        """Compute predicted firing rate from stimulus.

        Crops around the RF (if configured), applies temporal filtering, computes
        the spatially-weighted mean and local spatial contrast, combines them, and
        passes through the softplus nonlinearity.

        Args:
            x: Input stimulus (batch, channels, time, height, width)
            data_key: Unused; accepted for compatibility with multi-session dataloaders.

        Returns:
            Predicted response (batch, time_out, 1)
        """
        if self.do_crop:
            x = self.crop_input(x)

        temp_filtered = self.temporal_filter_stimulus(x)
        imean = self.compute_imean(temp_filtered)
        lsc = self.compute_lsc(temp_filtered, imean)

        combined = imean + self.w * lsc
        output = self.nl_a * nn.functional.softplus(self.nl_b * combined + self.nl_c)

        return rearrange(output, "batch time -> batch time 1")

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / model_output.numel()
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / model_output.numel()
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Override implementation to allow parameter clamping at each training step"""
        optimizer.step(closure=optimizer_closure)

        # To avoid negative model predictions, clamp the output gain of the nonlinearity
        self.nl_a.data.clamp_(min=0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_decay_factor = 0.3
        patience = 5
        tolerance = 0.0005
        min_lr = self.learning_rate * (lr_decay_factor**3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_factor,
            patience=patience,
            threshold=tolerance,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_correlation",
                "frequency": 1,
            },
        }


class SpatialContrastCoreReadout(BaseCoreReadout):
    """Multi-cell Spatial Contrast model following the core-readout pattern.

    Uses a DummyCore (pass-through) paired with a MultiSpatialContrastReadout
    that handles N neurons per session. Frozen STA-derived spatial and temporal
    filters are loaded at construction time; only 4 parameters per neuron
    (w, a, b, c) are learned.

    Designed for Hydra config instantiation. The STA file pattern uses
    ``{retina_index}`` and ``{cell_index}`` placeholders, filled with the session
    key and with the per-neuron ids found in ``data_info["sessions_kwargs"][session][cell_id_key]``.

    With ``learn_filters=True`` the filters loaded from the STAs only initialise the model: the spatial
    Gaussians (centre, widths, orientation) and the temporal filters are trained along with w, a, b, c.

    The STAs are only needed for training: ``load_from_checkpoint`` takes the filters from the checkpoint, so a
    trained model loads without its ``sta_dir``.

    With ``filter_init="random"`` no STAs are read at all: every neuron starts with the same isotropic Gaussian of
    width ``gaussian_init_sigma`` at the centre of the stimulus, and a temporal filter drawn from U(-a, a) with
    ``a = sqrt(3 / temporal_filter_length)``. This needs ``learn_filters`` and ``temporal_filter_length``.

    Args:
        in_shape: (channels, time, height, width) of input stimulus.
        n_neurons_dict: Session key -> number of neurons.
        sta_dir: Directory containing STA .npy files. Not read when ``filters`` is given or with
            ``filter_init="random"``.
        sta_file_pattern: Filename pattern with {retina_index} and {cell_index}
            placeholders, e.g. "cell_data_{retina_index}_WN_stas_cell_{cell_index}.npy".
        flip_sta: Whether to flip STAs horizontally.
        temporal_crop_frames: Number of most-recent STA frames to keep.
        sigma_contour: Spatial filter mask contour in sigmas.
        cut_first_n_frames: Frames cut by DummyCore (temporal alignment). None means 0.
            Mutually exclusive with target_model_cut_frames.
        target_model_cut_frames: If set, the DummyCore cut is derived so that, together with the
            frames the temporal filter consumes, the model output is this many frames shorter than its
            input. Keeps models with different filter lengths evaluated on the same frames.
        color_squashing_weights: Weights to collapse a multi-channel stimulus to greyscale inside the core.
            Required when the stimulus has more than one channel. Must match the weights the filters were
            learned with, which is checked against ``export_info.json`` in ``sta_dir`` when present.
        cell_id_key: Key in ``data_info["sessions_kwargs"][session]`` holding the per-neuron ids used in
            the STA file names (e.g. "roi_ids" for filters written by export_lnp_filters_as_stas).
        learn_contrast_weight: If False, w is fixed at 0, turning the model into an LN model with the same
            filters and nonlinearity.
        learn_filters: If True, the spatial and temporal filters are trained too.
        temporal_filter_length: With learn_filters, left-pad the temporal filters with zeros to this
            length, so the model can learn a longer filter than the STA has (e.g. from a single frame).
        filter_learning_rate: Learning rate for the filter parameters. None uses learning_rate.
        sigma_floor: With learn_filters, lower bound on the Gaussian widths in pixels.
        filter_init: Where the filters start: "sta" (fitted to the STAs in ``sta_dir``) or "random" (a centred
            Gaussian and a random temporal filter, see above).
        gaussian_init_sigma: With ``filter_init="random"``, the width of the initial Gaussians in pixels.
        init_offset_from_mean_activity: Initialise each neuron's offset c from its mean activity in
            ``data_info["mean_activity_dict"]``, so that a * softplus(c) starts at the neuron's mean rate.
        scheduler_metric: Validation metric the learning rate schedule follows: "correlation"
            (val_evaluation_loss, as BaseCoreReadout) or "poisson" (val_loss, the training objective).
            Use the same metric for the early stopping and checkpoint callbacks.
        w_init, a_init, b_init, c_init: Initial per-neuron parameter values.
        learning_rate: Learning rate for AdamW.
        neuron_chunk_size: Max neurons processed at once in the readout forward
            pass. Controls the GPU memory/speed trade-off. Default 32 works
            on 8 GB GPUs.
        loss: Training loss module.
        evaluation_loss: Evaluation metric module.
        data_info: Data metadata dict (must contain sessions_kwargs with the cell ids).
        filters: Per-session ``(spatial [N, H, W], temporal [N, T])`` filter buffers of a trained model, used
            instead of reading the STAs in ``sta_dir``. For restoring a checkpoint, which ``load_from_checkpoint``
            does: with ``learn_filters`` the learnable Gaussians are not derived from them, and take their values
            from the checkpoint's state dict.
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        sta_dir: Union[Path, str, None],
        sta_file_pattern: str = "cell_data_{retina_index}_WN_stas_cell_{cell_index}.npy",
        flip_sta: bool = False,
        temporal_crop_frames: int | None = 30,
        sigma_contour: float = 3.0,
        cut_first_n_frames: int | None = None,
        target_model_cut_frames: int | None = None,
        color_squashing_weights: Sequence[float] | None = None,
        cell_id_key: str = "cell_indices",
        learn_contrast_weight: bool = True,
        learn_filters: bool = False,
        temporal_filter_length: int | None = None,
        filter_learning_rate: float | None = None,
        sigma_floor: float = 1.0,
        filter_init: Literal["sta", "random"] = "sta",
        gaussian_init_sigma: float = 1.5,
        init_offset_from_mean_activity: bool = False,
        scheduler_metric: Literal["correlation", "poisson"] = "correlation",
        w_init: float = 0.0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        learning_rate: float = 0.01,
        neuron_chunk_size: int = 32,
        loss: nn.Module | None = None,
        evaluation_loss: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
        filters: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        # Note: argument names are not reassigned below, because Lightning saves the hyperparameters from the
        # constructor's local variables, and a derived value would no longer reload.
        stimulus_shape = tuple(in_shape)
        target_spatial_shape = (stimulus_shape[-2], stimulus_shape[-1])

        if stimulus_shape[0] > 1 and color_squashing_weights is None:
            raise ValueError(
                f"The spatial contrast model works on a single channel, but the stimulus has {stimulus_shape[0]}. "
                "Set color_squashing_weights to collapse the channels to greyscale."
            )
        if color_squashing_weights is not None and len(color_squashing_weights) != stimulus_shape[0]:
            raise ValueError(
                f"Got {len(color_squashing_weights)} color_squashing_weights "
                f"for a {stimulus_shape[0]}-channel stimulus."
            )
        if target_model_cut_frames is not None and cut_first_n_frames is not None:
            raise ValueError("Set either cut_first_n_frames or target_model_cut_frames, not both.")

        if data_info is None:
            raise ValueError("data_info is required for SpatialContrastCoreReadout")

        if filter_init not in ("sta", "random"):
            raise ValueError(f"filter_init must be 'sta' or 'random', got {filter_init!r}.")
        if filter_init == "random" and (not learn_filters or temporal_filter_length is None):
            raise ValueError("filter_init='random' needs learn_filters=True and a temporal_filter_length.")

        if filters is None and filter_init == "random":
            assert temporal_filter_length is not None
            spatial_filters_dict, temporal_filters_dict, gaussian_params_dict = self._random_filters(
                n_neurons_dict=n_neurons_dict,
                spatial_shape=target_spatial_shape,
                temporal_filter_length=temporal_filter_length,
                sigma=gaussian_init_sigma,
                sigma_contour=sigma_contour,
            )
        elif filters is None:
            if sta_dir is None:
                raise ValueError("Set sta_dir, or pass the filters themselves.")
            spatial_filters_dict, temporal_filters_dict, gaussian_params_dict = self._load_filters_from_stas(
                sta_dir=sta_dir,
                n_neurons_dict=n_neurons_dict,
                sessions_kwargs=data_info.get("sessions_kwargs", {}),
                sta_file_pattern=sta_file_pattern,
                cell_id_key=cell_id_key,
                flip_sta=flip_sta,
                target_spatial_shape=target_spatial_shape,
                temporal_crop_frames=temporal_crop_frames,
                sigma_contour=sigma_contour,
                color_squashing_weights=color_squashing_weights,
            )
        else:
            spatial_filters_dict = {key: filters[key][0] for key in n_neurons_dict}
            temporal_filters_dict = {key: filters[key][1] for key in n_neurons_dict}
            # The learnable Gaussians are not derived from the filters: restoring a checkpoint overwrites them
            gaussian_params_dict = {key: torch.zeros(n, 5) for key, n in n_neurons_dict.items()}

        filter_lengths = {t.shape[-1] for t in temporal_filters_dict.values()}
        if len(filter_lengths) > 1:
            raise ValueError(f"Temporal filters differ in length across sessions: {sorted(filter_lengths)}.")
        filter_length = filter_lengths.pop()

        if temporal_filter_length is not None:
            if temporal_filter_length < filter_length:
                raise ValueError(
                    f"{temporal_filter_length=} is shorter than the {filter_length}-frame temporal filters; "
                    "use temporal_crop_frames to crop them."
                )
            # Zero-pad on the past side: the last entry multiplies the current frame
            pad = temporal_filter_length - filter_length
            temporal_filters_dict = {k: F.pad(v, (pad, 0)) for k, v in temporal_filters_dict.items()}
            filter_length = temporal_filter_length

        if target_model_cut_frames is not None:
            core_cut_frames = target_model_cut_frames - (filter_length - 1)
            if core_cut_frames < 0:
                raise ValueError(
                    f"{target_model_cut_frames=} is less than the {filter_length - 1} frames a "
                    f"{filter_length}-frame temporal filter consumes."
                )
            LOGGER.info(
                f"Cutting {core_cut_frames} frames in the core so the model cuts {target_model_cut_frames} in total."
            )
        else:
            core_cut_frames = cut_first_n_frames if cut_first_n_frames is not None else 0

        core = DummyCore(
            cut_first_n_frames=core_cut_frames,
            color_squashing_weights=None if color_squashing_weights is None else tuple(color_squashing_weights),
        )

        mean_activity_dict = data_info.get("mean_activity_dict")

        # What the readout receives: greyscale, with the core's frames cut
        readout_in_shape = (1, stimulus_shape[1] - core_cut_frames, *target_spatial_shape)
        readout_kwargs: dict[str, Any] = dict(
            in_shape=readout_in_shape,
            n_neurons_dict=n_neurons_dict,
            spatial_filters_dict=spatial_filters_dict,
            temporal_filters_dict=temporal_filters_dict,
            w_init=w_init,
            a_init=a_init,
            b_init=b_init,
            c_init=c_init,
            mean_activity_dict=mean_activity_dict,
            neuron_chunk_size=neuron_chunk_size,
            init_offset_from_mean_activity=init_offset_from_mean_activity,
        )
        readout: MultiSpatialContrastReadout
        if learn_filters:
            readout = MultiLearnableFilterSpatialContrastReadout(
                gaussian_params_dict=gaussian_params_dict,
                sigma_contour=sigma_contour,
                sigma_floor=sigma_floor,
                **readout_kwargs,
            )
        else:
            readout = MultiSpatialContrastReadout(**readout_kwargs)

        if not learn_contrast_weight:
            # Zero first: AdamW skips parameters without gradients (weight decay included), so w stays at 0
            for session_readout in readout.values():
                assert isinstance(session_readout, SpatialContrastReadout)
                session_readout.w.data.zero_()
                session_readout.w.requires_grad_(False)

        # Default to avg=True Poisson loss to match the single-cell SC model.
        # The BaseCoreReadout default (avg=False → loss.sum()) produces gradient
        # magnitudes that scale with neuron count, causing NaN for large sessions.
        if loss is None:
            loss = PoissonLoss3d(avg=True)

        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            loss=loss,
            evaluation_loss=evaluation_loss,
            data_info=data_info,
        )
        self.filter_learning_rate = filter_learning_rate
        if scheduler_metric not in ("correlation", "poisson"):
            raise ValueError(f"scheduler_metric must be 'correlation' or 'poisson', got {scheduler_metric!r}.")
        self.scheduler_metric = scheduler_metric

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path, IO],
        map_location: Any = None,
        hparams_file: Union[str, Path, None] = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        **kwargs: Any,
    ) -> "SpatialContrastCoreReadout":
        """Load a trained model with the filters stored in its checkpoint, so the STAs it was trained from (its
        ``sta_dir``) need not exist. Arguments as ``LightningModule.load_from_checkpoint``."""
        if "filters" not in kwargs:
            state_dict = pl_load(checkpoint_path, map_location="cpu", weights_only=weights_only)["state_dict"]
            if hasattr(checkpoint_path, "seek"):
                checkpoint_path.seek(0)
            prefix, suffix = "readout.", ".spatial_filters"
            sessions = [key[len(prefix) : -len(suffix)] for key in state_dict if key.endswith(suffix)]
            kwargs["filters"] = {
                session: (state_dict[f"{prefix}{session}{suffix}"], state_dict[f"{prefix}{session}.temporal_filters"])
                for session in sessions
            }
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    @staticmethod
    def _random_filters(
        n_neurons_dict: dict[str, int],
        spatial_shape: tuple[int, int],
        temporal_filter_length: int,
        sigma: float,
        sigma_contour: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-session filters for ``filter_init="random"``, in the layout `_load_filters_from_stas` returns.

        The temporal filters come from torch's global RNG, which Lightning seeds with the run's seed.
        """
        height, width = spatial_shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
        )
        bound = (3.0 / temporal_filter_length) ** 0.5

        spatial_filters_dict: dict[str, torch.Tensor] = {}
        temporal_filters_dict: dict[str, torch.Tensor] = {}
        gaussian_params_dict: dict[str, torch.Tensor] = {}
        for session_key, n_neurons in n_neurons_dict.items():
            gaussian_params = torch.tensor([(width - 1) / 2, (height - 1) / 2, sigma, sigma, 0.0]).repeat(n_neurons, 1)
            gaussian, normalized_distance = gaussian_2d_torch(x_grid, y_grid, *gaussian_params.unbind(dim=1))
            spatial = gaussian * (normalized_distance <= sigma_contour)
            spatial_filters_dict[session_key] = spatial / spatial.flatten(start_dim=1).norm(dim=1)[:, None, None]
            temporal_filters_dict[session_key] = torch.empty(n_neurons, temporal_filter_length).uniform_(-bound, bound)
            gaussian_params_dict[session_key] = gaussian_params
        return spatial_filters_dict, temporal_filters_dict, gaussian_params_dict

    @staticmethod
    def _load_filters_from_stas(
        sta_dir: Union[Path, str],
        n_neurons_dict: dict[str, int],
        sessions_kwargs: dict[str, Any],
        sta_file_pattern: str,
        cell_id_key: str,
        flip_sta: bool,
        target_spatial_shape: tuple[int, int],
        temporal_crop_frames: int | None,
        sigma_contour: float,
        color_squashing_weights: Sequence[float] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-session spatial filters [N, H, W], temporal filters [N, T] and Gaussian fits [N, 5] from the STAs."""
        local_sta_dir = get_local_file_path(Path(sta_dir).as_posix())
        SpatialContrastCoreReadout._check_export_info(local_sta_dir, color_squashing_weights)

        spatial_filters_dict: dict[str, torch.Tensor] = {}
        temporal_filters_dict: dict[str, torch.Tensor] = {}
        gaussian_params_dict: dict[str, torch.Tensor] = {}

        for session_key, n_neurons in n_neurons_dict.items():
            session_kw = sessions_kwargs.get(session_key, {})
            cell_ids = session_kw.get(cell_id_key)
            if cell_ids is None:
                raise ValueError(
                    f"{cell_id_key} not found in data_info['sessions_kwargs']['{session_key}']. "
                    f"Ensure the dataloader provides {cell_id_key} in session_kwargs, or set cell_id_key."
                )
            if len(cell_ids) != n_neurons:
                raise ValueError(f"Session {session_key}: {len(cell_ids)} {cell_id_key} for {n_neurons} neurons.")

            spatial_list = []
            temporal_list = []
            gaussian_list = []
            failed_fits = []
            for cell_id in cell_ids:
                file_name = sta_file_pattern.format(retina_index=session_key, cell_index=int(cell_id))
                spatial_filter, temporal_filter, gaussian_params = load_sta_and_extract_filters(
                    sta_dir=local_sta_dir,
                    file_name=file_name,
                    flip_sta=flip_sta,
                    target_spatial_shape=target_spatial_shape,
                    temporal_crop_frames=temporal_crop_frames,
                    sigma_contour=sigma_contour,
                )
                if not gaussian_params["success"]:
                    failed_fits.append(int(cell_id))
                spatial_list.append(torch.from_numpy(spatial_filter))
                temporal_list.append(torch.from_numpy(temporal_filter))
                gaussian_list.append(
                    [gaussian_params[k] for k in ("center_x", "center_y", "sigma_x", "sigma_y", "theta")]
                )

            if failed_fits:
                LOGGER.warning(
                    f"Session {session_key}: the Gaussian fit failed for {cell_id_key} {failed_fits}; "
                    "their spatial filters fall back to the fit's initial guess."
                )
            filter_lengths = {len(t) for t in temporal_list}
            if len(filter_lengths) > 1:
                raise ValueError(
                    f"Session {session_key}: temporal filters of different lengths {sorted(filter_lengths)}. "
                    "Set temporal_crop_frames to crop them to a common length."
                )

            spatial_filters_dict[session_key] = torch.stack(spatial_list)
            temporal_filters_dict[session_key] = torch.stack(temporal_list)
            gaussian_params_dict[session_key] = torch.tensor(gaussian_list, dtype=torch.float32)

            LOGGER.info(
                f"Session {session_key}: loaded {len(cell_ids)} STA filters "
                f"(spatial: {spatial_filters_dict[session_key].shape}, "
                f"temporal: {temporal_filters_dict[session_key].shape})"
            )

        return spatial_filters_dict, temporal_filters_dict, gaussian_params_dict

    @staticmethod
    def _check_export_info(sta_dir: Union[Path, str], color_squashing_weights: Sequence[float] | None) -> None:
        """Refuse filters that were learned on a differently squashed stimulus than this model will see."""
        export_info_path = Path(sta_dir) / "export_info.json"
        if not export_info_path.exists():
            return
        with open(export_info_path) as f:
            export_info = json.load(f)
        if "color_squashing_weights" not in export_info:
            return
        exported = export_info["color_squashing_weights"]
        matches = (exported is None and color_squashing_weights is None) or (
            exported is not None
            and color_squashing_weights is not None
            and len(exported) == len(color_squashing_weights)
            and all(abs(a - b) < 1e-6 for a, b in zip(exported, color_squashing_weights))
        )
        if not matches:
            raise ValueError(
                f"The filters in {sta_dir} were learned with color_squashing_weights={exported}, but this model "
                f"uses {color_squashing_weights}."
            )

    def configure_optimizers(self):
        filter_parameters = [
            p
            for readout in self.readout.values()
            if isinstance(readout, SpatialContrastReadout)
            for p in readout.filter_parameters()
        ]
        if not filter_parameters and self.scheduler_metric == "correlation":
            return super().configure_optimizers()

        # The filter geometry gets no weight decay: decay would pull the Gaussian centres towards pixel (0, 0) and
        # the widths and orientations towards fixed values, regardless of the data.
        filter_ids = {id(p) for p in filter_parameters}
        other_parameters = [p for p in self.parameters() if id(p) not in filter_ids]
        filter_learning_rate = self.filter_learning_rate or self.learning_rate
        parameter_groups = [{"params": other_parameters, "lr": self.learning_rate}]
        if filter_parameters:
            parameter_groups.append({"params": filter_parameters, "lr": filter_learning_rate, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(parameter_groups)

        # Same schedule as BaseCoreReadout.configure_optimizers, with a minimum learning rate per group. The
        # Poisson loss sits on a small, data-dependent scale, so its plateau threshold is relative.
        lr_decay_factor = 0.3
        if self.scheduler_metric == "poisson":
            monitor, mode, threshold, threshold_mode = "val_loss", "min", 1e-4, "rel"
        else:
            monitor, mode, threshold, threshold_mode = "val_evaluation_loss", "max", 0.0005, "abs"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=lr_decay_factor,
            patience=5,
            threshold=threshold,
            threshold_mode=threshold_mode,
            min_lr=[group["lr"] * lr_decay_factor**3 for group in parameter_groups],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "frequency": 1,
            },
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """After each step, keep nl_a >= 0 to avoid negative predictions (and learned RF centres on the stimulus)."""
        optimizer.step(closure=optimizer_closure)

        for readout in self.readout.values():
            assert isinstance(readout, SpatialContrastReadout)
            readout.apply_constraints()
