"""
Spatial Contrast (SC) models for neural response prediction.

Extends the Linear-Nonlinear (LN) model by adding a local spatial contrast term.
The spatial and temporal filters are frozen from pre-computed STAs; only 4 parameters
per neuron are learned: the contrast weight (w) and three nonlinearity parameters (a, b, c).

Provides two variants:
- SingleCellSpatialContrast: standalone LightningModule for one neuron at a time
- SpatialContrastCoreReadout: multi-cell core-readout model using DummyCore + MultiSpatialContrastReadout

Reference: Sridhar S, Vystrčilová M, Khani MH, Karamanlis D, Schreyer HM, Ramakrishna V,
Krüppel S, Zapp SJ, Mietsch M, Ecker AS, Gollisch T (2025): Modeling spatial contrast
sensitivity in responses of primate retinal ganglion cells to natural movies.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import DummyCore
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.nonlinearities import ParametrizedSoftplus
from openretina.modules.readout.spatial_contrast_model_readout import (
    MultiSpatialContrastReadout,
    SpatialContrastReadout,
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
        self.nonlinearity = ParametrizedSoftplus(w=a_init, a=b_init, b=c_init, learn_a=True)
        # The `w` here is different from the `w` used in ParametrizedSoftplus
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
        output = self.nonlinearity(combined)

        return rearrange(output, "batch time -> batch time 1")

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
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

        # To avoid negative model predictions, clamp the gain parameter in the nonlinearity
        self.nonlinearity.w.clamp_(min=0.0)

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
    ``{retina_index}`` and ``{cell_index}`` placeholders that are filled
    from ``data_info["sessions_kwargs"]``.

    Args:
        in_shape: (channels, time, height, width) of input stimulus.
        n_neurons_dict: Session key -> number of neurons.
        sta_dir: Directory containing STA .npy files.
        sta_file_pattern: Filename pattern with {retina_index} and {cell_index}
            placeholders, e.g. "cell_data_{retina_index}_WN_stas_cell_{cell_index}.npy".
        flip_sta: Whether to flip STAs horizontally.
        temporal_crop_frames: Number of most-recent STA frames to keep.
        sigma_contour: Spatial filter mask contour in sigmas.
        cut_first_n_frames: Frames cut by DummyCore (temporal alignment).
        w_init, a_init, b_init, c_init: Initial per-neuron parameter values.
        learning_rate: Learning rate for AdamW.
        neuron_chunk_size: Max neurons processed at once in the readout forward
            pass. Controls the GPU memory/speed trade-off. Default 32 works
            on 8 GB GPUs.
        loss: Training loss module.
        evaluation_loss: Evaluation metric module.
        data_info: Data metadata dict (must contain sessions_kwargs with cell_indices).
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        sta_dir: Union[Path, str],
        sta_file_pattern: str = "cell_data_{retina_index}_WN_stas_cell_{cell_index}.npy",
        flip_sta: bool = False,
        temporal_crop_frames: int | None = 30,
        sigma_contour: float = 3.0,
        cut_first_n_frames: int = 0,
        w_init: float = 0.0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        learning_rate: float = 0.01,
        neuron_chunk_size: int = 32,
        loss: nn.Module | None = None,
        evaluation_loss: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
    ):
        in_shape = tuple(in_shape)
        target_spatial_shape = (in_shape[-2], in_shape[-1])

        local_sta_dir = get_local_file_path(Path(sta_dir).as_posix())

        # Extract cell indices from data_info
        if data_info is None:
            raise ValueError("data_info is required for SpatialContrastCoreReadout")
        sessions_kwargs = data_info.get("sessions_kwargs", {})

        spatial_filters_dict: dict[str, torch.Tensor] = {}
        temporal_filters_dict: dict[str, torch.Tensor] = {}

        for session_key, n_neurons in n_neurons_dict.items():
            session_kw = sessions_kwargs.get(session_key, {})
            cell_indices = session_kw.get("cell_indices")
            if cell_indices is None:
                raise ValueError(
                    f"cell_indices not found in data_info['sessions_kwargs']['{session_key}']. "
                    "Ensure the dataloader provides cell_indices in session_kwargs."
                )

            spatial_list = []
            temporal_list = []
            for cell_idx in cell_indices:
                file_name = sta_file_pattern.format(retina_index=session_key, cell_index=cell_idx)
                spatial_filter, temporal_filter, _ = load_sta_and_extract_filters(
                    sta_dir=local_sta_dir,
                    file_name=file_name,
                    flip_sta=flip_sta,
                    target_spatial_shape=target_spatial_shape,
                    temporal_crop_frames=temporal_crop_frames,
                    sigma_contour=sigma_contour,
                )
                spatial_list.append(torch.from_numpy(spatial_filter))
                temporal_list.append(torch.from_numpy(temporal_filter))

            spatial_filters_dict[session_key] = torch.stack(spatial_list)
            temporal_filters_dict[session_key] = torch.stack(temporal_list)

            LOGGER.info(
                f"Session {session_key}: loaded {len(cell_indices)} STA filters "
                f"(spatial: {spatial_filters_dict[session_key].shape}, "
                f"temporal: {temporal_filters_dict[session_key].shape})"
            )

        core = DummyCore(cut_first_n_frames=cut_first_n_frames)

        mean_activity_dict = data_info.get("mean_activity_dict")

        readout = MultiSpatialContrastReadout(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            spatial_filters_dict=spatial_filters_dict,
            temporal_filters_dict=temporal_filters_dict,
            w_init=w_init,
            a_init=a_init,
            b_init=b_init,
            c_init=c_init,
            mean_activity_dict=mean_activity_dict,
            neuron_chunk_size=neuron_chunk_size,
        )

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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Clamp nl_a >= 0 after each step to avoid negative predictions."""
        optimizer.step(closure=optimizer_closure)

        for readout in self.readout.values():
            assert isinstance(readout, SpatialContrastReadout)
            readout.nl_a.data.clamp_(min=0.0)
