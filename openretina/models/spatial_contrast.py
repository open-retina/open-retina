"""
Spatial Contrast (SC) Model for single-cell neural response prediction.

This model extends the Linear-Nonlinear (LN) framework by incorporating a local
spatial contrast term. Unlike the standard LN model where filters are learnable,
the SC model uses pre-computed STAs for spatial/temporal filters, with only
4 learnable parameters in the output nonlinearity and spatial contrast weighting.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from lightning import LightningModule

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.utils.sta_processing import load_sta_and_extract_filters


class SpatialContrastNonlinearity(nn.Module):
    """
    Parametrized softplus nonlinearity: a * log(1 + exp(b*x + c))

    All three parameters (a, b, c) are learnable.

    Args:
        a: Initial value for output scaling parameter (default 1.0)
        b: Initial value for input scaling parameter (default 1.0)
        c: Initial value for offset parameter (default 0.0)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.log(1 + torch.exp(self.b * x + self.c))


class SingleCellSpatialContrast(LightningModule):
    """
    Single-cell Spatial Contrast model implemented as a PyTorch LightningModule.

    Model output formula:
        y = nonlinearity(imean + w * lsc)

    Where:
        - imean = sum(spatial_filter * temporally_filtered_stimulus) over space
        - lsc = sqrt(sum(spatial_filter * (temp_stim - imean)^2) / sum(spatial_filter))
        - nonlinearity(x) = a * log(1 + exp(b*x + c))

    Learnable parameters (4 total):
        - a: Output scaling in softplus
        - b: Input scaling in softplus
        - c: Offset in softplus
        - w: Weight for local spatial contrast

    Non-learnable (frozen from STA):
        - temporal_filter: Registered as buffer
        - spatial_filter: Gaussian-fitted, registered as buffer

    Parameters
    ----------
    in_shape:
        Tuple (channels, time, height, width) describing the expected stimulus shape
        *excluding* batch dimension.
    sta_dir:
        Directory containing STA files.
    sta_file_name:
        Name of the STA file for this cell (e.g., 'cell_data_01_WN_stas_cell_8.npy').
    flip_sta:
        Whether to flip STA horizontally (required for NM dataset compatibility).
    sta_crop:
        Pixels to crop from STA. Can be int (same for all sides) or tuple
        (top, bottom, left, right).
    temporal_crop_frames:
        Number of frames to keep in temporal filter. If None, uses all frames.
    sigma_contour:
        Number of standard deviations for spatial filter contour (default 3.0).
    spat_crop_size:
        Optional (height, width) to crop input stimulus around RF center.
        If None, uses full spatial dimensions.
    learning_rate:
        Learning rate for optimizer.
    w_init:
        Initial value for spatial contrast weight.
    a_init, b_init, c_init:
        Initial values for nonlinearity parameters.
    loss:
        Training loss function. Defaults to PoissonLoss3d().
    validation_loss:
        Validation loss/metric function. Defaults to CorrelationLoss3d(avg=True).

    Input / Output shapes
    ---------------------
    Forward input `x`:
        Tensor of shape (batch, channels, time, height, width).
    Forward output:
        Tensor of shape (batch, time_out, neurons) where neurons=1 and
        time_out = time - temporal_filter_length + 1.
    """

    # Type annotations for registered buffers
    spatial_filter: torch.Tensor
    temporal_filter: torch.Tensor
    spatial_filter_sum: torch.Tensor

    # Type annotation for crop bounds
    _crop_bounds: tuple[int, int, int, int] | None

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        sta_dir: str,
        sta_file_name: str,
        flip_sta: bool = False,
        sta_crop: int | Tuple[int, int, int, int] = 0,
        temporal_crop_frames: int | None = None,
        sigma_contour: float = 3.0,
        spat_crop_size: Tuple[int, int] | None = None,
        learning_rate: float = 1e-3,
        w_init: float = 0.0,
        a_init: float = 1.0,
        b_init: float = 1.0,
        c_init: float = 0.0,
        loss=None,
        validation_loss=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "validation_loss"])

        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d(per_neuron=True, avg=True)
        self.validation_loss = validation_loss if validation_loss is not None else CorrelationLoss3d(avg=True)

        # Extract filters from STA
        spatial_filter, temporal_filter, gaussian_params = load_sta_and_extract_filters(
            sta_dir=sta_dir,
            file_name=sta_file_name,
            flip_sta=flip_sta,
            crop=sta_crop,
            temporal_crop_frames=temporal_crop_frames,
            sigma_contour=sigma_contour,
        )

        # Store Gaussian parameters for reference
        self.gaussian_params = gaussian_params

        # Get RF location from Gaussian fit (in full STA coordinates)
        full_rf_location = (int(gaussian_params["center_y"]), int(gaussian_params["center_x"]))

        # Store input shape info
        self.in_shape = in_shape
        self.in_channels = in_shape[0]

        # Spatial cropping setup - crop both input AND spatial filter to spat_crop_size
        self.spat_crop_size = spat_crop_size
        self.do_crop = spat_crop_size is not None and (
            in_shape[-1] != spat_crop_size[1] or in_shape[-2] != spat_crop_size[0]
        )

        if self.do_crop and spat_crop_size is not None:
            # Crop the spatial filter around the RF location to match spat_crop_size
            crop_h, crop_w = spat_crop_size
            h, w = spatial_filter.shape
            center_y, center_x = full_rf_location

            # Compute crop boundaries (boundary-safe)
            y_start = max(0, center_y - crop_h // 2)
            y_end = min(h, y_start + crop_h)
            y_start = max(0, y_end - crop_h)  # Adjust if we hit the bottom edge

            x_start = max(0, center_x - crop_w // 2)
            x_end = min(w, x_start + crop_w)
            x_start = max(0, x_end - crop_w)  # Adjust if we hit the right edge

            # Crop the spatial filter
            spatial_filter = spatial_filter[y_start:y_end, x_start:x_end]

            # Update RF location to be relative to the cropped filter
            # (should be approximately at the center of the crop)
            self.rf_location = full_rf_location
            self._crop_bounds = (y_start, y_end, x_start, x_end)
        else:
            self.rf_location = full_rf_location
            self._crop_bounds = None

        # Register filters as buffers (non-trainable, but part of state_dict)
        self.register_buffer("spatial_filter", torch.from_numpy(spatial_filter))
        self.register_buffer("temporal_filter", torch.from_numpy(temporal_filter))

        # Pre-compute spatial filter sum for lsc normalization
        self.register_buffer("spatial_filter_sum", torch.tensor(spatial_filter.sum()))

        # Learnable parameters
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float32), requires_grad=True)
        self.nonlinearity = SpatialContrastNonlinearity(a=a_init, b=b_init, c=c_init)

    def crop_input(self, x: Float[torch.Tensor, "batch channels time height width"]) -> torch.Tensor:
        """
        Crop input stimulus around the receptive field center.

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)

        Returns:
            Cropped tensor of shape (batch, channels, time, crop_h, crop_w)
        """
        if self.spat_crop_size is None:
            return x

        _, _, _, h, w = x.shape
        crop_h, crop_w = self.spat_crop_size
        center_y, center_x = self.rf_location

        # Compute crop boundaries (boundary-safe)
        y_start = max(0, center_y - crop_h // 2)
        y_end = min(h, y_start + crop_h)
        y_start = max(0, y_end - crop_h)  # Adjust if we hit the bottom edge

        x_start = max(0, center_x - crop_w // 2)
        x_end = min(w, x_start + crop_w)
        x_start = max(0, x_end - crop_w)  # Adjust if we hit the right edge

        return x[:, :, :, y_start:y_end, x_start:x_end]

    def temporal_filter_stimulus(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch time_out height width"]:
        """
        Apply temporal filter to stimulus using 1D convolution.

        The convolution is applied in "valid" mode, so the output time dimension
        is reduced by (temporal_filter_length - 1).

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)

        Returns:
            Temporally filtered tensor of shape (batch, time_out, height, width)
        """
        batch, channels, time, h, w = x.shape

        # Reshape for conv1d: (batch * h * w, channels, time)
        x_reshaped = rearrange(x, "b c t h w -> (b h w) c t")

        # Prepare kernel shape (out_channels=1, in_channels=1, kernel_size)
        # Note: conv1d performs cross-correlation, which naturally matches STA temporal order
        # (index 0 = furthest past, index -1 = closest to spike)
        kernel = self.temporal_filter.view(1, 1, -1)

        # Apply conv1d (valid mode, no padding)
        filtered = F.conv1d(x_reshaped, kernel, padding=0)

        # Reshape back: (batch, time_out, h, w) where time_out = time - filter_length + 1
        return rearrange(filtered, "(b h w) 1 t -> b t h w", b=batch, h=h, w=w)

    def compute_imean(
        self, temp_filtered: Float[torch.Tensor, "batch time height width"]
    ) -> Float[torch.Tensor, "batch time"]:
        """
        Compute spatial mean weighted by spatial filter.

        imean = sum(spatial_filter * temp_filtered) over spatial dimensions

        Args:
            temp_filtered: Temporally filtered stimulus (batch, time, height, width)

        Returns:
            Weighted spatial mean (batch, time)
        """
        # spatial_filter: (height, width), temp_filtered: (batch, time, height, width)
        # Broadcast and sum over spatial dimensions
        weighted = self.spatial_filter * temp_filtered  # (batch, time, height, width)
        return weighted.sum(dim=(-2, -1))  # (batch, time)

    def compute_lsc(
        self,
        temp_filtered: Float[torch.Tensor, "batch time height width"],
        imean: Float[torch.Tensor, "batch time"],
    ) -> Float[torch.Tensor, "batch time"]:
        """
        Compute local spatial contrast.

        lsc = sqrt(sum(spatial_filter * (temp_filtered - imean)^2) / sum(spatial_filter))

        Args:
            temp_filtered: Temporally filtered stimulus (batch, time, height, width)
            imean: Weighted spatial mean (batch, time)

        Returns:
            Local spatial contrast (batch, time)
        """
        # Broadcast imean to spatial dimensions
        imean_broadcast = imean[:, :, None, None]  # (batch, time, 1, 1)

        # Squared deviation
        deviation_sq = (temp_filtered - imean_broadcast) ** 2

        # Weight by spatial filter and sum
        weighted_var = (self.spatial_filter * deviation_sq).sum(dim=(-2, -1))

        # Normalize by filter sum and take sqrt (add epsilon for numerical stability)
        lsc = torch.sqrt(weighted_var / (self.spatial_filter_sum + 1e-8))

        return lsc  # (batch, time)

    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"], data_key=None, **kwargs
    ) -> Float[torch.Tensor, "batch time neurons"]:
        """
        Forward pass of the Spatial Contrast model.

        Steps:
        1. (Optional) Crop input around RF center
        2. Apply temporal filtering
        3. Compute imean (spatial mean)
        4. Compute lsc (local spatial contrast)
        5. Combine: combined = imean + w * lsc
        6. Apply nonlinearity
        7. Reshape to (batch, time, 1)

        Args:
            x: Input stimulus tensor (batch, channels, time, height, width)
            data_key: Unused, kept for API compatibility

        Returns:
            Predicted neural response (batch, time_out, 1)
        """
        # Step 1: Crop if needed
        if self.do_crop:
            x = self.crop_input(x)

        # Step 2: Apply temporal filtering
        temp_filtered = self.temporal_filter_stimulus(x)

        # Step 3: Compute imean
        imean = self.compute_imean(temp_filtered)

        # Step 4: Compute lsc
        lsc = self.compute_lsc(temp_filtered, imean)

        # Step 5: Combine with learnable weight
        combined = imean + self.w * lsc

        # Step 6: Apply nonlinearity
        output = self.nonlinearity(combined)

        # Step 7: Reshape to (batch, time, 1)
        return rearrange(output, "batch time -> batch time 1")

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        """
        Training step following the existing LN model pattern.

        Args:
            batch: Tuple of (session_id, DataPoint)
            batch_idx: Index of the batch

        Returns:
            Training loss value
        """
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        """
        Validation step following the existing LN model pattern.

        Args:
            batch: Tuple of (session_id, DataPoint)
            batch_idx: Index of the batch

        Returns:
            Validation loss value
        """
        session_id, data_point = batch

        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer with learning rate scheduler.

        Uses AdamW optimizer with ReduceLROnPlateau scheduler that monitors
        validation correlation.
        """
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
