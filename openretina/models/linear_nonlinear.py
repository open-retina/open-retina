from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from lightning import LightningModule

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.layers import regularizers
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.nonlinearities import ParametrizedSoftplus
from openretina.utils.model_utils import create_gaussian_kernel_cholesky


class SingleCellSeparatedLNP(LightningModule):
    """
    Single-cell, separable LNP model implemented as a PyTorch LightningModule.

    This model implements an LNP-style encoding model where the linear filter is
    constrained to be *space-time separable*.

    The special feature of this model is that it is "single-cell", meaning that it is only able
    to predict a single neuron's activity, in a single session, unlike other Core-Readout models.

    Spatial filtering is performed with a 3D convolution whose kernel size is
    (1, H, W), i.e. no temporal mixing in the first stage. Temporal filtering
    is performed with a second 3D convolution spanning the full input time axis
    (T, 1, 1). The separable rank controls the number of spatial and temporal components used.

    The module crops the input around a receptive-field
    location so that the spatial kernel operates on a local patch rather than
    the full image.

    Parameters
    ----------
    in_shape:
        Tuple (channels, time, height, width) describing the expected stimulus shape
        *excluding* batch dimension.
    rf_location:
        Optional (y, x) center location (in input pixel coordinates) used when
        cropping to a spatial patch of size `spat_kernel_size`. If None, defaults
        to the spatial center of the input.
    spat_kernel_size:
        (height, width) of the spatial kernel / crop window.
    learning_rate:
        Learning rate used by optimizer.
    rank:
        Separable rank specifies the number of spatial and temporal filter pairs that can be learned to predict.
        rank=1 corresponds to a single spatial filter and a single temporal filter.
        max rank can be  which corresponds to a full 3d convolution.
    smooth_weight_spat:
        Weight for spatial smoothness regularization (applied to `space_conv`).
    smooth_weight_temp:
        Weight for temporal smoothness regularization (applied to `time_conv`).
    sparse_weight:
        Weight for L1 sparsity penalty on both spatial and temporal kernels.
    smooth_regularizer_spat:
        Name of the spatial regularizer class in `regularizers.__dict__`.
        Examples in this code path include "LaplaceL2norm" or "GaussianLaplaceL2".
    smooth_regularizer_temp:
        Name of the temporal regularizer class in `regularizers.__dict__`.
    smooth_regularizer:
        Currently stored but not used directly in this implementation (kept for API
        compatibility / future use).
    laplace_padding:
        Passed through to regularizer constructors as `padding=...`. For GaussianLaplaceL2,
        a `kernel=...` argument is also supplied.
    fig_gaussian:
        if True, the spatial kernel learns the parameters of a single 2d gaussian instead of
        learning each pixel value
    nonlinearity:
        Output nonlinearity applied after the temporal stage. If "parametrized_softplus",
        uses `ParametrizedSoftplus()`. Otherwise, uses `torch.nn.functional.<nonlinearity>`
        via `F.__dict__[nonlinearity]` (e.g. "exp", "softplus", ...).
    normalize_weights:
        If True, renormalizes spatial and temporal kernels to unit norm at every
        forward pass (in-place, under no_grad).
    loss:
        Training loss. Defaults to `PoissonLoss3d()` if None.
    validation_loss:
        Validation metric/loss. Defaults to `CorrelationLoss3d(avg=True)` if None.
        During training/validation, correlation is logged as the negative of this loss.

    Input / Output shapes
    ---------------------
    Forward input `x`:
        Tensor of shape (batch, channels, time, height, width).
    Forward output:
        Tensor of shape (batch, time, neurons) where neurons=1.

    Notes
    -----
    *Cropping*: If `spat_kernel_size` does not match the full input spatial size,
    the module crops a patch centered at `rf_location` before applying convolutions.
    Cropping is boundary-safe (it clips at edges).

    *Regularization*:
        regularizer() = laplace() + sparse_weight * weights_l1()
    where laplace() applies the configured smoothness penalties to spatial and
    temporal kernels, and weights_l1() applies L1 to both kernels.

    Logged metrics
    --------------
    Training:
        - regularization_loss_core
        - train_total_loss
        - train_loss
        - train_correlation
    Validation:
        - val_loss
        - val_regularization_loss
        - val_total_loss
        - val_correlation
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        rf_location: Optional[Int[tuple, "y x"]] = None,
        spat_kernel_size: Int[tuple, "height width"] = (15, 15),
        learning_rate: float = 1e-3,
        rank: int = 1,
        smooth_weight_spat: float = 0.0,
        smooth_weight_temp: float = 0.0,
        sparse_weight: float = 0.0,
        smooth_regularizer_spat: str = "LaplaceL2norm",
        smooth_regularizer_temp: str = "Laplace1d",
        smooth_regularizer: str = "LaplaceL2norm",
        laplace_padding=None,
        nonlinearity: str = "exp",
        normalize_weights: bool = True,
        loss=None,
        validation_loss=None,
        fit_gaussian: bool = False,
        l11: Optional[float] = None,
        l22: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d()
        self.validation_loss = validation_loss if validation_loss is not None else CorrelationLoss3d(avg=True)
        self.smooth_weight_spat = smooth_weight_spat
        self.smooth_weight_temp = smooth_weight_temp
        self.sparse_weight = sparse_weight
        self.smooth_regularizer_spat = smooth_regularizer_spat
        self.smooth_regularizer_temp = smooth_regularizer_temp
        self.smooth_regularizer = smooth_regularizer
        self.normalize_weights = normalize_weights
        self.crop = (in_shape[-1] != spat_kernel_size[1]) or (in_shape[-2] != spat_kernel_size[0])
        # if location is not provided, use the center of the input
        if rf_location is None:
            rf_location = (in_shape[2] // 2, in_shape[3] // 2)
        self.location = rf_location

        self.fit_gaussian = fit_gaussian

        regularizer_config_spat = (
            dict(padding=laplace_padding, kernel=self.spat_kernel_size)
            if smooth_regularizer_spat == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        regularizer_config_temp = (
            dict(padding=laplace_padding, kernel=self.temp_kernel_size)
            if smooth_regularizer_temp == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )

        self._smooth_reg_fn_spat = regularizers.__dict__[smooth_regularizer_spat](**regularizer_config_spat)
        self._smooth_reg_fn_temp = regularizers.__dict__[smooth_regularizer_temp](**regularizer_config_temp)

        self.kernel_size = spat_kernel_size
        self.in_channels = in_shape[0]
        self.nonlinearity = (
            ParametrizedSoftplus() if nonlinearity == "parametrized_softplus" else F.__dict__[nonlinearity]
        )

        if not self.fit_gaussian:
            self.space_conv = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=rank,
                kernel_size=(1, *self.kernel_size),  # Not using time
                bias=False,
                stride=1,
            )
            nn.init.xavier_normal_(self.space_conv.weight.data)

        else:
            if rank != 1:
                raise ValueError("Currently cannot have a multi-rank model and fit a Gaussian to the spatial filter")
            if l11 is None or l22 is None:
                raise ValueError("Must provide sigma_x and sigma y when fit Gaussian to spatial filter")
            self.space_gauss_conv = GaussianConv3d(
                in_channels=self.in_channels, out_channels=rank, kernel_size=self.kernel_size, l11=l11, l22=l22
            )
        self.time_conv = nn.Conv3d(
            in_channels=rank, out_channels=1, kernel_size=(in_shape[1], 1, 1), bias=False, stride=1
        )

        nn.init.xavier_normal_(self.time_conv.weight.data)

    # TODO: Think about whether or not to leave this to the dataloader
    def crop_input(self, input_tensor: Float[torch.Tensor, "batch channels t h w"]):
        """Crops the input tensor to the size of the receptive field."""
        batch_size, channels, time, h, w = input_tensor.shape
        input_tensor = input_tensor[
            :,
            :,
            :,
            self.location[0] - min(self.kernel_size[0] // 2, self.location[0]) : self.location[0]
            + min(self.kernel_size[0] // 2 + self.kernel_size[0] % 2, h - self.location[0]),
            self.location[1] - min(self.kernel_size[1] // 2, self.location[1]) : self.location[1]
            + min(self.kernel_size[1] // 2 + self.kernel_size[1] % 2, w - self.location[1]),
        ]
        return input_tensor

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        if self.crop:
            images = self.crop_input(data_point.inputs)
        else:
            images = data_point.inputs
        model_output = self.forward(images, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        regularization = self.regularizer()
        total_loss = loss + regularization
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("regularization_loss_core", regularization, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        if self.crop:
            images = self.crop_input(data_point.inputs)
        else:
            images = data_point.inputs
        model_output = self.forward(images, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        regularization = self.regularizer()
        total_loss = loss + regularization
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_regularization_loss", regularization, logger=True)
        self.log("val_total_loss", total_loss, logger=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

    def laplace(self):
        if self.fit_gaussian:
            return self.smooth_weight_temp * self._smooth_reg_fn_temp(self.time_conv.weight.squeeze(-1, -2))
        return self.smooth_weight_spat * self._smooth_reg_fn_spat(
            self.space_conv.weight.squeeze(2)
        ) + self.smooth_weight_temp * self._smooth_reg_fn_temp(self.time_conv.weight.squeeze(-1, -2))

    def weights_l1(self, average: bool = True):
        """Returns l1 regularization across all weight dimensions

        Args:
            average (bool, optional): use mean of weights instead of sum. Defaults to True.
        """
        if average:
            if self.fit_gaussian:
                return self.time_conv.weight.abs().mean()
            return torch.abs(self.space_conv.weight).mean() + torch.abs(self.time_conv.weight).mean()
        else:
            if self.fit_gaussian:
                return self.time_conv.weight.abs().sum()
            return torch.abs(self.space_conv.weight).sum() + torch.abs(self.time_conv.weight).sum()

    def normalize_kernels(self):
        """Normalizes the kernels to have unit norm."""
        with torch.no_grad():
            self.space_conv.weight.data /= self.space_conv.weight.data.norm(keepdim=True)
            self.time_conv.weight.data /= self.time_conv.weight.data.norm(keepdim=True)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        if self.normalize_weights:
            self.normalize_kernels()
        if self.fit_gaussian:
            out = self.space_gauss_conv(x)
        else:
            out = self.space_conv(x)
        out = self.time_conv(out)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons", neurons=1)
        return out

    def regularizer(self):
        return self.laplace() + self.sparse_weight * self.weights_l1()


class GaussianConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, l11: float = 0.5, l22: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Learnable parameters for mean and sigma
        self.mean_x = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mean_y = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.l11_raw = nn.Parameter(torch.ones(1) * l11, requires_grad=True)
        self.l22_raw = nn.Parameter(torch.ones(1) * l22, requires_grad=True)
        self.l12 = nn.Parameter(torch.zeros(1), requires_grad=True)

        x_coord = torch.linspace(-kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[0])
        y_coord = torch.linspace(-kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[1])
        x, y = torch.meshgrid(x_coord, y_coord)

        self.register_buffer("x_coord", x_coord.clone())
        self.register_buffer("y_coord", y_coord.clone())
        self.register_buffer("x", x.clone())
        self.register_buffer("y", y.clone())

    def forward(self, x, data_key=None):
        kernel = create_gaussian_kernel_cholesky(
            self.l11_raw,
            self.l22_raw,
            self.l12,
            self.x,
            self.y,
            self.mean_x,
            self.mean_y,
            self.kernel_size,
            amplitude=1.0,
            lower=None,
            upper=None,
        )
        x = F.conv3d(x, kernel, padding=0)

        return x
