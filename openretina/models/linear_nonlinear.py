from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from lightning import LightningModule

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import DummyCore
from openretina.modules.layers import regularizers
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.nonlinearities import parametrized_softplus
from openretina.modules.readout.multi_readout import MultiReadoutBase


class LNP(nn.Module):
    # Linear nonlinear Poisson
    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        smooth_weight: float = 0.0,
        sparse_weight: float = 0.0,
        smooth_regularizer: str = "LaplaceL2norm",
        laplace_padding=None,
        nonlinearity: str = "exp",
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
            bias=False,
            stride=1,
        )

        nn.init.xavier_normal_(self.inner_product_kernel.weight.data)

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )

        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)

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
        # Squeezing out the empty time dimension so we can use 2D regularizers
        return self._smooth_reg_fn(self.inner_product_kernel.weight.squeeze(2))

    def regularizer(self, **kwargs):
        return self.smooth_weight * self.laplace() + self.sparse_weight * self.weights_l1()

    def initialize(self, *args, **kwargs):
        pass


class MultipleLNP(BaseCoreReadout):
    def __init__(
        self, in_shape: tuple[int, int, int, int], n_neurons_dict: dict[str, int], learning_rate: float, **kwargs
    ):
        # The multiple LNP model acts like a readout reading directly from the frames of the videos
        readout = MultiReadoutBase(in_shape, n_neurons_dict, base_readout=LNP, **kwargs)  # type: ignore
        super().__init__(
            core=DummyCore(),
            readout=readout,
            learning_rate=learning_rate,
        )

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key: str | None = None) -> torch.Tensor:
        output_lnp = self.readout(x, data_key=data_key)

        return output_lnp


class SingleCellSeparatedLNP(LightningModule):
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
        fit_gaussian: bool = False,
        normalize_weights: bool = True,
        loss=None,
        correlation_loss=None,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d()
        self.correlation_loss = correlation_loss if correlation_loss is not None else CorrelationLoss3d(avg=True)
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
        self.n_neurons = 1
        self.nonlinearity = (
            parametrized_softplus() if nonlinearity == "parametrized_softplus" else F.__dict__[nonlinearity]
        )
        self.fit_gaussian = fit_gaussian
        self.space_conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=rank,
            kernel_size=(1, *self.kernel_size),  # Not using time
            bias=False,
            stride=1,
        )
        self.time_conv = nn.Conv3d(
            in_channels=rank, out_channels=1, kernel_size=(in_shape[1], 1, 1), bias=False, stride=1
        )

        nn.init.xavier_normal_(self.space_conv.weight.data)
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
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)

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
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_regularization_loss", regularization, logger=True)
        self.log("val_total_loss", total_loss, logger=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

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

    def laplace(self):
        return self.smooth_weight_spat * self._smooth_reg_fn_spat(
            self.space_conv.weight.squeeze(2)
        ) + self.smooth_weight_temp * self._smooth_reg_fn_temp(self.time_conv.weight.squeeze(-1, -2))

    def weights_l1(self, average: bool = True):
        """Returns l1 regularization across all weight dimensions

        Args:
            average (bool, optional): use mean of weights instead of sum. Defaults to True.
        """
        if average:
            return self.space_conv.weight.abs().mean() + self.time_conv.weight.abs().mean()
        else:
            return self.space_conv.weight.abs().sum() + self.time_conv.weight.abs().sum()

    def normalize_kernels(self):
        """Normalizes the kernels to have unit norm."""
        with torch.no_grad():
            self.space_conv.weight.data /= self.space_conv.weight.data.norm(keepdim=True)
            self.time_conv.weight.data /= self.time_conv.weight.data.norm(keepdim=True)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        if self.normalize_weights:
            self.normalize_kernels()

        out = self.space_conv(x)
        out = self.time_conv(out)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons")
        return out

    def regularizer(self):
        return self.laplace() + self.sparse_weight * self.weights_l1()
