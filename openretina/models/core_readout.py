import logging
import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
from jaxtyping import Int
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.core.base_core import Core, SimpleCoreWrapper
from openretina.modules.core.gru_core import ConvGRUCore
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.readout.multi_readout import MultiGaussianReadoutWrapper

LOGGER = logging.getLogger(__name__)


class BaseCoreReadout(LightningModule):
    def __init__(
        self,
        core: Core,
        readout: nn.Module,
        learning_rate: float,
        loss: nn.Module | None = None,
        correlation_loss: nn.Module | None = None,
    ):
        super().__init__()

        self.core = core
        self.readout = readout
        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d()
        self.correlation_loss = correlation_loss if correlation_loss is not None else CorrelationLoss3d(avg=True)

    def on_train_epoch_end(self):
        # Compute the 2-norm for each layer at the end of the epoch
        core_norms = grad_norm(self.core, norm_type=2)
        self.log_dict(core_norms, on_step=False, on_epoch=True)
        readout_norms = grad_norm(self.readout, norm_type=2)
        self.log_dict(readout_norms, on_step=False, on_epoch=True)

    def forward(self, x: torch.Tensor, data_key: str) -> torch.Tensor:
        output_core = self.core(x)
        output_readout = self.readout(output_core, data_key=data_key)
        return output_readout

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        regularization_loss_core = self.core.regularizer()
        regularization_loss_readout = self.readout.regularizer(session_id)
        total_loss = loss + regularization_loss_core + regularization_loss_readout
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)

        self.log("regularization_loss_core", regularization_loss_core, on_step=False, on_epoch=True)
        self.log("regularization_loss_readout", regularization_loss_readout, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_correlation", correlation, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        regularization_loss_core = self.core.regularizer()
        regularization_loss_readout = self.readout.regularizer(session_id)
        total_loss = loss + regularization_loss_core + regularization_loss_readout
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_regularization_loss_core", regularization_loss_core, logger=True)
        self.log("val_regularization_loss_readout", regularization_loss_readout, logger=True)
        self.log("val_total_loss", total_loss, logger=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[str, DataPoint], batch_idx: int, dataloader_idx) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)
        self.log_dict(
            {
                "test_loss": loss,
                "test_correlation": correlation,
            }
        )

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

    def save_weight_visualizations(self, folder_path: str) -> None:
        self.core.save_weight_visualizations(os.path.join(folder_path, "weights_core"))
        self.readout.save_weight_visualizations(os.path.join(folder_path, "weights_readout"))

    def compute_readout_input_shape(
        self, core_in_shape: tuple[int, int, int, int], core: Core
    ) -> tuple[int, int, int, int]:
        # Use the same device as the core's parameters to avoid potential errors at init.
        device = next(core.parameters()).device

        with torch.no_grad():
            core_test_output = core.forward(torch.zeros((1,) + tuple(core_in_shape), device=device))

        return core_test_output.shape[1:]  # type: ignore

    @staticmethod
    def stimulus_shape(file_name: str) -> tuple[int, int, int, int]:
        """Function to infer the stimulus shape from the file name
        Note: this is a hack, in the future we can e.g. store the stimulus shape directly.
        """

        default_time_dim = 80
        if "hoefling" in file_name:
            if "high_res" in file_name:
                return 2, default_time_dim, 72, 64
            else:
                return 2, default_time_dim, 18, 16
        elif "maheswaranathan" in file_name:
            return 1, default_time_dim, 50, 50
        elif "karamanlis" in file_name:
            return 1, default_time_dim, 60, 80
        else:
            raise ValueError(f"Could not infer stimulus shape from {file_name=}")


class CoreReadout(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        temporal_kernel_sizes: Iterable[int],
        spatial_kernel_sizes: Iterable[int],
        n_neurons_dict: dict[str, int],
        core_gamma_input: float,
        core_gamma_hidden: float,
        core_gamma_in_sparse: float,
        core_gamma_temporal: float,
        core_input_padding: bool,
        core_hidden_padding: bool,
        readout_scale: bool,
        readout_bias: bool,
        readout_gaussian_masks: bool,
        readout_gaussian_mean_scale: float,
        readout_gaussian_var_scale: float,
        readout_positive: bool,
        readout_gamma: float,
        readout_gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
        learning_rate: float = 0.01,
        cut_first_n_frames_in_core: int = 30,
        dropout_rate: float = 0.0,
        maxpool_every_n_layers: Optional[int] = None,
        downsample_input_kernel_size: Optional[tuple[int, int, int]] = None,
    ):
        core = SimpleCoreWrapper(
            channels=(in_shape[0], *hidden_channels),
            temporal_kernel_sizes=tuple(temporal_kernel_sizes),
            spatial_kernel_sizes=tuple(spatial_kernel_sizes),
            gamma_input=core_gamma_input,
            gamma_temporal=core_gamma_temporal,
            gamma_in_sparse=core_gamma_in_sparse,
            gamma_hidden=core_gamma_hidden,
            cut_first_n_frames=cut_first_n_frames_in_core,
            dropout_rate=dropout_rate,
            maxpool_every_n_layers=maxpool_every_n_layers,
            downsample_input_kernel_size=downsample_input_kernel_size,
            input_padding=core_input_padding,
            hidden_padding=core_hidden_padding,
        )

        # Run one forward pass to determine output shape of core
        in_shape_readout = self.compute_readout_input_shape(in_shape, core)
        LOGGER.info(f"{in_shape_readout=}")

        readout = MultiGaussianReadoutWrapper(
            in_shape_readout,
            n_neurons_dict,
            readout_scale,
            readout_bias,
            readout_gaussian_masks,
            readout_gaussian_mean_scale,
            readout_gaussian_var_scale,
            readout_positive,
            readout_gamma,
            readout_gamma_masks,
            readout_reg_avg,
        )

        super().__init__(core=core, readout=readout, learning_rate=learning_rate)
        self.save_hyperparameters()


class GRUCoreReadout(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        temporal_kernel_sizes: Iterable[int],
        spatial_kernel_sizes: Iterable[int],
        n_neurons_dict: dict[str, int],
        core_gamma_hidden: float,
        core_gamma_input: float,
        core_gamma_in_sparse: float,
        core_gamma_temporal: float,
        core_bias: bool,
        core_input_padding: bool,
        core_hidden_padding: bool,
        core_use_gru: bool,
        core_use_projections: bool,
        readout_scale: bool,
        readout_bias: bool,
        readout_gaussian_masks: bool,
        readout_gaussian_mean_scale: float,
        readout_gaussian_var_scale: float,
        readout_positive: bool,
        readout_gamma: float,
        readout_gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
        learning_rate: float = 0.01,
        core_gru_kwargs: Optional[dict] = None,
    ):
        core = ConvGRUCore(  # type: ignore
            input_channels=in_shape[0],
            hidden_channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_sizes,
            spatial_kernel_size=spatial_kernel_sizes,
            layers=len(tuple(hidden_channels)),
            gamma_hidden=core_gamma_hidden,
            gamma_input=core_gamma_input,
            gamma_in_sparse=core_gamma_in_sparse,
            gamma_temporal=core_gamma_temporal,
            final_nonlinearity=True,
            bias=core_bias,
            input_padding=core_input_padding,
            hidden_padding=core_hidden_padding,
            batch_norm=True,
            batch_norm_scale=True,
            batch_norm_momentum=0.1,
            batch_adaptation=False,
            use_avg_reg=False,
            nonlinearity="ELU",
            conv_type="custom_separable",
            use_gru=core_use_gru,
            use_projections=core_use_projections,
            gru_kwargs=core_gru_kwargs,
        )

        # Run one forward pass to determine output shape of core
        in_shape_readout = self.compute_readout_input_shape(in_shape, core)
        LOGGER.info(f"{in_shape_readout=}")

        readout = MultiGaussianReadoutWrapper(
            in_shape_readout,
            n_neurons_dict,
            readout_scale,
            readout_bias,
            readout_gaussian_masks,
            readout_gaussian_mean_scale,
            readout_gaussian_var_scale,
            readout_positive,
            readout_gamma,
            readout_gamma_masks,
            readout_reg_avg,
        )

        super().__init__(core=core, readout=readout, learning_rate=learning_rate)
        self.save_hyperparameters()
