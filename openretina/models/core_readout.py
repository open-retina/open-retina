import inspect
import logging
import os
from typing import Any, Iterable, Literal, Optional

import hydra.utils
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.core.base_core import Core, SimpleCoreWrapper
from openretina.modules.core.gru_core import ConvGRUCore
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.readout.multi_readout import (
    MultiGaussianReadoutWrapper,
    MultiSampledGaussianReadoutWrapper,
)
from openretina.utils.file_utils import get_cache_directory, get_local_file_path

LOGGER = logging.getLogger(__name__)

_GIN_MODEL_CHECKPOINTS_BASE_PATH = "https://gin.g-node.org/teulerlab/open-retina/raw/checkpoints/model_checkpoints"
_HUGGINGFACE_CHECKPOINTS_BASE_PATH = (
    "https://huggingface.co/datasets/open-retina/open-retina/tree/main/model_checkpoints"
)
_MODEL_NAME_TO_REMOTE_LOCATION = {
    "hoefling_2024_base_low_res": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/hoefling_2024_base_low_res.ckpt",
    "hoefling_2024_base_high_res": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/hoefling_2024_base_high_res.ckpt",
    "karamanlis_2024_gru": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/karamanlis_2024_GRU.ckpt",
    "karamanlis_2024_base": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/karamanlis_2024_base.ckpt",
    "maheswaranathan_2023_gru": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/maheswaranathan_2023_GRU.ckpt",
    "maheswaranathan_2023_base": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/24-01-2025/maheswaranathan_2023_base.ckpt",
}


class BaseCoreReadout(LightningModule):
    def __init__(
        self,
        core: Core,
        readout: nn.Module,
        learning_rate: float,
        loss: nn.Module | None = None,
        correlation_loss: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.core = core
        self.readout = readout
        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d()
        self.correlation_loss = correlation_loss if correlation_loss is not None else CorrelationLoss3d(avg=True)
        if data_info is None:
            data_info = {}
        self.data_info = data_info

    def on_train_epoch_end(self):
        # Compute the 2-norm for each layer at the end of the epoch
        core_norms = grad_norm(self.core, norm_type=2)
        self.log_dict(core_norms, on_step=False, on_epoch=True)
        readout_norms = grad_norm(self.readout, norm_type=2)
        self.log_dict(readout_norms, on_step=False, on_epoch=True)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key: str | None = None) -> torch.Tensor:
        output_core = self.core(x)
        output_readout = self.readout(output_core, data_key=data_key)
        return output_readout

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        regularization_loss_core = self.core.regularizer()
        regularization_loss_readout = self.readout.regularizer(session_id)  # type: ignore
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
        regularization_loss_readout = self.readout.regularizer(session_id)  # type: ignore
        total_loss = loss + regularization_loss_core + regularization_loss_readout
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)

        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_regularization_loss_core", regularization_loss_core, logger=True)
        self.log("val_regularization_loss_readout", regularization_loss_readout, logger=True)
        self.log("val_total_loss", total_loss, logger=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[str, DataPoint], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
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

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        """Save weight visualizations for core and readout modules.

        Args:
            folder_path: Base directory to save visualizations
            file_format: Image format for saved files
            state_suffix: Optional suffix for state identification
        """

        # Helper function to call save_weight_visualizations with dynamic parameter support
        def _call_save_viz(module: Any, subfolder: str) -> None:
            full_path = os.path.join(folder_path, subfolder)

            # Check if the method supports state_suffix parameter
            if "state_suffix" in inspect.signature(module.save_weight_visualizations).parameters:
                kwargs = {"state_suffix": state_suffix}
            else:
                kwargs = {}

            module.save_weight_visualizations(full_path, file_format, **kwargs)

        _call_save_viz(self.core, "weights_core")
        _call_save_viz(self.readout, "weights_readout")

    def compute_readout_input_shape(
        self, core_in_shape: tuple[int, int, int, int], core: Core
    ) -> tuple[int, int, int, int]:
        # Use the same device as the core's parameters to avoid potential errors at init.
        device = next(core.parameters()).device

        with torch.no_grad():
            stimulus = torch.zeros((1,) + tuple(core_in_shape), device=device)
            core_test_output = core.forward(stimulus)

        return core_test_output.shape[1:]  # type: ignore

    def stimulus_shape(self, time_steps: int, num_batches: int = 1) -> tuple[int, int, int, int, int]:
        channels, width, height = self.data_info["input_shape"]  # type: ignore
        return num_batches, channels, time_steps, width, height

    def update_model_data_info(self, data_info: dict[str, Any]) -> None:
        """To update relevant model attributes when loading a (trained) model and training it with new data only."""
        # update model.data_info and n_neurons_dict with the new data info
        for key in data_info.keys():
            if key == "input_shape":
                assert all(self.data_info[key][dim] == data_info[key][dim] for dim in range(len(data_info[key]))), (
                    f"Input shapes don't match: model has {self.data_info[key]}, new data has {data_info[key]}"
                )
            else:
                self.data_info[key].update(data_info[key])

        # update saved hyperparameters (so that the model can be loaded from checkpoint correctly)
        if hasattr(self, "hparams"):
            self.hparams["n_neurons_dict"] = self.data_info["n_neurons_dict"]
            self.hparams["data_info"] = self.data_info


class UnifiedCoreReadout(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        n_neurons_dict: dict[str, int],
        core: DictConfig,
        readout: DictConfig,
        learning_rate: float = 0.001,
        data_info: dict[str, Any] | None = None,
    ):
        core.channels = (in_shape[0], *hidden_channels)
        core_module = hydra.utils.instantiate(
            core,
        )

        # Build readout via Hydra partial with shape-dependent injection
        if "in_shape" not in readout:
            in_shape_readout = self.compute_readout_input_shape(in_shape, core_module)
            readout["in_shape"] = (in_shape_readout[0],) + in_shape_readout[1:]
        readout_module = hydra.utils.instantiate(
            readout,
            n_neurons_dict=n_neurons_dict,
        )

        # if calling save_hyperparameters after __init__ it leads to errors related to data_info["session_kwargs"]
        self.save_hyperparameters()
        super().__init__(core=core_module, readout=readout_module, learning_rate=learning_rate, data_info=data_info)


class CoreReadout(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        temporal_kernel_sizes: Iterable[int],
        spatial_kernel_sizes: Iterable[int],
        n_neurons_dict: dict[str, int],
        core_gamma_input: float = 0.0,
        core_gamma_hidden: float = 0.0,
        core_gamma_in_sparse: float = 0.0,
        core_gamma_temporal: float = 40.0,
        core_input_padding: bool | str | int | tuple[int, int, int] = False,
        core_hidden_padding: bool | str | int | tuple[int, int, int] = True,
        readout_scale: bool = True,
        readout_bias: bool = True,
        readout_gaussian_masks: bool = True,
        readout_gaussian_mean_scale: float = 6.0,
        readout_gaussian_var_scale: float = 4.0,
        readout_positive: bool = True,
        readout_gamma: float = 0.4,
        readout_gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
        learning_rate: float = 0.01,
        cut_first_n_frames_in_core: int = 30,
        dropout_rate: float = 0.0,
        maxpool_every_n_layers: Optional[int] = None,
        downsample_input_kernel_size: Optional[tuple[int, int, int]] = None,
        convolution_type: str = "custom_separable",
        color_squashing_weights: tuple[float, ...] | None = None,
        data_info: dict[str, Any] | None = None,
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
            color_squashing_weights=color_squashing_weights,
            hidden_padding=core_hidden_padding,
            convolution_type=convolution_type,
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

        super().__init__(core=core, readout=readout, learning_rate=learning_rate, data_info=data_info)
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
        batch_adaptation: bool = False,
        learning_rate: float = 0.01,
        core_gru_kwargs: Optional[dict] = None,
        convolution_type: str = "custom_separable",
        data_info: dict[str, Any] | None = None,
    ):
        core = ConvGRUCore(  # type: ignore
            n_neurons_dict=n_neurons_dict,
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
            batch_adaptation=batch_adaptation,
            use_avg_reg=False,
            nonlinearity="ELU",
            conv_type=convolution_type,
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

        super().__init__(core=core, readout=readout, learning_rate=learning_rate, data_info=data_info)
        self.save_hyperparameters()


class CoreGaussianReadout(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        temporal_kernel_sizes: Iterable[int],
        spatial_kernel_sizes: Iterable[int],
        n_neurons_dict: dict[str, int],
        core_gamma_input: float = 0.0,
        core_gamma_hidden: float = 0.0,
        core_gamma_in_sparse: float = 0.0,
        core_gamma_temporal: float = 40.0,
        core_input_padding: bool = False,
        core_hidden_padding: bool = False,
        readout_bias: bool = True,
        init_mu_range: float = 0.1,
        init_sigma_range: float = 0.1,
        readout_gamma: float = 0.4,
        readout_reg_avg: bool = False,
        batch_sample: bool = False,
        align_corners: bool = True,
        gauss_type: Literal["full", "iso"] = "full",
        grid_mean_predictor=None,
        shared_features=None,
        init_grid=None,
        shared_grid=None,
        mean_activity=None,
        learning_rate: float = 0.01,
        cut_first_n_frames_in_core: int = 0,
        dropout_rate: float = 0.0,
        maxpool_every_n_layers: Optional[int] = None,
        downsample_input_kernel_size: Optional[tuple[int, int, int]] = None,
        convolution_type: str = "full",
        data_info: dict[str, Any] | None = None,
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
            convolution_type=convolution_type,
        )

        in_shape_readout = self.compute_readout_input_shape(in_shape, core)
        readout = MultiSampledGaussianReadoutWrapper(
            in_shape=in_shape_readout,
            n_neurons_dict=n_neurons_dict,
            bias=readout_bias,
            init_mu_range=init_mu_range,
            init_sigma_range=init_sigma_range,
            gamma_readout=readout_gamma,
            batch_sample=batch_sample,
            align_corners=align_corners,
            gauss_type=gauss_type,
            grid_mean_predictor=grid_mean_predictor,
            shared_features=shared_features,
            shared_grid=shared_grid,
            init_grid=init_grid,
            mean_activity=mean_activity,
            readout_reg_avg=readout_reg_avg,
        )

        super().__init__(core=core, readout=readout, learning_rate=learning_rate, data_info=data_info)
        self.save_hyperparameters()


def load_core_readout_from_remote(
    model_name: str,
    device: str,
    cache_directory_path: str | os.PathLike | None = None,
) -> BaseCoreReadout:
    if cache_directory_path is None:
        cache_directory_path = get_cache_directory()
    if model_name not in _MODEL_NAME_TO_REMOTE_LOCATION:
        raise ValueError(
            f"Model name {model_name} not supported for download yet."
            f"The following names are supported: {sorted(_MODEL_NAME_TO_REMOTE_LOCATION.keys())}"
        )
    remote_path = _MODEL_NAME_TO_REMOTE_LOCATION[model_name]
    local_path = get_local_file_path(remote_path, cache_directory_path)

    try:
        model = UnifiedCoreReadout.load_from_checkpoint(local_path, map_location=device)
    except:  # noqa: E722
        LOGGER.warning("Could not load UnifiedCoreReadout, trying to load legacy models.")
        if "gru" in model_name.lower():
            model = GRUCoreReadout.load_from_checkpoint(local_path, map_location=device)
        else:
            model = CoreReadout.load_from_checkpoint(local_path, map_location=device)
    return model


def load_core_readout_model(
    model_path_or_name: str,
    device: str,
    is_gru_model: bool | None = None,  # for legacy models
    cache_directory_path: str | os.PathLike | None = None,
) -> BaseCoreReadout:
    if cache_directory_path is None:
        cache_directory_path = get_cache_directory()
    if model_path_or_name in _MODEL_NAME_TO_REMOTE_LOCATION:
        return load_core_readout_from_remote(model_path_or_name, device)

    local_path = get_local_file_path(model_path_or_name, cache_directory_path)
    try:
        model = UnifiedCoreReadout.load_from_checkpoint(local_path, map_location=device)
    except:  # noqa: E722
        LOGGER.warning("Could not load UnifiedCoreReadout, trying to load legacy models.")
        if is_gru_model:
            model = GRUCoreReadout.load_from_checkpoint(local_path, map_location=device)
        else:
            model = CoreReadout.load_from_checkpoint(local_path, map_location=device)
    return model
