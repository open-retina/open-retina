import inspect
import logging
import os
import warnings
from typing import Any, Iterable, Optional

import hydra.utils
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.core.base_core import Core, SimpleCoreWrapper
from openretina.modules.losses import CorrelationLoss3d, PoissonLoss3d
from openretina.modules.readout.multi_readout import MultiGaussianMaskReadout, MultiReadoutBase
from openretina.utils.file_utils import get_cache_directory, get_local_file_path

LOGGER = logging.getLogger(__name__)

_GIN_MODEL_CHECKPOINTS_BASE_PATH = "https://gin.g-node.org/teulerlab/open-retina/raw/checkpoints/model_checkpoints"
_HUGGINGFACE_CHECKPOINTS_BASE_PATH = (
    "https://huggingface.co/datasets/open-retina/open-retina/tree/main/model_checkpoints"
)
_MODEL_NAME_TO_REMOTE_LOCATION = {
    "hoefling_2024_base_low_res": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/27-11-2025/hoefling_2024_base_low_res.ckpt",
    "hoefling_2024_base_low_res_grey_scale": (
        f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/27-11-2025/hoefling_2024_base_low_res_grey_scale.ckpt"
    ),
    "hoefling_2024_base_high_res": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/27-11-2025/hoefling_2024_base_high_res.ckpt",
    "karamanlis_2024_base_mouse": f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/27-11-2025/karamanlis_2024_base_mouse.ckpt",
    "karamanlis_2024_base_marmoset": (
        f"{_HUGGINGFACE_CHECKPOINTS_BASE_PATH}/27-11-2025/karamanlis_2024_base_marmoset.ckpt"
    ),
    # "maheswaranathan_2023_base": f"",  # Todo: update
}


class BaseCoreReadout(LightningModule):
    """
    Base module for models combining a shared core and a multi-session readout.
    All models following the Core Readout pattern should inherit from this class.

    This LightningModule encapsulates a model made of a shared core and a flexible multi-session readout,
    suitable for training across-session architectures. It defines training, validation, and testing steps,
    provides hooks for optimizer and scheduler configuration, and methods for handling data info and visualization.
    """

    def __init__(
        self,
        core: Core,
        readout: MultiReadoutBase,
        learning_rate: float,
        loss: nn.Module | None = None,
        validation_loss: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
    ):
        """
        Initializes a BaseCoreReadout module.

        Args:
            core (Core): The shared feature extraction core network.
            readout (MultiReadoutBase): The multi-session readout module mapping core features to neuron outputs
                per session.
            learning_rate (float): Learning rate for network training.
            loss (nn.Module, optional): Loss function for training. Defaults to PoissonLoss3d if None.
            validation_loss (nn.Module, optional): Loss used to compute correlation performance metric.
                Defaults to CorrelationLoss3d (avg=True) if None.
            data_info (dict[str, Any], optional): Dictionary containing data-specific metadata, such as input_shape,
                session neuron counts, etc. If None, defaults to empty dict.
        """
        super().__init__()

        self.core = core
        self.readout = readout
        self.learning_rate = learning_rate
        self.loss = loss if loss is not None else PoissonLoss3d()
        self.validation_loss = validation_loss if validation_loss is not None else CorrelationLoss3d(avg=True)
        if data_info is None:
            data_info = {}
        self.data_info = data_info

        # Finally, save hyperparameters without logging them to the logger objects for now
        self.save_hyperparameters(logger=False)

    def on_fit_start(self):
        for lg in self.trainer.loggers:
            lg.log_hyperparams({k: v for k, v in self.hparams.items() if k not in {"data_info", "n_neurons_dict"}})

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
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

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
        correlation = -self.validation_loss.forward(model_output, data_point.targets)

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
        avg_correlation = -self.validation_loss.forward(model_output, data_point.targets)
        per_neuron_correlation = self.validation_loss._per_neuron_correlations

        # Add metric and performances to data_info for downstream tasks
        if "pretrained_performance_metric" not in self.data_info:
            self.data_info["pretrained_performance_metric"] = "test " + type(self.validation_loss).__name__

        if "pretrained_performance" not in self.data_info:
            self.data_info["pretrained_performance"] = {}

        # Also add cut frames if not present
        if "model_cut_frames" not in self.data_info:
            self.data_info["model_cut_frames"] = data_point.targets.size(1) - model_output.size(1)

        self.data_info["pretrained_performance"][session_id] = per_neuron_correlation

        self.log_dict(
            {
                "test_loss": loss,
                "test_correlation": avg_correlation,
            }
        )

        return loss

    def on_test_end(self):
        # Update internal lightning hyperparameters to save the updated data_info after testing.
        self.hparams["data_info"] = self.data_info

        # Save using checkpointer callback (should always exist with our default configs)
        if self.trainer and self.trainer.checkpoint_callback:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                final_path = best_model_path.replace(".ckpt", "_final.ckpt")
                self.trainer.save_checkpoint(final_path)

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
        """
        Save weight visualizations for core and readout modules.

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
        try:
            device = next(core.parameters()).device
        except StopIteration:
            # No parameters (e.g., when using DummyCore), assume core can be run on cpu
            device = torch.device("cpu")

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

    @property
    def pretrained_cfg(self) -> dict[str, Any]:
        """Alias for data_info, following `timm` (Pytorch Image Models) conventions."""
        return self.data_info


class UnifiedCoreReadout(BaseCoreReadout):
    """
    A flexible core-readout model for multi-session neural data, designed for Hydra config workflows.

    This class is the recommended entry point for defining core-readout models via config files using Hydra.
    It allows unified instantiation of arbitrary core and readout modules, specified via DictConfig,
    enabling rapid experimentation and extensibility. Supports all multi-session settings, custom core/readout
    combinations, and integration with configuration-driven pipelines (including hyperparameter optimization).

    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        core: DictConfig,
        readout: DictConfig,
        hidden_channels: tuple[int, ...] | Iterable[int] | None = None,
        learning_rate: float = 0.001,
        data_info: dict[str, Any] | None = None,
    ):
        """
        Initializes a UnifiedCoreReadout for multi-session configurable neural modeling via Hydra configs.

        Args:
            in_shape (tuple[int, int, int, int]):
                Input shape as (channels, time, height, width) for the core module.
            hidden_channels (Iterable[int]):
                List of hidden channels for the core; used in core config initialization.
            n_neurons_dict (dict[str, int]):
                Mapping from session/dataset identifier to neuron count for each session.
            core (DictConfig):
                Hydra config for instantiating the core module (should specify class and params).
            readout (DictConfig):
                Hydra config for the readout module (specifies type and custom session-aware params).
            learning_rate (float, optional):
                Learning rate for model training. Defaults to 0.001.
            data_info (dict[str, Any], optional):
                Additional metadata dictionary, e.g., with input shape and neuron mapping.
        """
        # Make sure in_shape and hidden_channels are a tuple
        # (with hydra configs they can be a `omegaconf.listconfig.ListConfig`).
        # This lead to an error when logging hyperparameters with the csv logger during training.
        in_shape = tuple(in_shape)
        if hidden_channels is not None:
            hidden_channels = tuple(hidden_channels)
            core.channels = (in_shape[0], *hidden_channels)

        core_module = hydra.utils.instantiate(
            core,
            n_neurons_dict=n_neurons_dict,
        )

        # determine input_shape of readout if it is not already present
        if "in_shape" not in readout:
            in_shape_readout = self.compute_readout_input_shape(in_shape, core_module)
            readout["in_shape"] = (in_shape_readout[0],) + tuple(in_shape_readout[1:])

        # Extract mean_activity_dict from data_info if available
        mean_activity_dict = None if data_info is None else data_info.get("mean_activity_dict")

        readout_module = hydra.utils.instantiate(
            readout,
            n_neurons_dict=n_neurons_dict,
            mean_activity_dict=mean_activity_dict,
        )

        super().__init__(core=core_module, readout=readout_module, learning_rate=learning_rate, data_info=data_info)


class ExampleCoreReadout(BaseCoreReadout):
    """
    Example implementation of a custom Core-Readout model, using a convolutional core and a Gaussian readout.

    This class serves as a guide for constructing custom Core-Readout models without using the unified Hydra
    configuration system and the `UnifiedCoreReadout` class. Use this model as a reference if you wish to instantiate
    or design core/readout units directly in code rather than through configuration files. For most workflows,
    especially those using Hydra, `UnifiedCoreReadout` is preferred for maximum flexibility.

    N.B., this class is provided as a reference example.
    """

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
        warnings.warn(
            "You are using ExampleCoreReadout, which is intended as a reference/example class for custom "
            "core-readout model implementations. For most configuration-driven workflows, especially if you "
            "use Hydra, consider using UnifiedCoreReadout instead, or writing your own class that inherits "
            "from BaseCoreReadout.",
            UserWarning,
            stacklevel=2,
        )
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

        readout = MultiGaussianMaskReadout(
            in_shape_readout,
            n_neurons_dict,
            readout_scale,
            readout_bias,
            readout_gaussian_mean_scale,
            readout_gaussian_var_scale,
            readout_positive,
            readout_gamma,
            readout_gamma_masks,
            readout_reg_avg,
        )

        super().__init__(core=core, readout=readout, learning_rate=learning_rate, data_info=data_info)

    def on_load_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint["state_dict"]

        readout_bias_keys = [k for k in state_dict.keys() if k.startswith("readout.") and k.endswith(".bias_param")]
        for key in readout_bias_keys:
            new_key = key.removesuffix(".bias_param") + ".bias"
            state_dict[new_key] = state_dict.pop(key)
        if len(readout_bias_keys) > 0:
            LOGGER.warning(f"Renamed the following readout bias keys: {readout_bias_keys}")


def load_core_readout_from_remote(
    model_name: str,
    device: str,
    cache_directory_path: str | os.PathLike | None = None,
) -> BaseCoreReadout:
    if cache_directory_path is None:
        cache_directory_path = get_cache_directory()
    if model_name not in _MODEL_NAME_TO_REMOTE_LOCATION:
        raise ValueError(
            f"Model name {model_name} not supported for download yet. "
            f"The following names are supported: {sorted(_MODEL_NAME_TO_REMOTE_LOCATION.keys())}"
        )
    remote_path = _MODEL_NAME_TO_REMOTE_LOCATION[model_name]
    local_path = get_local_file_path(remote_path, cache_directory_path)
    try:
        return UnifiedCoreReadout.load_from_checkpoint(local_path, map_location=device)
    except:  # noqa: E722
        # Support for legacy ExampleCoreReadout model
        return ExampleCoreReadout.load_from_checkpoint(local_path, map_location=device)


def load_core_readout_model(
    model_path_or_name: str,
    device: str,
    cache_directory_path: str | os.PathLike | None = None,
) -> BaseCoreReadout:
    if cache_directory_path is None:
        cache_directory_path = get_cache_directory()
    if model_path_or_name in _MODEL_NAME_TO_REMOTE_LOCATION:
        return load_core_readout_from_remote(model_path_or_name, device)

    local_path = get_local_file_path(model_path_or_name, cache_directory_path)
    try:
        return UnifiedCoreReadout.load_from_checkpoint(local_path, map_location=device)
    except:  # noqa: E722
        # Support for legacy ExampleCoreReadout model
        return ExampleCoreReadout.load_from_checkpoint(local_path, map_location=device)
