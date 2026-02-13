import logging
from typing import Any, Iterable

import hydra.utils
import torch
import torch.nn as nn
from jaxtyping import Int
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from openretina.data_io.base_dataloader import DataPoint
from openretina.modules.core.base_core import Core
from openretina.modules.losses import FutureFrameMSELoss
from openretina.utils.optimizer_utils import instantiate_optimizer, instantiate_scheduler

LOGGER = logging.getLogger(__name__)


def _set_mapping_value(mapping: DictConfig | dict[str, Any], key: str, value: Any) -> None:
    mapping[key] = value


def _is_missing_or_absent(mapping: DictConfig | dict[str, Any], key: str) -> bool:
    if isinstance(mapping, DictConfig):
        return key not in mapping or OmegaConf.is_missing(mapping, key)
    return key not in mapping or mapping[key] in {"???", None}


def _is_hydra_config(obj: Any) -> bool:
    return isinstance(obj, DictConfig) or (isinstance(obj, dict) and "_target_" in obj)


class UnifiedFuturePredictor(LightningModule):
    """
    Core + lightweight decoder model for self-supervised future-frame prediction.
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        core: DictConfig,
        decoder: DictConfig,
        hidden_channels: tuple[int, ...] | Iterable[int] | None = None,
        prediction_horizon: int = 1,
        reverse_video_direction: bool = False,
        learning_rate: float = 1e-3,
        loss: nn.Module | DictConfig | None = None,
        data_info: dict[str, Any] | None = None,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
    ) -> None:
        super().__init__()
        in_shape = tuple(in_shape)
        if prediction_horizon < 0:
            raise ValueError(f"{prediction_horizon=} must be >= 0")

        if hidden_channels is not None:
            hidden_channels = tuple(hidden_channels)
            _set_mapping_value(core, "channels", (in_shape[0], *hidden_channels))

        core_module = hydra.utils.instantiate(core, n_neurons_dict={})
        core_out_shape = self._compute_core_output_shape(in_shape, core_module)
        if _is_missing_or_absent(decoder, "in_channels"):
            decoder["in_channels"] = core_out_shape[0]
        if _is_missing_or_absent(decoder, "out_channels"):
            decoder["out_channels"] = in_shape[0]
        decoder_module = hydra.utils.instantiate(decoder)

        if loss is not None and _is_hydra_config(loss):
            loss_module = hydra.utils.instantiate(loss)
        elif loss is None:
            loss_module = FutureFrameMSELoss(avg=True)
        else:
            loss_module = loss

        self.core = core_module
        self.decoder = decoder_module
        self.loss = loss_module
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.reverse_video_direction = reverse_video_direction
        self.data_info = {} if data_info is None else data_info
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        self.save_hyperparameters(logger=False)

    @staticmethod
    def _compute_core_output_shape(
        core_in_shape: tuple[int, int, int, int],
        core: Core,
    ) -> tuple[int, int, int, int]:
        try:
            device = next(core.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        with torch.no_grad():
            stimulus = torch.zeros((1,) + tuple(core_in_shape), device=device)
            core_test_output = core.forward(stimulus)
        return core_test_output.shape[1:]  # type: ignore

    def _prepare_stimulus(self, stimulus: torch.Tensor) -> torch.Tensor:
        if self.reverse_video_direction:
            return torch.flip(stimulus, dims=(2,))
        return stimulus

    def _future_target(self, stimulus: torch.Tensor) -> torch.Tensor:
        if self.prediction_horizon >= stimulus.size(2):
            raise ValueError(
                "prediction_horizon "
                f"({self.prediction_horizon}) must be smaller than chunk length ({stimulus.size(2)})."
            )
        if self.prediction_horizon == 0:
            return stimulus
        return stimulus[:, :, self.prediction_horizon :, ...]

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        hidden = self.core(stimulus)
        reconstructed = self.decoder(hidden)
        return reconstructed

    def _shared_step(self, batch: tuple[str, DataPoint], stage: str) -> torch.Tensor:
        _, data_point = batch
        stimulus = self._prepare_stimulus(data_point.inputs)
        model_output = self.forward(stimulus)

        t_delta = stimulus.size(2) - model_output.size(2)
        target_start = t_delta + self.prediction_horizon
        if target_start >= stimulus.size(2):
            raise ValueError(f"{target_start=}, {stimulus.size(2)=}")
        target = stimulus[:, :, target_start:]
        model_output = model_output[:, :, : target.size()]

        prediction_loss = self.loss.forward(model_output, target)
        regularization_loss_core = self.core.regularizer()
        regularization_loss_decoder = self.decoder.regularizer() if hasattr(self.decoder, "regularizer") else 0.0
        total_loss = prediction_loss + regularization_loss_core + regularization_loss_decoder

        if self._trainer is not None:
            on_step = False
            on_epoch = True
            self.log(f"{stage}_loss", prediction_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
            self.log(
                f"{stage}_regularization_loss_core",
                regularization_loss_core,
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                f"{stage}_regularization_loss_decoder",
                regularization_loss_decoder,
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(f"{stage}_total_loss", total_loss, on_step=on_step, on_epoch=on_epoch, prog_bar=stage != "train")
        return total_loss

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: tuple[str, DataPoint], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        del batch_idx, dataloader_idx
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = instantiate_optimizer(
            self.optimizer_config,
            self.parameters(),
            self.learning_rate,
        )
        scheduler_dict = instantiate_scheduler(
            self.lr_scheduler_config,
            optimizer,
            self.learning_rate,
            trainer=getattr(self, "trainer", None),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
