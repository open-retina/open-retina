"""
Colour LN model with one learnable Gaussian spatial filter per colour channel.

ColourGaussianLNCoreReadout is the colour counterpart of the LN model trained with SpatialContrastCoreReadout
(``learn_contrast_weight=False``, ``learn_filters=True``, ``filter_init="random"``): instead of squashing the stimulus
to greyscale, every neuron gets an independent Gaussian spatial filter and temporal filter for each channel (see
ColourGaussianLNReadout). The optimiser, learning-rate schedule and loss are those of SpatialContrastCoreReadout, so
that the two models differ only in their filters.

With ``color_squashing_weights`` the core squashes the stimulus to greyscale first, and the model is the greyscale LN
model itself. It then computes the same function as SpatialContrastCoreReadout with the contrast weight fixed at 0,
but projects the stimulus onto the spatial filters before filtering in time, which on a 72x64 stimulus is far cheaper
than filtering every pixel.
"""

import logging
from typing import Any, Callable, Literal, Sequence

import torch
import torch.nn as nn
from jaxtyping import Int
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer

from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import DummyCore
from openretina.modules.losses import PoissonLoss3d
from openretina.modules.readout.colour_gaussian_ln_readout import (
    ColourGaussianLNReadout,
    MultiColourGaussianLNReadout,
)
from openretina.utils.optimizer_utils import WarmupReduceLROnPlateau

LOGGER = logging.getLogger(__name__)


class ColourGaussianLNCoreReadout(BaseCoreReadout):
    """Multi-session colour LN model: a pass-through core and a MultiColourGaussianLNReadout.

    Args:
        in_shape: (channels, time, height, width) of the stimulus. Every channel gets its own filters.
        color_squashing_weights: If set, the core collapses the channels to greyscale with these weights, and the
            readout gets one set of filters per neuron (see the module docstring).
        n_neurons_dict: Number of neurons per session.
        temporal_filter_length: Number of frames of the temporal filters.
        cut_first_n_frames: Frames the core cuts. Set this or ``target_model_cut_frames``.
        target_model_cut_frames: Frames the whole model cuts, temporal filter included, as in
            SpatialContrastCoreReadout: the core cuts the frames the temporal filter does not consume.
        sigma_contour: Each spatial filter is masked to this many standard deviations of its current Gaussian.
        sigma_floor: Lower bound on the Gaussian widths, in pixels.
        gaussian_init_sigma: Initial width of every Gaussian, in pixels.
        smooth_weight_temporal: Temporal smoothness weight on the scale of the LNP readouts' weight (see
            SpatialContrastCoreReadout): the penalty is divided by the number of loss elements in each batch.
        init_offset_from_mean_activity: Initialise each neuron's offset c from its mean activity in
            ``data_info["mean_activity_dict"]``.
        scheduler_metric: Validation metric the learning rate schedule follows: "correlation"
            (val_evaluation_loss) or "poisson" (val_loss). Use the same metric for early stopping and checkpointing.
        scheduler_warmup_epochs: Epochs at the start during which the learning rate schedule ignores the metric.
        a_init, b_init, c_init: Initial nonlinearity parameters.
        learning_rate: Learning rate of the nonlinearity parameters.
        filter_learning_rate: Learning rate of the filter parameters. None uses learning_rate.
        loss: Training loss module. Defaults to the mean Poisson loss, as in SpatialContrastCoreReadout.
        evaluation_loss: Evaluation metric module.
        data_info: Data metadata dict.
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        temporal_filter_length: int = 31,
        cut_first_n_frames: int | None = None,
        target_model_cut_frames: int | None = None,
        color_squashing_weights: Sequence[float] | None = None,
        sigma_contour: float = 1.5,
        sigma_floor: float = 1.0,
        gaussian_init_sigma: float = 1.5,
        smooth_weight_temporal: float = 0.0,
        init_offset_from_mean_activity: bool = False,
        scheduler_metric: Literal["correlation", "poisson"] = "correlation",
        scheduler_warmup_epochs: int = 0,
        a_init: float = 4.0,
        b_init: float = 1.0,
        c_init: float = -5.0,
        learning_rate: float = 0.01,
        filter_learning_rate: float | None = None,
        loss: nn.Module | None = None,
        evaluation_loss: nn.Module | None = None,
        data_info: dict[str, Any] | None = None,
    ):
        stimulus_shape = tuple(in_shape)
        if target_model_cut_frames is not None and cut_first_n_frames is not None:
            raise ValueError("Set either cut_first_n_frames or target_model_cut_frames, not both.")
        if data_info is None:
            raise ValueError("data_info is required for ColourGaussianLNCoreReadout")
        if scheduler_metric not in ("correlation", "poisson"):
            raise ValueError(f"scheduler_metric must be 'correlation' or 'poisson', got {scheduler_metric!r}.")
        if color_squashing_weights is not None and len(color_squashing_weights) != stimulus_shape[0]:
            raise ValueError(
                f"Got {len(color_squashing_weights)} color_squashing_weights "
                f"for a {stimulus_shape[0]}-channel stimulus."
            )

        if target_model_cut_frames is not None:
            core_cut_frames = target_model_cut_frames - (temporal_filter_length - 1)
            if core_cut_frames < 0:
                raise ValueError(
                    f"{target_model_cut_frames=} is less than the {temporal_filter_length - 1} frames a "
                    f"{temporal_filter_length}-frame temporal filter consumes."
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
        readout_channels = stimulus_shape[0] if color_squashing_weights is None else 1
        readout = MultiColourGaussianLNReadout(
            in_shape=(readout_channels, stimulus_shape[1] - core_cut_frames, *stimulus_shape[2:]),
            n_neurons_dict=n_neurons_dict,
            temporal_filter_length=temporal_filter_length,
            sigma_contour=sigma_contour,
            sigma_floor=sigma_floor,
            gaussian_init_sigma=gaussian_init_sigma,
            smooth_weight_temporal=smooth_weight_temporal,
            a_init=a_init,
            b_init=b_init,
            c_init=c_init,
            mean_activity_dict=data_info.get("mean_activity_dict"),
            init_offset_from_mean_activity=init_offset_from_mean_activity,
        )

        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            loss=loss if loss is not None else PoissonLoss3d(avg=True),
            evaluation_loss=evaluation_loss,
            data_info=data_info,
        )
        self.filter_learning_rate = filter_learning_rate
        self.scheduler_metric = scheduler_metric
        self.scheduler_warmup_epochs = scheduler_warmup_epochs

    def readout_regularization(self, session_id: str, model_output: torch.Tensor) -> torch.Tensor:
        # As in SpatialContrastCoreReadout: the penalty is weighed against a loss summed over the batch
        return super().readout_regularization(session_id, model_output) / model_output.numel()

    def configure_optimizers(self):
        # Same optimiser and schedule as SpatialContrastCoreReadout with learnable filters
        filter_parameters = [
            p
            for readout in self.readout.values()
            if isinstance(readout, ColourGaussianLNReadout)
            for p in readout.filter_parameters()
        ]
        filter_ids = {id(p) for p in filter_parameters}
        other_parameters = [p for p in self.parameters() if id(p) not in filter_ids]
        filter_learning_rate = self.filter_learning_rate or self.learning_rate
        # The filter geometry gets no weight decay, which would pull the Gaussians towards fixed values
        parameter_groups = [
            {"params": other_parameters, "lr": self.learning_rate},
            {"params": filter_parameters, "lr": filter_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(parameter_groups)

        lr_decay_factor = 0.3
        if self.scheduler_metric == "poisson":
            monitor, mode, threshold, threshold_mode = "val_loss", "min", 1e-4, "rel"
        else:
            monitor, mode, threshold, threshold_mode = "val_evaluation_loss", "max", 0.0005, "abs"
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=lr_decay_factor,
            patience=5,
            threshold=threshold,
            threshold_mode=threshold_mode,
            min_lr=[group["lr"] * lr_decay_factor**3 for group in parameter_groups],
            warmup_epochs=self.scheduler_warmup_epochs,
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
        """After each step, keep nl_a >= 0 and the Gaussian centres on the stimulus."""
        optimizer.step(closure=optimizer_closure)

        for readout in self.readout.values():
            assert isinstance(readout, ColourGaussianLNReadout)
            readout.apply_constraints()
