"""Exponential moving average (EMA) of model weights with decay warm-up."""

from collections.abc import Iterator
from itertools import chain
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch import nn
from torch.optim.swa_utils import AveragedModel


class _RampedAveragedModel(AveragedModel):
    def __init__(self, model: nn.Module, decay: float, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.decay = decay
        # Promote the shadow itself so that small updates survive between optimizer steps.
        for tensor in chain(self.module.parameters(), self.module.buffers()):
            if tensor.dtype in (torch.float16, torch.bfloat16):
                tensor.data = tensor.data.float()
            elif tensor.dtype == torch.complex32:
                tensor.data = tensor.data.to(torch.complex64)

    @torch.no_grad()
    def update_parameters(self, model: nn.Module) -> None:
        n = int(self.n_averaged.item())
        old_weight = min(self.decay, (1 + n) / (10 + n)) if n else 0.0
        new_weight = max(1.0 - self.decay, 9.0 / (10 + n))
        averages: Iterator[torch.Tensor] = self.module.parameters()
        current: Iterator[torch.Tensor] = model.parameters()
        if self.use_buffers:
            averages = chain(averages, self.module.buffers())
            current = chain(current, model.buffers())
        else:
            for average, value in zip(self.module.buffers(), model.buffers()):
                average.copy_(value)

        for average, value in zip(averages, current):
            if old_weight == 0.0 or not (average.is_floating_point() or average.is_complex()):
                average.copy_(value)
            else:
                # A weighted sum preserves infinite mask values and avoids overflow in the difference.
                average.mul_(old_weight).add_(value.to(average), alpha=new_weight)
        self.n_averaged.add_(1)


class RampedEMAWeightAveraging(EMAWeightAveraging):
    """Average model weights with a decay that increases toward the configured target.

    FP16 and BF16 weights accumulate in FP32. Integer and boolean buffers use their latest
    values. Configure only one EMA callback per Trainer because this shares Lightning's
    ``EMAWeightAveraging`` checkpoint key.

    Args:
        decay: target decay of the average.
        update_every_n_steps: update the average every this many optimizer steps.
        **kwargs: further arguments of ``EMAWeightAveraging``, including step and epoch scheduling.
    """

    def __init__(self, decay: float = 0.999, update_every_n_steps: int = 1, **kwargs: Any):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        super().__init__(decay=decay, update_every_n_steps=update_every_n_steps, **kwargs)
        self.decay = decay
        self._validation_backup: dict[torch.Tensor, torch.Tensor] = {}

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        # Lightning creates the averaged model before strategy setup configures the live model.
        if is_overridden("configure_model", pl_module):
            rank_zero_warn("Weight averaging configures the model before sharding and may require extra memory.")
            pl_module.configure_model()
        device = pl_module.device if self._device is None else self._device
        self._average_model = _RampedAveragedModel(
            pl_module, self.decay, device=device, use_buffers=self._use_buffers, **self._kwargs
        )

    @torch.no_grad()
    def _swap_models(self, pl_module: LightningModule) -> None:
        assert self._average_model is not None
        shadow = self._average_model.module
        for average, current in zip(
            chain(shadow.parameters(), shadow.buffers()), chain(pl_module.parameters(), pl_module.buffers())
        ):
            if average.dtype == current.dtype:
                tmp = average.clone()
                average.copy_(current)
                current.copy_(tmp)
            elif average in self._validation_backup:
                current.copy_(self._validation_backup.pop(average))
            else:
                # Swapping back through the live dtype would round away accumulated updates.
                self._validation_backup[average] = current.to(device=average.device, copy=True)
                current.copy_(average)

    @property
    def state_key(self) -> str:
        return EMAWeightAveraging.__qualname__


EMACallback = RampedEMAWeightAveraging
