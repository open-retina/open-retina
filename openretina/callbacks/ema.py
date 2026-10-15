"""Exponential moving average (EMA) of model weights.

  Inspired by the weight EMA with decay 0.999 of the training recipe of DwiseNeuro, the Sensorium 2023 winning solution:
  https://github.com/lRomul/sensorium
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import nn


class EMACallback(Callback):
    """Keeps an exponential moving average of the model weights and uses it for validation, checkpoints and test.

    The EMA weights are swapped in at the start of every validation epoch during ``fit`` and left in place through
    validation, so ``val_evaluation_loss``, ``EarlyStopping`` and ``ModelCheckpoint`` all see the EMA model, and the
    saved checkpoints contain EMA weights (``trainer.test(ckpt_path="best")`` therefore tests EMA weights). The raw
    training weights are restored at the start of the next training epoch. After training, the EMA weights stay in
    the live module.

    All floating-point entries of the state dict are averaged, including BatchNorm running statistics.

    Args:
        decay: target decay of the average. The effective decay ramps up as ``min(decay, (1 + n) / (10 + n))`` with
            ``n`` the number of updates, so early random weights are forgotten quickly. With ``decay=0.999`` the
            ramp reaches the target after about 9,000 updates.
        update_every_n_steps: update the average every this many training batches.
    """

    def __init__(self, decay: float = 0.999, update_every_n_steps: int = 1):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every_n_steps < 1:
            raise ValueError(f"update_every_n_steps must be >= 1, got {update_every_n_steps}")
        self.decay = float(decay)
        self.update_every_n_steps = int(update_every_n_steps)
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._step = 0
        self._n_updates = 0

    @staticmethod
    def _ema_params(module: nn.Module) -> dict[str, torch.Tensor]:
        return {n: p for n, p in module.state_dict().items() if torch.is_floating_point(p)}

    def effective_decay(self) -> float:
        return min(self.decay, (1.0 + self._n_updates) / (10.0 + self._n_updates))

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._shadow:  # keep a shadow restored from a checkpoint
            self._shadow = {n: p.detach().clone() for n, p in self._ema_params(pl_module).items()}

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        self._step += 1
        if self._step % self.update_every_n_steps != 0:
            return
        self._n_updates += 1
        d = self.effective_decay()
        with torch.no_grad():
            for name, param in self._ema_params(pl_module).items():
                shadow = self._shadow.get(name)
                if shadow is None or shadow.shape != param.shape:
                    self._shadow[name] = param.detach().clone()
                else:
                    shadow.mul_(d).add_(param.detach(), alpha=1.0 - d)

    def _swap_in_ema(self, pl_module: LightningModule) -> None:
        if not self._shadow:
            return
        self._backup = {n: p.detach().clone() for n, p in self._ema_params(pl_module).items()}
        pl_module.load_state_dict(self._shadow, strict=False)

    def _restore_raw(self, pl_module: LightningModule) -> None:
        if self._backup:
            pl_module.load_state_dict(self._backup, strict=False)
            self._backup = {}

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Only swap during fit. A standalone trainer.validate()/test(ckpt_path="best") after fit has already loaded
        # the checkpointed (EMA) weights; swapping in the live shadow would score the final EMA state instead.
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._swap_in_ema(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._restore_raw(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._restore_raw(pl_module)
        self._swap_in_ema(pl_module)
        self._backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "step": self._step, "n_updates": self._n_updates, "shadow": self._shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self._step = state_dict.get("step", 0)
        self._n_updates = state_dict.get("n_updates", self._step // self.update_every_n_steps)
        self._shadow = state_dict.get("shadow", {})
