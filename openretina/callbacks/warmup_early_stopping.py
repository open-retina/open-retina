from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping


class WarmupEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores the monitored metric during the first ``warmup_epochs`` epochs.

    ``trainer.min_epochs`` is not enough for this: EarlyStopping keeps counting during the minimum epochs, and a stop
    signal raised there stays set, so training stops as soon as the minimum is reached, even if the metric has improved
    since. Here no check runs during the warm-up, so the patience counts, and the best value starts, from the first
    epoch after it.
    """

    def __init__(self, *args: Any, warmup_epochs: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}.")
        self.warmup_epochs = warmup_epochs

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        return super()._should_skip_check(trainer) or trainer.current_epoch < self.warmup_epochs
