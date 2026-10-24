"""WarmupEarlyStopping: a metric that gets worse during the warm-up must not stop training once it is over."""

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel

from openretina.callbacks.warmup_early_stopping import WarmupEarlyStopping

# Rises for 10 epochs (a transient), then falls for 15, then stays flat
SCRIPTED_METRIC = [float(epoch) for epoch in range(10)] + [10.0 - 0.5 * i for i in range(15)] + [3.0] * 30


class ScriptedMetricModel(BoringModel):
    def validation_step(self, batch, batch_idx):
        self.log("scripted", SCRIPTED_METRIC[self.current_epoch])
        return super().validation_step(batch, batch_idx)


def fit(callback: pl.Callback, **trainer_kwargs) -> pl.Trainer:
    torch.manual_seed(0)
    trainer = pl.Trainer(
        max_epochs=len(SCRIPTED_METRIC),
        callbacks=[callback],
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        **trainer_kwargs,
    )
    trainer.fit(ScriptedMetricModel())
    return trainer


def test_min_epochs_alone_stops_right_after_the_transient() -> None:
    """The failure WarmupEarlyStopping exists for: the stop signal raised during min_epochs fires when they end."""
    trainer = fit(pl.callbacks.EarlyStopping(monitor="scripted", mode="min", patience=5), min_epochs=10)
    assert trainer.current_epoch == 10


def test_warm_up_waits_for_the_improvement_and_stops_on_the_plateau() -> None:
    callback = WarmupEarlyStopping(monitor="scripted", mode="min", patience=5, warmup_epochs=10)
    trainer = fit(callback)

    # The metric improves until epoch 24 and is flat after, so patience 5 stops it after epoch 29
    assert callback.stopped_epoch == 29
    assert callback.best_score.item() == 3.0
    assert trainer.current_epoch == 30


def test_negative_warm_up_raises() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        WarmupEarlyStopping(monitor="scripted", warmup_epochs=-1)
