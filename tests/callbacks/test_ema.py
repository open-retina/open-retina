import lightning
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openretina.callbacks.ema import EMACallback


class _TinyModule(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 1)
        self.bn = nn.BatchNorm1d(3)
        self.val_weights: list[torch.Tensor] = []

    def forward(self, x):
        return self.layer(self.bn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_weights.append(self.layer.weight.detach().clone())
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 3, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]])
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _trainer(callbacks) -> lightning.Trainer:
    return lightning.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        accelerator="cpu",
    )


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        EMACallback(decay=1.0)
    with pytest.raises(ValueError):
        EMACallback(update_every_n_steps=0)


def test_effective_decay_ramp() -> None:
    ema = EMACallback(decay=0.999)
    assert ema.effective_decay() == pytest.approx(0.1)
    ema._n_updates = 90
    assert ema.effective_decay() == pytest.approx(0.91)
    ema._n_updates = 100_000
    assert ema.effective_decay() == pytest.approx(0.999)


def test_ema_weights_used_for_validation_and_kept_after_fit() -> None:
    torch.manual_seed(0)
    module = _TinyModule()
    ema = EMACallback(decay=0.9)
    _trainer([ema]).fit(module, train_dataloaders=_loader(), val_dataloaders=_loader())

    shadow_weight = ema._shadow["layer.weight"]
    # validation of the last epoch saw the EMA weights, and they are left in the module after fit
    assert torch.allclose(module.val_weights[-1], shadow_weight)
    assert torch.allclose(module.layer.weight, shadow_weight)
    # BatchNorm running statistics are averaged as well
    assert "bn.running_mean" in ema._shadow
    # 3 epochs x 8 batches
    assert ema._n_updates == 24

    restored = EMACallback()
    restored.load_state_dict(ema.state_dict())
    assert restored._n_updates == 24
    assert torch.equal(restored._shadow["layer.weight"], shadow_weight)


def test_raw_weights_restored_for_training() -> None:
    torch.manual_seed(0)
    module = _TinyModule()
    ema = EMACallback(decay=0.999)

    class _Check(lightning.Callback):
        def __init__(self):
            self.diffs: list[float] = []

        def on_train_epoch_start(self, trainer, pl_module):
            # runs after EMACallback.on_train_epoch_start (callback order), so the raw weights must be back
            if trainer.current_epoch > 0:
                assert ema._backup == {}
                self.diffs.append(float((pl_module.layer.weight.detach() - ema._shadow["layer.weight"]).abs().max()))

    check = _Check()
    _trainer([ema, check]).fit(module, train_dataloaders=_loader(), val_dataloaders=_loader())
    # with a slow average the raw training weights differ from the EMA weights
    assert len(check.diffs) == 2 and all(d > 0 for d in check.diffs)
