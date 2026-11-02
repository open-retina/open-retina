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


def _trainer(callbacks, **overrides) -> lightning.Trainer:
    kwargs = dict(
        max_epochs=3,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        accelerator="cpu",
    )
    kwargs.update(overrides)
    return lightning.Trainer(**kwargs)


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


def test_raw_weights_restored_after_mid_epoch_validation() -> None:
    torch.manual_seed(0)
    module = _TinyModule()
    ema = EMACallback(decay=0.999)

    class _Check(lightning.Callback):
        def __init__(self):
            self.previous_weights = None
            self.n_checks = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.previous_weights is not None:
                for name, value in pl_module.state_dict().items():
                    torch.testing.assert_close(value, self.previous_weights[name], rtol=0, atol=0)
                self.n_checks += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.previous_weights = {n: p.detach().clone() for n, p in pl_module.state_dict().items()}

        def on_validation_epoch_start(self, trainer, pl_module):
            torch.testing.assert_close(pl_module.layer.weight, ema._shadow["layer.weight"], rtol=0, atol=0)

    check = _Check()
    _trainer([ema, check], val_check_interval=0.5).fit(module, _loader(), _loader())
    assert check.n_checks == 23
    assert len(module.val_weights) == 6


@pytest.mark.parametrize("save_after_fit", [False, True])
def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, save_after_fit) -> None:
    class _MomentumModule(_TinyModule):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    torch.manual_seed(0)
    reference = _MomentumModule()
    reference_ema = EMACallback(decay=0.999)
    reference_trainer = _trainer([reference_ema])
    reference_trainer.fit(reference, _loader(), _loader())

    torch.manual_seed(0)
    module = _MomentumModule()
    ema = EMACallback(decay=0.999)
    checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_last=True)
    trainer = _trainer([ema, checkpoint], max_epochs=1, enable_checkpointing=True)
    trainer.fit(module, _loader(), _loader())
    if save_after_fit:
        checkpoint_path = str(tmp_path / "after_fit.ckpt")
        trainer.save_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint.last_model_path

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_ema = saved["callbacks"][ema.state_key]
    torch.testing.assert_close(saved["state_dict"]["layer.weight"], saved_ema["shadow"]["layer.weight"])
    assert not torch.equal(saved_ema["backup"]["layer.weight"], saved_ema["shadow"]["layer.weight"])

    # Standalone evaluation must keep the checkpoint's EMA weights, even when raw weights were also saved.
    trainer.validate(module, _loader(), ckpt_path=checkpoint_path, weights_only=False, verbose=False)
    torch.testing.assert_close(module.val_weights[-1], saved_ema["shadow"]["layer.weight"])

    resumed = _MomentumModule()
    resumed_ema = EMACallback(decay=0.999)
    resumed_trainer = _trainer([resumed_ema])
    resumed_trainer.fit(resumed, _loader(), _loader(), ckpt_path=checkpoint_path, weights_only=False)
    assert resumed_ema._n_updates == reference_ema._n_updates
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in reference_ema._backup.items():
        torch.testing.assert_close(resumed_ema._backup[name], expected, rtol=0, atol=0)
    reference_state = reference_trainer.optimizers[0].state_dict()["state"]
    resumed_state = resumed_trainer.optimizers[0].state_dict()["state"]
    for param_id, state in reference_state.items():
        torch.testing.assert_close(resumed_state[param_id]["momentum_buffer"], state["momentum_buffer"], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_restored_shadow_matches_model_device_and_dtype(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    module = _TinyModule().to(device=device)
    ema = EMACallback(decay=0.9)
    # Simulate CPU callback tensors restored from a checkpoint with a different model precision.
    shadow = {n: torch.zeros_like(p, device="cpu", dtype=torch.float64) for n, p in ema._ema_params(module).items()}
    ema.load_state_dict({"shadow": shadow, "n_updates": 10})
    ema.on_train_start(None, module)
    ema.on_train_batch_end(None, module, None, None, 0)

    for name, param in ema._ema_params(module).items():
        actual = ema._shadow[name]
        assert actual.device == param.device
        assert actual.dtype == param.dtype
        torch.testing.assert_close(actual, param * (1 - ema.effective_decay()))
