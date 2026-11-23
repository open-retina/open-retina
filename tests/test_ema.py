import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging

from openretina.callbacks.ema import RampedEMAWeightAveraging, _RampedAveragedModel


@pytest.mark.parametrize("decay, expected_means", [(0.2, (9.0, 5.0)), (0.0, (11.0, 4.0))])
def test_ramped_decay_and_buffer_updates(decay, expected_means):
    model = torch.nn.BatchNorm1d(1)
    model.register_buffer("attention_mask", torch.tensor([-torch.inf]))
    average = _RampedAveragedModel(model, decay=decay, use_buffers=True)
    average.update_parameters(model)

    # At n=1 the ramp gives decay 2/11. At n=2 the configured cap applies.
    for value, expected in zip((11, 4), expected_means):
        model.running_mean.fill_(value)
        model.num_batches_tracked.fill_(value)
        average.update_parameters(model)
        assert average.module.running_mean.item() == pytest.approx(expected)
        assert average.module.num_batches_tracked.item() == value
        assert average.module.attention_mask.item() == -torch.inf


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_small_updates_survive_validation_in_fp32(dtype):
    model = LightningModule()
    model.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))
    trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    callback = RampedEMAWeightAveraging()
    callback.setup(trainer, model, "fit")
    average = callback._average_model
    average.n_averaged.fill_(9000)
    model.weight.fill_(1.25)

    for _ in range(4):
        average.update_parameters(model)
    expected = torch.tensor(1.25 - 0.25 * 0.999**4, dtype=torch.float32)
    torch.testing.assert_close(average.module.weight, expected)

    callback.on_validation_epoch_start(trainer, model)
    torch.testing.assert_close(model.weight, expected.to(dtype))
    callback.on_validation_epoch_end(trainer, model)
    assert model.weight.item() == 1.25
    torch.testing.assert_close(average.module.weight, expected)


def test_ramped_ema_uses_lightning_checkpoint_key():
    assert RampedEMAWeightAveraging().state_key == EMAWeightAveraging().state_key
