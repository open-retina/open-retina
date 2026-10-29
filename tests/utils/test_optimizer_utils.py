"""The warm-up ReduceLROnPlateau: no learning-rate cut, and no plateau bookkeeping, during the warm-up."""

import pytest
import torch
from omegaconf import OmegaConf

from openretina.utils.optimizer_utils import WarmupReduceLROnPlateau, instantiate_scheduler


def make_scheduler(warmup_epochs: int, patience: int = 2) -> tuple[torch.optim.Optimizer, WarmupReduceLROnPlateau]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = WarmupReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience, warmup_epochs=warmup_epochs
    )
    return optimizer, scheduler


def test_a_rising_loss_during_the_warm_up_does_not_cut_the_learning_rate() -> None:
    optimizer, scheduler = make_scheduler(warmup_epochs=5)
    for loss in [1.0, 2.0, 3.0, 4.0, 5.0]:
        scheduler.step(loss)
    assert optimizer.param_groups[0]["lr"] == 1.0


def test_the_plateau_test_starts_after_the_warm_up() -> None:
    """A low loss inside the warm-up must not become the best value the later losses are compared against."""
    optimizer, scheduler = make_scheduler(warmup_epochs=2, patience=2)
    for loss in [0.1, 0.1]:  # warm-up
        scheduler.step(loss)
    for loss in [1.0, 0.9, 0.8]:  # improving after it
        scheduler.step(loss)
    assert optimizer.param_groups[0]["lr"] == 1.0

    for loss in [0.8, 0.8, 0.8]:  # plateau
        scheduler.step(loss)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_no_warm_up_behaves_like_reduce_lr_on_plateau() -> None:
    optimizer, scheduler = make_scheduler(warmup_epochs=0)
    reference_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    reference = torch.optim.lr_scheduler.ReduceLROnPlateau(reference_optimizer, mode="min", factor=0.5, patience=2)
    for loss in [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]:
        scheduler.step(loss)
        reference.step(loss)
        assert optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]


def test_negative_warm_up_raises() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        make_scheduler(warmup_epochs=-1)


def test_instantiated_from_config_with_a_computed_min_lr() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    config = OmegaConf.create(
        {
            "_target_": "openretina.utils.optimizer_utils.WarmupReduceLROnPlateau",
            "mode": "min",
            "factor": 0.3,
            "patience": 5,
            "warmup_epochs": 30,
            "monitor": "val_loss",
        }
    )
    scheduler_dict = instantiate_scheduler(config, optimizer, 0.01)
    scheduler = scheduler_dict["scheduler"]

    assert isinstance(scheduler, WarmupReduceLROnPlateau)
    assert scheduler.warmup_epochs == 30
    assert scheduler_dict["monitor"] == "val_loss"
    assert scheduler.min_lrs == pytest.approx([0.01 * 0.3**3])
