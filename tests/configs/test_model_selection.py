"""The model_selection config group switches early stopping, checkpointing and the LR schedule together."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).parents[2] / "configs")
CONFIGS = [
    "hoefling_2024_sc_low_res_lnp_grey",
    "hoefling_2024_core_readout_low_res_lnp_grey_separable",
    "hoefling_2024_core_readout_low_res_lnp_separable",
]
EXPECTED = {
    "val_loss": ("val_loss", "min", "poisson"),
    "val_correlation": ("val_evaluation_loss", "max", "correlation"),
}


@pytest.mark.parametrize("config_name", CONFIGS)
@pytest.mark.parametrize("selection", [None, "val_loss", "val_correlation"])
def test_selection_sets_every_monitor(config_name: str, selection: str | None) -> None:
    overrides = [] if selection is None else [f"model_selection={selection}"]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    monitor, mode, metric = EXPECTED[selection or "val_loss"]
    callbacks = cfg.training_callbacks
    assert (callbacks.early_stopping.monitor, callbacks.early_stopping.mode) == (monitor, mode)
    assert (callbacks.model_checkpoint.monitor, callbacks.model_checkpoint.mode) == (monitor, mode)
    assert callbacks.model_checkpoint.filename.startswith(f"{{epoch:02d}}_{{{monitor}:")
    assert (cfg.lr_scheduler.monitor, cfg.lr_scheduler.mode) == (monitor, mode)
    assert cfg.selection_metric == metric
    if "scheduler_metric" in cfg.model:
        assert cfg.model.scheduler_metric == metric
    # Both criteria wait out the same warm-up before stopping early or cutting the learning rate
    assert cfg.lr_scheduler._target_ == "openretina.utils.optimizer_utils.WarmupReduceLROnPlateau"
    assert cfg.lr_scheduler.warmup_epochs == cfg.trainer.min_epochs == cfg.selection_warmup_epochs == 30
    if "scheduler_warmup_epochs" in cfg.model:
        assert cfg.model.scheduler_warmup_epochs == 30


def test_colour_lnp_has_a_temporal_kernel_per_channel() -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="hoefling_2024_core_readout_low_res_lnp_separable")

    assert cfg.model.core.get("color_squashing_weights") is None
    assert cfg.model.core.cut_first_n_frames + cfg.model.readout.temporal_kernel_size - 1 == 30
    assert cfg.model.readout.temporal_kernel_per_channel
    assert cfg.model.readout.smooth_weight_temporal == 1e3
