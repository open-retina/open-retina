"""Greyscale LNP wiring: the readout channel count is derived from the core, not declared.

`configs/model/linear_nonlinear_poisson.yaml` leaves `readout.in_shape` missing, so
`UnifiedCoreReadout` probes the core to determine it. That makes colour squashing inside
`DummyCore` the mechanism that turns the LNP into a greyscale model - and it also means a
silent failure there would produce a colour model that merely looks greyscale.
"""

from pathlib import Path

import hydra
import lightning
import pytest
import torch
from omegaconf import OmegaConf

from openretina.models.core_readout import UnifiedCoreReadout
from openretina.modules.core.base_core import DummyCore
from openretina.modules.readout.linear_nonlinear_poison import LNPReadout

IN_SHAPE = (2, 150, 18, 16)  # channels time height width; the stimulus is green + UV
N_NEURONS_DICT = {"session_a": 5}
SPATIAL_WEIGHTS = 18 * 16


def build_model(color_squashing_weights: list[float] | None) -> UnifiedCoreReadout:
    core = OmegaConf.create(
        {
            "_target_": "openretina.modules.core.base_core.DummyCore",
            "_convert_": "object",
            "cut_first_n_frames": 30,
            "color_squashing_weights": color_squashing_weights,
        }
    )
    readout = OmegaConf.create(
        {
            "_target_": "openretina.modules.readout.multi_readout.MultipleLNPReadout",
            "_convert_": "object",
            "in_shape": "???",  # missing on purpose: derived by probing the core
            "nonlinearity": "softplus",
            "bias": True,
        }
    )
    return UnifiedCoreReadout(
        in_shape=IN_SHAPE,
        n_neurons_dict=N_NEURONS_DICT,
        core=core,
        readout=readout,
        data_info={"input_shape": (2, 18, 16), "n_neurons_dict": N_NEURONS_DICT},
    )


@pytest.mark.parametrize(
    ("color_squashing_weights", "expected_channels"),
    [(None, 2), ([0.5, 0.5], 1)],
)
def test_squashing_halves_the_readout_kernels(color_squashing_weights, expected_channels: int) -> None:
    model = build_model(color_squashing_weights)
    readout = model.readout["session_a"]
    # ModuleDict lookups are typed as plain Modules; narrow so the kernel is reachable.
    assert isinstance(readout, LNPReadout)

    assert readout.in_channels == expected_channels
    assert readout.inner_product_kernel.weight.shape == (5, expected_channels, 1, 18, 16)
    assert readout.inner_product_kernel.weight[0].numel() == expected_channels * SPATIAL_WEIGHTS


def test_squashed_model_sees_the_channel_mean() -> None:
    """Shape alone does not prove greyscale: check the squash is numerically a mean."""
    model = build_model([0.5, 0.5])
    green, uv = torch.rand(2, 1, *IN_SHAPE[1:]), torch.rand(2, 1, *IN_SHAPE[1:])
    mean = 0.5 * (green + uv)

    with torch.no_grad():
        from_colors = model(torch.cat([green, uv], dim=1), "session_a")
        from_mean = model(torch.cat([mean, mean], dim=1), "session_a")

    assert from_colors.shape == (2, IN_SHAPE[1] - 30, 5)
    assert torch.allclose(from_colors, from_mean, atol=1e-6)


def test_greyscale_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Reload from an actual file: `load_from_checkpoint` rebuilds the core from stored hparams.

    Going through a written checkpoint is the point - the squashing weights have to survive both
    the state dict and the pickled hparams, where `color_squashing_weights` is a `ListConfig`.
    Asserting on a live `state_dict()` alone leaves that reconstruction path untested.
    """
    model = build_model([0.5, 0.5])
    state_dict = model.state_dict()

    assert "core.color_squashing_layer.channel_weights" in state_dict
    assert list(model.hparams["core"]["color_squashing_weights"]) == [0.5, 0.5]

    checkpoint_path = tmp_path / "grey.ckpt"
    torch.save(
        {
            "state_dict": state_dict,
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint_path,
    )
    restored = UnifiedCoreReadout.load_from_checkpoint(checkpoint_path, map_location="cpu")

    assert isinstance(restored.core, DummyCore)
    assert restored.core.color_squashing_layer is not None
    assert restored.core.color_squashing_layer.channel_weights.tolist() == [0.5, 0.5]
    restored_readout = restored.readout["session_a"]
    assert isinstance(restored_readout, LNPReadout)
    assert restored_readout.in_channels == 1

    # A colour model must reject greyscale weights rather than quietly dropping them.
    with pytest.raises(RuntimeError):
        build_model(None).load_state_dict(state_dict, strict=True)


@pytest.mark.parametrize(
    ("config_name", "expected_channels"),
    [
        ("hoefling_2024_core_readout_low_res_lnp", 2),
        ("hoefling_2024_core_readout_low_res_lnp_grey", 1),
    ],
)
def test_shipped_config_builds_the_model_it_describes(config_name: str, expected_channels: int) -> None:
    """Compose the real config: the hand-built ones above cannot catch composition bugs.

    Both config bugs this feature actually hit were invisible to the tests here. A missing
    `optimizer`/`lr_scheduler` default group breaks the `${optimizer}` interpolation only once
    `configure_optimizers` runs, and `retina_pixel_size_um` at the wrong nesting was silently
    dropped instead of reaching `data_info` - which is how two published checkpoints ended up
    recording the high-res pixel size.
    """
    with hydra.initialize(config_path="../../configs", version_base="1.3"):
        cfg = hydra.compose(config_name=config_name)

    assert cfg.data_io.data_info.retina_pixel_size_um == 50  # 4x the high-res default of 12.5
    assert cfg.model.readout.smooth_weight == 30000.0  # the model-level default of 1.0 is inert

    cfg.model.n_neurons_dict = N_NEURONS_DICT
    model = UnifiedCoreReadout(data_info={"n_neurons_dict": N_NEURONS_DICT}, **cfg.model)

    readout = model.readout["session_a"]
    assert isinstance(readout, LNPReadout)
    assert readout.in_channels == expected_channels
    assert readout.inner_product_kernel.weight[0].numel() == expected_channels * SPATIAL_WEIGHTS
    assert readout.smooth_weight == 30000.0

    # Resolves `${optimizer}` and `${lr_scheduler}`, which the default groups have to supply.
    optimizer_config = model.configure_optimizers()
    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
    assert isinstance(optimizer_config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
