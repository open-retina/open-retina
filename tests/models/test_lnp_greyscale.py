"""Greyscale LNP wiring: the readout channel count is derived from the core, not declared.

`configs/model/linear_nonlinear_poisson.yaml` leaves `readout.in_shape` missing, so
`UnifiedCoreReadout` probes the core to determine it. That makes colour squashing inside
`DummyCore` the mechanism that turns the LNP into a greyscale model - and it also means a
silent failure there would produce a colour model that merely looks greyscale.
"""

import pytest
import torch
from omegaconf import OmegaConf

from openretina.models.core_readout import UnifiedCoreReadout
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


def test_greyscale_checkpoint_roundtrip() -> None:
    """`load_from_checkpoint` rebuilds the core from hparams, so the weights must be stored there."""
    model = build_model([0.5, 0.5])
    state_dict = model.state_dict()

    assert "core.color_squashing_layer.channel_weights" in state_dict
    assert list(model.hparams["core"]["color_squashing_weights"]) == [0.5, 0.5]

    build_model([0.5, 0.5]).load_state_dict(state_dict, strict=True)

    # A colour model must reject greyscale weights rather than quietly dropping them.
    with pytest.raises(RuntimeError):
        build_model(None).load_state_dict(state_dict, strict=True)
