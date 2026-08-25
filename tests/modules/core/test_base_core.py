import pytest
import torch

from openretina.modules.core.base_core import DummyCore

IN_SHAPE = (1, 2, 150, 18, 16)  # batch channels time height width
CUT_FRAMES = 30


def _two_channel_input() -> torch.Tensor:
    x = torch.ones(IN_SHAPE)
    x[:, 0] *= 2.0
    x[:, 1] *= 3.0
    return x


@pytest.mark.parametrize("weights", [(0.5, 0.5), [0.5, 0.5]])
def test_dummy_core_squashes_color_to_greyscale(weights) -> None:
    """Hydra hands over a plain list, direct callers a tuple; both must work."""
    core = DummyCore(cut_first_n_frames=CUT_FRAMES, color_squashing_weights=weights)
    x = _two_channel_input()

    with torch.no_grad():
        out = core(x)

    assert out.shape == (1, 1, IN_SHAPE[2] - CUT_FRAMES, 18, 16)
    # 0.5 * 2 + 0.5 * 3 == 2.5, i.e. the mean of the two channels.
    assert torch.allclose(out, torch.full_like(out, 2.5))


def test_dummy_core_without_squashing_preserves_channels() -> None:
    """Backwards compatibility: the default must leave the stimulus untouched."""
    core = DummyCore(cut_first_n_frames=CUT_FRAMES)
    x = _two_channel_input()

    with torch.no_grad():
        out = core(x)

    assert core.color_squashing_layer is None
    assert out.shape == (1, 2, IN_SHAPE[2] - CUT_FRAMES, 18, 16)
    assert torch.allclose(out, x[:, :, CUT_FRAMES:])


def test_dummy_core_rejects_unknown_kwargs() -> None:
    """A swallowed typo would silently train on all color channels; it must raise instead."""
    with pytest.raises(TypeError):
        DummyCore(cut_first_n_frames=CUT_FRAMES, color_squashing_weight=(0.5, 0.5))  # type: ignore[call-arg]
