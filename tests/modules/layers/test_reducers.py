import pytest
import torch

from openretina.modules.layers.reducers import WeightedChannelSumLayer


@pytest.mark.parametrize(
    "init_channel_weights",
    [(5, 7), (0.5, 0.5)],
)
def test_weighted_channel_sum_layer(
    init_channel_weights: tuple[float, ...],
):
    layer = WeightedChannelSumLayer(init_channel_weights)

    x = torch.ones((1, 2, 50, 18, 16))
    x[:, 0, ...] *= 2
    x[:, 1, ...] *= 3
    x_out = layer.forward(x)
    assert x_out.shape == (1, 1, 50, 18, 16)
    assert torch.allclose(
        x_out, x[:, 0:1, ...] * init_channel_weights[0] + x[:, 1:2, ...] * init_channel_weights[1], atol=1e-9
    )


def test_weighted_channel_sum_layer_passes_through_single_channel():
    """A stimulus that is already greyscale must be returned untouched."""
    layer = WeightedChannelSumLayer((0.5, 0.5))

    x = torch.ones((1, 1, 50, 18, 16))
    assert torch.equal(layer.forward(x), x)


@pytest.mark.parametrize("init_channel_weights", [(0.5,), (0.3, 0.3, 0.4)])
def test_weighted_channel_sum_layer_rejects_channel_mismatch(
    init_channel_weights: tuple[float, ...],
):
    """A single weight would otherwise broadcast and silently sum all channels equally."""
    layer = WeightedChannelSumLayer(init_channel_weights)

    with pytest.raises(ValueError, match="channel weights"):
        layer.forward(torch.ones((1, 2, 50, 18, 16)))
