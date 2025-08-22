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
