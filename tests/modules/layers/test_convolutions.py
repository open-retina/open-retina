import pytest
import torch

from openretina.modules.layers.convolutions import STSeparableBatchConv3d


@pytest.mark.parametrize(
    "temporal_kernel_size, spatial_kernel_size",
    [
        (1, 1),
        (1, 3),
        (5, 3),
    ],
)
def test_load_core_readout_from_remote(
    temporal_kernel_size: int,
    spatial_kernel_size: int,
) -> None:
    in_channels = 2
    conv = STSeparableBatchConv3d(
        in_channels=in_channels,
        out_channels=5,
        log_speed_dict={},
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
    )
    input_ = torch.zeros((1, in_channels, 7, 18, 16))
    res = conv.forward(input_)
    assert res is not None
