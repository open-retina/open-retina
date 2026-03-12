import numpy as np
import pytest
import torch



import jax.numpy as jnp
from flax import nnx

from openretina.modules.layers.convolutions import (
    STSeparableBatchConv3d as TorchSTSeparableBatchConv3d,
)
from openretina.modules.layers.convolutions import compute_temporal_kernel as torch_compute_temporal_kernel
from openretina.modules.layers.jax_convolutions import (
    STSeparableBatchConv3d,
)
from openretina.modules.layers.jax_convolutions import compute_temporal_kernel as jax_compute_temporal_kernel
from openretina.modules.layers.jax_convolutions import temporal_smoothing as jax_temporal_smoothing


@pytest.mark.parametrize("temporal_kernel_size,spatial_kernel_size", [(1, 1), (1, 3), (5, 3)])
def test_st_separable_batch_conv3d_jax_shape(
    temporal_kernel_size: int,
    spatial_kernel_size: int,
) -> None:
    in_channels = 2
    subsampling_factor = min(temporal_kernel_size, 3)
    conv = STSeparableBatchConv3d(
        in_channels=in_channels,
        out_channels=5,
        log_speed_dict={},
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        subsampling_factor=subsampling_factor,
        rngs=nnx.Rngs(0),
    )

    input_ = jnp.zeros((1, in_channels, 7, 18, 16), dtype=jnp.float32)
    res = conv(input_)
    assert res.shape[0] == 1
    assert res.shape[1] == 5


def test_compute_temporal_kernel_parity() -> None:
    rng = np.random.default_rng(123)
    out_channels = 4
    in_channels = 2
    length = 9
    subsampling_factor = 3
    k = length // subsampling_factor

    log_speed = rng.normal(size=(1,)).astype(np.float32)
    sin_weights = rng.normal(size=(out_channels, in_channels, k)).astype(np.float32)
    cos_weights = rng.normal(size=(out_channels, in_channels, k)).astype(np.float32)

    torch_kernel = torch_compute_temporal_kernel(
        torch.tensor(log_speed),
        torch.tensor(sin_weights),
        torch.tensor(cos_weights),
        length=length,
        subsampling_factor=subsampling_factor,
    ).detach().cpu().numpy()

    jax_kernel = np.asarray(
        jax_compute_temporal_kernel(
            jnp.asarray(log_speed),
            jnp.asarray(sin_weights),
            jnp.asarray(cos_weights),
            length=length,
            subsampling_factor=subsampling_factor,
        )
    )

    assert np.allclose(jax_kernel, torch_kernel, rtol=1e-5, atol=1e-6)


def test_temporal_smoothing_shape_and_value() -> None:
    rng = np.random.default_rng(321)
    sin = rng.normal(size=(5, 2, 7)).astype(np.float32)
    cos = rng.normal(size=(5, 2, 7)).astype(np.float32)

    result = jax_temporal_smoothing(jnp.asarray(sin), jnp.asarray(cos))
    assert result.shape == ()
    assert float(result) >= 0.0


@pytest.mark.parametrize("padding", [0, "same"])
def test_forward_parity_with_torch(padding: int | str) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    in_channels = 2
    out_channels = 3
    temporal_kernel_size = 9
    spatial_kernel_size = 5
    subsampling_factor = 3

    torch_conv = TorchSTSeparableBatchConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        log_speed_dict={},
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        subsampling_factor=subsampling_factor,
        bias=True,
        padding=padding,
    )

    jax_conv = STSeparableBatchConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        log_speed_dict={},
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        subsampling_factor=subsampling_factor,
        bias=True,
        padding=padding,
        rngs=nnx.Rngs(0),
    )

    sin = rng.normal(size=torch_conv.sin_weights.shape).astype(np.float32)
    cos = rng.normal(size=torch_conv.cos_weights.shape).astype(np.float32)
    spatial = rng.normal(size=torch_conv.weight_spatial.shape).astype(np.float32)
    bias = rng.normal(size=torch_conv.bias.shape).astype(np.float32)

    with torch.no_grad():
        torch_conv.sin_weights.copy_(torch.tensor(sin))
        torch_conv.cos_weights.copy_(torch.tensor(cos))
        torch_conv.weight_spatial.copy_(torch.tensor(spatial))
        torch_conv.bias.copy_(torch.tensor(bias))

    jax_conv.sin_weights.value = jnp.asarray(sin)
    jax_conv.cos_weights.value = jnp.asarray(cos)
    jax_conv.weight_spatial.value = jnp.asarray(spatial)
    if jax_conv.bias is not None:
        jax_conv.bias.value = jnp.asarray(bias)

    x_np = rng.normal(size=(2, in_channels, 17, 18, 16)).astype(np.float32)
    torch_out = torch_conv(torch.tensor(x_np)).detach().cpu().numpy()
    jax_out = np.asarray(jax_conv(jnp.asarray(x_np)))

    assert np.allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)
