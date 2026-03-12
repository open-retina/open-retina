import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from openretina.modules.core.base_core import SimpleCoreWrapper as TorchSimpleCoreWrapper
from openretina.modules.core.jax_base_core import SimpleCoreWrapper


def _make_cores(
    *,
    gamma_input: float = 1.2,
    gamma_hidden: float = 0.7,
    gamma_in_sparse: float = 0.3,
    gamma_temporal: float = 0.9,
) -> tuple[TorchSimpleCoreWrapper, SimpleCoreWrapper]:
    kwargs = dict(
        channels=(2, 4, 5),
        temporal_kernel_sizes=(9, 5),
        spatial_kernel_sizes=(5, 3),
        gamma_input=gamma_input,
        gamma_hidden=gamma_hidden,
        gamma_in_sparse=gamma_in_sparse,
        gamma_temporal=gamma_temporal,
        dropout_rate=0.0,
        cut_first_n_frames=0,
        maxpool_every_n_layers=None,
        downsample_input_kernel_size=None,
        input_padding=0,
        hidden_padding=(0, 1, 1),
        color_squashing_weights=None,
        convolution_type="custom_separable",
    )

    torch_core = TorchSimpleCoreWrapper(**kwargs)
    jax_core = SimpleCoreWrapper(**kwargs, rngs=nnx.Rngs(0))
    return torch_core, jax_core


def _copy_torch_to_jax(torch_core: TorchSimpleCoreWrapper, jax_core: SimpleCoreWrapper) -> None:
    for torch_layer, jax_layer in zip(torch_core.features, jax_core.features, strict=True):
        with torch.no_grad():
            jax_layer.conv.sin_weights.value = jnp.asarray(torch_layer.conv.sin_weights.detach().cpu().numpy())
            jax_layer.conv.cos_weights.value = jnp.asarray(torch_layer.conv.cos_weights.detach().cpu().numpy())
            jax_layer.conv.weight_spatial.value = jnp.asarray(torch_layer.conv.weight_spatial.detach().cpu().numpy())

            jax_layer.norm.scale.value = jnp.asarray(torch_layer.norm.weight.detach().cpu().numpy())
            jax_layer.norm.bias.value = jnp.asarray(torch_layer.norm.bias.detach().cpu().numpy())
            jax_layer.norm.running_mean.value = jnp.asarray(torch_layer.norm.running_mean.detach().cpu().numpy())
            jax_layer.norm.running_var.value = jnp.asarray(torch_layer.norm.running_var.detach().cpu().numpy())

            jax_layer.bias.bias.value = jnp.asarray(torch_layer.bias.bias.detach().cpu().numpy())


def test_core_forward_parity_with_torch() -> None:
    torch.manual_seed(1)
    rng = np.random.default_rng(1)

    torch_core, jax_core = _make_cores()
    _copy_torch_to_jax(torch_core, jax_core)

    torch_core.eval()

    x_np = rng.normal(size=(2, 2, 17, 18, 16)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_core(torch.tensor(x_np)).detach().cpu().numpy()

    jax_out = np.asarray(jax_core(jnp.asarray(x_np), train=False))

    assert torch_out.shape == jax_out.shape
    assert np.allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("cut_first_n_frames", [0, 3, 7])
def test_cut_first_n_frames_behavior(cut_first_n_frames: int) -> None:
    core = SimpleCoreWrapper(
        channels=(2, 4),
        temporal_kernel_sizes=(5,),
        spatial_kernel_sizes=(3,),
        gamma_input=0.0,
        gamma_hidden=0.0,
        gamma_in_sparse=0.0,
        gamma_temporal=0.0,
        cut_first_n_frames=cut_first_n_frames,
        input_padding=0,
        hidden_padding=(0, 1, 1),
        convolution_type="custom_separable",
        rngs=nnx.Rngs(0),
    )

    x = jnp.zeros((1, 2, 20, 8, 8), dtype=jnp.float32)
    y = core(x, train=False)
    assert y.shape[2] == (20 - 5 + 1) - cut_first_n_frames


def test_regularizer_term_parity_with_torch() -> None:
    torch.manual_seed(2)
    torch_core, jax_core = _make_cores(gamma_input=1.0, gamma_hidden=1.0, gamma_in_sparse=1.0, gamma_temporal=1.0)
    _copy_torch_to_jax(torch_core, jax_core)

    with torch.no_grad():
        torch_spatial = float(torch_core.spatial_laplace().cpu().item())
        torch_temporal = float(torch_core.temporal_smoothness().cpu().item())
        torch_g0 = float(torch_core.group_sparsity_0().cpu().item())
        torch_g = float(torch_core.group_sparsity().cpu().item())
        torch_total = float(torch_core.regularizer().cpu().item())

    jax_spatial = float(np.asarray(jax_core.spatial_laplace()))
    jax_temporal = float(np.asarray(jax_core.temporal_smoothness()))
    jax_g0 = float(np.asarray(jax_core.group_sparsity_0()))
    jax_g = float(np.asarray(jax_core.group_sparsity()))
    jax_total = float(np.asarray(jax_core.regularizer()))

    assert np.allclose(jax_spatial, torch_spatial, rtol=1e-5, atol=1e-6)
    assert np.allclose(jax_temporal, torch_temporal, rtol=1e-5, atol=1e-6)
    assert np.allclose(jax_g0, torch_g0, rtol=1e-5, atol=1e-6)
    assert np.allclose(jax_g, torch_g, rtol=1e-5, atol=1e-6)
    assert np.allclose(jax_total, torch_total, rtol=1e-5, atol=1e-6)


def test_gamma_lazy_gating() -> None:
    _, jax_core = _make_cores(gamma_input=0.0, gamma_hidden=0.0, gamma_in_sparse=0.0, gamma_temporal=0.0)
    reg = float(np.asarray(jax_core.regularizer()))
    assert np.isclose(reg, 0.0)


@pytest.mark.parametrize("conv_type", ["separable", "full", "time_independent"])
def test_unsupported_conv_type_raises(conv_type: str) -> None:
    with pytest.raises(NotImplementedError):
        SimpleCoreWrapper(
            channels=(2, 4),
            temporal_kernel_sizes=(5,),
            spatial_kernel_sizes=(3,),
            gamma_input=0.0,
            gamma_hidden=0.0,
            gamma_in_sparse=0.0,
            gamma_temporal=0.0,
            input_padding=0,
            hidden_padding=(0, 1, 1),
            convolution_type=conv_type,
            rngs=nnx.Rngs(0),
        )
