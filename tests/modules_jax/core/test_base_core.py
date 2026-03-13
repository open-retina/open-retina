from pathlib import Path

import hydra
import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from openretina.data_io.hoefling_2024.dataloaders import natmov_dataloaders_v2
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.data_io.hoefling_2024.stimuli import movies_from_pickle
from openretina.models.core_readout import load_core_readout_from_remote
from openretina.modules.core.base_core import SimpleCoreWrapper as TorchSimpleCoreWrapper
from openretina.modules_jax.core.base_core import SimpleCoreWrapper
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.utils.misc import check_server_responding


def _can_reach_huggingface() -> bool:
    try:
        return check_server_responding("https://huggingface.co/")
    except Exception:
        return False


def _base_kwargs() -> dict:
    return dict(
        channels=(2, 4, 5),
        temporal_kernel_sizes=(9, 5),
        spatial_kernel_sizes=(5, 3),
        gamma_input=1.2,
        gamma_hidden=0.7,
        gamma_in_sparse=0.3,
        gamma_temporal=0.9,
        dropout_rate=0.0,
        cut_first_n_frames=0,
        maxpool_every_n_layers=None,
        downsample_input_kernel_size=None,
        input_padding=0,
        hidden_padding=(0, 1, 1),
        color_squashing_weights=None,
        convolution_type="custom_separable",
    )


def _make_cores(core_kwargs: dict) -> tuple[TorchSimpleCoreWrapper, SimpleCoreWrapper]:
    torch_core = TorchSimpleCoreWrapper(**core_kwargs)
    jax_core = SimpleCoreWrapper(**core_kwargs, rngs=nnx.Rngs(0))
    return torch_core, jax_core


def _copy_torch_to_jax(torch_core: TorchSimpleCoreWrapper, jax_core: SimpleCoreWrapper) -> None:
    for torch_layer, jax_layer in zip(torch_core.features, jax_core.features, strict=True):
        with torch.no_grad():
            jax_layer.conv.sin_weights.value = jnp.asarray(torch_layer.conv.sin_weights.detach().cpu().numpy())
            jax_layer.conv.cos_weights.value = jnp.asarray(torch_layer.conv.cos_weights.detach().cpu().numpy())
            jax_layer.conv.weight_spatial.value = jnp.asarray(torch_layer.conv.weight_spatial.detach().cpu().numpy())

            jax_layer.norm.scale.value = jnp.asarray(torch_layer.norm.weight.detach().cpu().numpy())
            jax_layer.norm.bias.value = jnp.asarray(torch_layer.norm.bias.detach().cpu().numpy())
            if hasattr(jax_layer.norm, "mean") and hasattr(jax_layer.norm, "var"):
                jax_layer.norm.mean.value = jnp.asarray(torch_layer.norm.running_mean.detach().cpu().numpy())
                jax_layer.norm.var.value = jnp.asarray(torch_layer.norm.running_var.detach().cpu().numpy())
            elif hasattr(jax_layer.norm, "running_mean") and hasattr(jax_layer.norm, "running_var"):
                jax_layer.norm.running_mean.value = jnp.asarray(torch_layer.norm.running_mean.detach().cpu().numpy())
                jax_layer.norm.running_var.value = jnp.asarray(torch_layer.norm.running_var.detach().cpu().numpy())
            else:
                raise RuntimeError("Unsupported NNX BatchNorm state layout in test harness.")

            jax_layer.bias.bias.value = jnp.asarray(torch_layer.bias.bias.detach().cpu().numpy())


def _run_forward_parity(core_kwargs: dict, input_shape: tuple[int, int, int, int, int], seed: int = 1) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    torch_core, jax_core = _make_cores(core_kwargs)
    _copy_torch_to_jax(torch_core, jax_core)
    torch_core.eval()

    x_np = rng.normal(size=input_shape).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_core(torch.tensor(x_np)).detach().cpu().numpy()
    jax_out = np.asarray(jax_core(jnp.asarray(x_np), train=False))

    assert torch_out.shape == jax_out.shape
    assert np.allclose(jax_out, torch_out, rtol=1e-4, atol=1e-5)


FORWARD_CASES = [
    (
        "baseline",
        _base_kwargs(),
        (2, 2, 17, 18, 16),
    ),
    (
        "same_padding",
        {
            **_base_kwargs(),
            "input_padding": True,
            "hidden_padding": True,
        },
        (2, 2, 17, 18, 16),
    ),
    (
        "with_pooling",
        {
            **_base_kwargs(),
            "input_padding": True,
            "hidden_padding": True,
            "maxpool_every_n_layers": 1,
            "temporal_kernel_sizes": (5, 3),
            "spatial_kernel_sizes": (5, 3),
        },
        (2, 2, 20, 32, 32),
    ),
    (
        "with_downsample",
        {
            **_base_kwargs(),
            "downsample_input_kernel_size": (1, 2, 2),
        },
        (2, 2, 17, 20, 18),
    ),
    (
        "with_color_squash",
        {
            **_base_kwargs(),
            "channels": (1, 4, 5),
            "color_squashing_weights": (0.7, 0.3),
        },
        (2, 2, 17, 18, 16),
    ),
    (
        "dropout_eval_mode",
        {
            **_base_kwargs(),
            "dropout_rate": 0.25,
            "input_padding": True,
            "hidden_padding": True,
        },
        (2, 2, 17, 18, 16),
    ),
]


@pytest.mark.parametrize("_name,core_kwargs,input_shape", FORWARD_CASES)
def test_core_forward_parity_across_configs(_name: str, core_kwargs: dict, input_shape: tuple[int, ...]) -> None:
    _run_forward_parity(core_kwargs, input_shape)


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


@pytest.mark.parametrize(
    "core_kwargs",
    [
        _base_kwargs(),
        {**_base_kwargs(), "channels": (2, 4), "temporal_kernel_sizes": (9,), "spatial_kernel_sizes": (5,)},
        {
            **_base_kwargs(),
            "channels": (1, 4, 5),
            "color_squashing_weights": (0.2, 0.8),
            "gamma_input": 1.0,
            "gamma_hidden": 1.0,
            "gamma_in_sparse": 1.0,
            "gamma_temporal": 1.0,
        },
    ],
)
def test_regularizer_term_parity_with_torch(core_kwargs: dict) -> None:
    torch.manual_seed(2)
    torch_core, jax_core = _make_cores(core_kwargs)
    _copy_torch_to_jax(torch_core, jax_core)

    with torch.no_grad():
        torch_spatial = float(torch_core.spatial_laplace().cpu().item())
        torch_temporal = float(torch_core.temporal_smoothness().cpu().item())
        torch_g0 = float(torch_core.group_sparsity_0().cpu().item())
        try:
            torch_g = float(torch_core.group_sparsity().cpu().item())
            torch_total = float(torch_core.regularizer().cpu().item())
        except RuntimeError as exc:
            assert "non-empty TensorList" in str(exc)
            with pytest.raises(RuntimeError, match="non-empty TensorList"):
                _ = jax_core.group_sparsity()
            with pytest.raises(RuntimeError, match="non-empty TensorList"):
                _ = jax_core.regularizer()
            return

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
    kwargs = {
        **_base_kwargs(),
        "gamma_input": 0.0,
        "gamma_hidden": 0.0,
        "gamma_in_sparse": 0.0,
        "gamma_temporal": 0.0,
    }
    _, jax_core = _make_cores(kwargs)
    reg = float(np.asarray(jax_core.regularizer()))
    assert np.isclose(reg, 0.0)


@pytest.mark.parametrize("conv_type", ["separable", "full", "time_independent"])
def test_unsupported_conv_type_raises(conv_type: str) -> None:
    kwargs = {
        **_base_kwargs(),
        "conv_type": conv_type,
    }
    with pytest.raises(NotImplementedError):
        SimpleCoreWrapper(
            channels=kwargs["channels"],
            temporal_kernel_sizes=kwargs["temporal_kernel_sizes"],
            spatial_kernel_sizes=kwargs["spatial_kernel_sizes"],
            gamma_input=kwargs["gamma_input"],
            gamma_hidden=kwargs["gamma_hidden"],
            gamma_in_sparse=kwargs["gamma_in_sparse"],
            gamma_temporal=kwargs["gamma_temporal"],
            input_padding=kwargs["input_padding"],
            hidden_padding=kwargs["hidden_padding"],
            convolution_type=conv_type,
            rngs=nnx.Rngs(0),
        )


def _build_jax_core_from_torch_core(torch_core: TorchSimpleCoreWrapper) -> SimpleCoreWrapper:
    channels = [torch_core.features[0].conv.in_channels]
    channels.extend(layer.conv.out_channels for layer in torch_core.features)

    temporal_kernel_sizes = tuple(layer.conv.temporal_kernel_size for layer in torch_core.features)
    spatial_kernel_sizes = tuple(layer.conv.spatial_kernel_size for layer in torch_core.features)

    has_any_pool = any(hasattr(layer, "pool") for layer in torch_core.features)
    if has_any_pool:
        raise NotImplementedError("This parity helper currently supports pretrained cores without max pooling.")

    color_squashing_weights = None
    if torch_core.color_squashing_layer is not None:
        color_squashing_weights = tuple(float(x) for x in torch_core.color_squashing_layer.channel_weights.detach().cpu())

    input_padding = torch_core.features[0].conv.padding
    if len(torch_core.features) > 1:
        hidden_padding = torch_core.features[1].conv.padding
    else:
        hidden_padding = torch_core.features[0].conv.padding

    downsample_input_kernel_size = (
        tuple(torch_core._downsample_input_kernel_size) if torch_core._downsample_input_kernel_size is not None else None
    )

    return SimpleCoreWrapper(
        channels=tuple(channels),
        temporal_kernel_sizes=temporal_kernel_sizes,
        spatial_kernel_sizes=spatial_kernel_sizes,
        gamma_input=torch_core.gamma_input,
        gamma_hidden=torch_core.gamma_hidden,
        gamma_in_sparse=torch_core.gamma_in_sparse,
        gamma_temporal=torch_core.gamma_temporal,
        dropout_rate=0.0,
        cut_first_n_frames=torch_core._cut_first_n_frames,
        maxpool_every_n_layers=None,
        downsample_input_kernel_size=downsample_input_kernel_size,
        input_padding=input_padding,
        hidden_padding=hidden_padding,
        color_squashing_weights=color_squashing_weights,
        convolution_type=torch_core.convolution_type,
        rngs=nnx.Rngs(0),
    )


@pytest.mark.skipif(
    condition=not _can_reach_huggingface(),
    reason="Hugging Face unreachable.",
)
def test_pretrained_torch_core_parity_on_real_natmov_inputs_train_and_eval() -> None:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="hoefling_2024_core_readout_low_res.yaml")

    movies_path = get_local_file_path(file_path=cfg.paths.movies_path, cache_folder=cfg.paths.data_dir)
    movies_dict = movies_from_pickle(movies_path)

    responses_path = get_local_file_path(file_path=cfg.paths.responses_path, cache_folder=cfg.paths.data_dir)
    responses_dict = load_h5_into_dict(file_path=responses_path)
    filtered_responses_dict = filter_responses(responses_dict, **cfg.quality_checks)
    final_responses = make_final_responses(filtered_responses_dict, response_type="natural")

    dataloaders = natmov_dataloaders_v2(
        neuron_data_dictionary=final_responses,
        movies_dictionary=movies_dict,
        allow_over_boundaries=cfg.dataloader.allow_over_boundaries,
        batch_size=8,
        train_chunk_size=cfg.dataloader.train_chunk_size,
        validation_clip_indices=cfg.dataloader.validation_clip_indices,
        array_backend="jax",
    )
    session_key = next(iter(dataloaders["train"].keys()))
    inputs, _ = next(iter(dataloaders["train"][session_key]))
    
    x_jax = jnp.asarray(inputs[:2])
    x_torch = torch.tensor(np.asarray(x_jax), dtype=torch.float32)

    model = load_core_readout_from_remote("hoefling_2024_base_low_res", device="cpu")
    torch_core = model.core
    assert isinstance(torch_core, TorchSimpleCoreWrapper)

    jax_core = _build_jax_core_from_torch_core(torch_core)
    _copy_torch_to_jax(torch_core, jax_core)

    torch_core.eval()
    with torch.no_grad():
        torch_eval = torch_core(x_torch).detach().cpu().numpy()
    jax_eval = np.asarray(jax_core(x_jax, train=False))
    assert torch_eval.shape == jax_eval.shape
    assert np.allclose(jax_eval, torch_eval, rtol=1e-4, atol=2e-5)

    _copy_torch_to_jax(torch_core, jax_core)
    torch_core.train()
    with torch.no_grad():
        torch_train = torch_core(x_torch).detach().cpu().numpy()
    jax_train = np.asarray(jax_core(x_jax, train=True))
    assert torch_train.shape == jax_train.shape
    assert np.allclose(jax_train, torch_train, rtol=1e-4, atol=2e-5)
