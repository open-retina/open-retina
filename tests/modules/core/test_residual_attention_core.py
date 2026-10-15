import os

import hydra
import pytest
import torch
from torch import nn

from openretina.models.core_readout import UnifiedCoreReadout
from openretina.modules.core.residual_attention_core import (
    CausalTemporalAttention,
    DepthwiseSpatialConv3d,
    DepthwiseTemporalConv3d,
    ResidualAttentionCore,
)
from openretina.utils.debug_model import is_model_causal

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs")


def _core(**kwargs) -> ResidualAttentionCore:
    params = dict(
        channels=(2, 8),
        temporal_kernel_sizes=(21,),
        spatial_kernel_sizes=(11,),
        gamma_input=0.0,
        gamma_temporal=40.0,
        gamma_in_sparse=0.0,
        cut_first_n_frames=0,
        input_padding=0,
        attn_window=16,
    )
    params.update(kwargs)
    return ResidualAttentionCore(**params)


def test_output_shape_and_crop() -> None:
    core = _core().eval()
    out = core(torch.randn(2, 2, 60, 18, 16))
    # stem: valid temporal 21 (-20 frames), valid spatial 11 (18x16 -> 8x6); 2 blocks with temporal kernel 6: -10
    assert out.shape == (2, 8, 30, 8, 6)


def test_no_attention_and_custom_blocks() -> None:
    core = _core(attn_blocks=0, n_blocks=3, block_temporal_kernel=[3, 3, 5]).eval()
    assert not any(isinstance(m, CausalTemporalAttention) for m in core.modules())
    assert core(torch.randn(1, 2, 50, 18, 16)).shape[2] == 50 - 20 - 2 - 2 - 4


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        _core(channels=(2, 8, 8))
    with pytest.raises(ValueError):
        _core(block_spatial_kernel=[5, 5, 5])
    with pytest.raises(ValueError):
        _core(attn_heads=3)


@pytest.mark.parametrize("attn_blocks", [0, 1])
def test_core_is_causal(attn_blocks: int) -> None:
    torch.manual_seed(0)
    core = _core(attn_blocks=attn_blocks).eval()
    x = torch.randn(1, 2, 80, 18, 16)
    with torch.no_grad():
        y = core(x)
        x2 = x.clone()
        x2[:, :, 60:] = torch.randn_like(x2[:, :, 60:])  # change frames >= 60
        y2 = core(x2)
    # output frame i corresponds to input frame i + 30; outputs for input frames < 60 must not change
    assert torch.allclose(y[:, :, :30], y2[:, :, :30], atol=1e-6)
    assert not torch.allclose(y[:, :, 30:], y2[:, :, 30:])


def test_attention_window_limits_context() -> None:
    torch.manual_seed(0)
    attn = CausalTemporalAttention(channels=8, heads=2, window=5).eval()
    x = torch.randn(1, 8, 20, 2, 2)
    with torch.no_grad():
        y = attn(x)
        x2 = x.clone()
        x2[:, :, 3] += torch.randn(8, 2, 2)  # perturb frame 3: only frames 3..7 may see it
        y2 = attn(x2)
    changed = (y - y2).abs().amax(dim=(0, 1, 3, 4)) > 1e-6
    assert changed.nonzero().flatten().tolist() == [3, 4, 5, 6, 7]


def test_fast_depthwise_convs_match_conv3d() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 9, 7, 5)
    spatial = DepthwiseSpatialConv3d(6, 6, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=6, bias=True)
    temporal = DepthwiseTemporalConv3d(6, 6, kernel_size=(4, 1, 1), groups=6, bias=True)
    for fast in (spatial, temporal):
        reference = nn.Conv3d(6, 6, fast.kernel_size, padding=fast.padding, groups=6, bias=True)
        reference.load_state_dict(fast.state_dict())
        assert torch.allclose(fast(x), reference(x), atol=1e-5)


def test_regularizer_and_backward() -> None:
    core = _core(gamma_input=0.1)
    out = core(torch.randn(2, 2, 40, 18, 16))
    loss = out.mean() + core.regularizer()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in core.parameters() if p.requires_grad)


def test_weight_visualization(tmp_path) -> None:
    _core().save_weight_visualizations(str(tmp_path))
    assert any(tmp_path.rglob("*.jpg"))


def test_model_config_builds_causal_unified_core_readout() -> None:
    with hydra.initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base="1.3"):
        cfg = hydra.compose(
            config_name="hoefling_2024_core_readout_low_res_residual_attention",
            overrides=["paths.cache_dir=/tmp", "paths.output_dir=/tmp"],
        )
    assert cfg.dataloader.train_chunk_size == 300
    assert cfg.dataloader.batch_size == 15
    assert cfg.trainer.max_epochs == 200
    assert "ema" in cfg.training_callbacks
    assert cfg.model.core.attn_window == 256

    # as in openretina.cli.train: the model config has no _target_ and is passed to UnifiedCoreReadout
    cfg.model.n_neurons_dict = {"session_a": 5, "session_b": 3}
    model = UnifiedCoreReadout(data_info={"input_shape": (2, 18, 16)}, **cfg.model)
    assert isinstance(model, UnifiedCoreReadout)
    assert isinstance(model.core, ResidualAttentionCore)
    stimulus = torch.rand(model.stimulus_shape(time_steps=60, num_batches=2))
    responses = model(stimulus, data_key="session_b")
    assert responses.shape == (2, 30, 3)
    assert torch.all(responses >= 0)
    assert is_model_causal(model)
