"""Residual depthwise core with causal temporal self-attention.


Inverted residual blocks with a factorised (2+1)D depthwise convolution (spatial depthwise conv followed by a
  temporal depthwise conv, narrow -> wide -> narrow channels) follow the DwiseNeuro core of the Sensorium 2023
  winning solution: https://github.com/lRomul/sensorium (Baikulov 2023).
- The attention block is a pre-norm transformer block that attends along time only,
  with a causal sliding-window mask and a learned per-head bias for each lag.

Layout (shapes for the Höfling low-res input, 2 x T x 18 x 16):

    stem   : the stock openretina first layer (``SimpleCoreWrapper`` with one layer):
             factorised conv, temporal 21 / spatial 11, valid               -> C x (T - 20) x 8 x 6
    block i: inverted residual block, keeps the spatial size:
               1x1x1 expand C -> e*C                    + BN + ELU
               depthwise spatial (1, k_s, k_s), 'same'  + BN + ELU
               depthwise temporal (k_t, 1, 1), valid    + BN + ELU   (crops k_t - 1 frames)
               1x1x1 project e*C -> C                   + BN
               out = ELU(x[:, :, k_t - 1:] + branch)
    attn   : causal temporal self-attention block(s) applied separately at every spatial position
             (pre-norm transformer block with a learned per-head lag bias and LayerScale residuals).
    output : C x (T - 20 - n_blocks * (k_t - 1)) x 8 x 6     (k_t = 6, 2 blocks -> T - 30, as the stock core)

Causality: every temporal convolution is valid, so its output at frame t depends only on frames <= t, and the
attention mask only allows frame t to attend to frames t - window + 1 .. t.

Regularisation: the stem keeps the stock regularisers (spatial Laplace, temporal smoothness, input sparsity),
so the stock ``gamma_*`` settings keep their meaning. Blocks and attention are regularised only by weight decay.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from openretina.modules.core.base_core import Core, SimpleCoreWrapper


class DepthwiseSpatialConv3d(nn.Conv3d):
    """(1, k, k) depthwise ``Conv3d`` computed as a per-frame depthwise ``Conv2d``.

    Same parameters and state dict as ``nn.Conv3d`` and identical outputs, but runs on the optimised 2D depthwise
    kernels instead of PyTorch's slow native depthwise-3D kernel.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = nn.functional.conv2d(
            x.transpose(1, 2).reshape(b * t, c, h, w),
            self.weight[:, :, 0],
            self.bias,
            padding=self.padding[1:],
            groups=self.groups,
        )
        return y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3]).transpose(1, 2)


class DepthwiseTemporalConv3d(nn.Conv3d):
    """(k, 1, 1) depthwise ``Conv3d`` (valid, causal) computed as a per-pixel depthwise ``Conv1d`` along time.

    Same parameters and state dict as ``nn.Conv3d`` and identical outputs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = nn.functional.conv1d(
            x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t),
            self.weight[:, :, :, 0, 0],
            self.bias,
            groups=self.groups,
        )
        return y.reshape(b, h, w, y.shape[1], y.shape[2]).permute(0, 3, 4, 1, 2)


class InvertedResidualBlock3d(nn.Module):
    """Inverted residual block: 1x1x1 expand, depthwise spatial, depthwise causal temporal, 1x1x1 project.

    The skip connection is cropped by ``temporal_kernel - 1`` frames at the start to match the valid temporal conv.
    """

    def __init__(self, channels: int, expansion: int = 4, spatial_kernel: int = 5, temporal_kernel: int = 6):
        super().__init__()
        hidden = channels * expansion
        self.temporal_crop = temporal_kernel - 1
        self.expand = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False), nn.BatchNorm3d(hidden), nn.ELU()
        )
        self.dw_spatial = nn.Sequential(
            DepthwiseSpatialConv3d(
                hidden,
                hidden,
                kernel_size=(1, spatial_kernel, spatial_kernel),
                padding=(0, spatial_kernel // 2, spatial_kernel // 2),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm3d(hidden),
            nn.ELU(),
        )
        self.dw_temporal = nn.Sequential(
            DepthwiseTemporalConv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1), groups=hidden, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ELU(),
        )
        self.project = nn.Sequential(nn.Conv3d(hidden, channels, kernel_size=1, bias=False), nn.BatchNorm3d(channels))
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project(self.dw_temporal(self.dw_spatial(self.expand(x))))
        return self.act(x[:, :, self.temporal_crop :] + h)


class CausalTemporalAttention(nn.Module):
    """Pre-norm transformer block that attends along time only, separately at every spatial position.

    Tokens are the C-dimensional feature vectors at each (position, frame). Frame t attends to frames
    t - window + 1 .. t (causal sliding window). Positions are encoded only by a learned per-head bias for each lag
    (relative positions, no absolute positions), so training and test sequences may differ in length. Weights are
    shared across spatial positions. Both residual branches (attention and MLP) are gated by LayerScale
    , initialised at ``ls_init``. 
    """

    def __init__(
        self,
        channels: int,
        heads: int = 2,
        window: int = 256,
        mlp_ratio: int = 4,
        ls_init: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels=} must be divisible by {heads=}")
        if window < 1:
            raise ValueError(f"{window=} must be positive")
        self.heads, self.window, self.dropout = heads, window, dropout
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_ratio * channels), nn.GELU(), nn.Linear(mlp_ratio * channels, channels)
        )
        self.ls1 = nn.Parameter(ls_init * torch.ones(channels))
        self.ls2 = nn.Parameter(ls_init * torch.ones(channels))
        self.lag_bias = nn.Parameter(torch.zeros(heads, window))

    def attn_bias(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Additive attention mask of shape (1, heads, t, t): learned lag bias inside the window, -inf elsewhere."""
        idx = torch.arange(t, device=device)
        lag = idx[:, None] - idx[None, :]  # query i, key j: lag = i - j
        allowed = (lag >= 0) & (lag < self.window)
        bias = self.lag_bias[:, lag.clamp(0, self.window - 1)]
        return bias.masked_fill(~allowed, float("-inf")).to(dtype).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: B, C, T, H, W
        b, c, t, h, w = x.shape
        z = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        qkv = self.qkv(self.norm1(z)).view(b * h * w, t, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: N, heads, T, d
        a = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.attn_bias(t, z.device, q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        z = z + self.ls1 * self.proj(a.transpose(1, 2).reshape(b * h * w, t, c))
        z = z + self.ls2 * self.mlp(self.norm2(z))
        return z.view(b, h, w, t, c).permute(0, 4, 3, 1, 2).contiguous()

    def extra_repr(self) -> str:
        return f"heads={self.heads}, window={self.window}"


class ResidualAttentionCore(Core):
    """Stock openretina first layer, inverted residual depthwise blocks, and causal temporal attention.

    Args:
        channels: (input channels, stem width). The stem width is also the width of the blocks and attention.
        temporal_kernel_sizes: temporal kernel of the stem; only the first entry is used.
        spatial_kernel_sizes: spatial kernel of the stem; only the first entry is used.
        gamma_input, gamma_temporal, gamma_in_sparse, gamma_hidden: stem regularisation, as in ``SimpleCoreWrapper``.
        n_blocks: number of inverted residual blocks.
        block_expansion: channel expansion factor inside the blocks.
        block_spatial_kernel: depthwise spatial kernel size, an int or one value per block.
        block_temporal_kernel: depthwise temporal kernel size, an int or one value per block.
        attn_blocks: number of causal temporal attention blocks after the residual blocks (0 disables attention).
        attn_heads: attention heads; must divide the stem width.
        attn_window: how many past frames (including the current one) each frame may attend to.
        attn_mlp_ratio: hidden width of the attention MLP relative to the stem width.
        attn_ls_init: initial LayerScale value of both attention-block residual branches.
        attn_dropout: dropout on the attention weights during training.
        n_neurons_dict: passed by ``UnifiedCoreReadout``; unused.
        **stem_kwargs: further keyword arguments for the stem ``SimpleCoreWrapper`` (e.g. ``convolution_type``,
            ``cut_first_n_frames``, ``input_padding``, ``dropout_rate``).
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        temporal_kernel_sizes: tuple[int, ...],
        spatial_kernel_sizes: tuple[int, ...],
        gamma_input: float,
        gamma_temporal: float,
        gamma_in_sparse: float,
        gamma_hidden: float = 0.0,
        n_blocks: int = 2,
        block_expansion: int = 4,
        block_spatial_kernel: int | list[int] = 5,
        block_temporal_kernel: int | list[int] = 6,
        attn_blocks: int = 1,
        attn_heads: int = 2,
        attn_window: int = 256,
        attn_mlp_ratio: int = 4,
        attn_ls_init: float = 0.1,
        attn_dropout: float = 0.0,
        n_neurons_dict: dict[str, int] | None = None,
        **stem_kwargs: Any,
    ):
        super().__init__()
        channels = tuple(channels)
        if len(channels) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects channels=(input channels, stem width), i.e. hidden_channels "
                f"with a single entry; got {channels=}"
            )
        self.stem = SimpleCoreWrapper(
            channels=channels,
            temporal_kernel_sizes=tuple(temporal_kernel_sizes)[:1],
            spatial_kernel_sizes=tuple(spatial_kernel_sizes)[:1],
            gamma_input=gamma_input,
            gamma_temporal=gamma_temporal,
            gamma_in_sparse=gamma_in_sparse,
            gamma_hidden=gamma_hidden,
            **stem_kwargs,
        )
        width = channels[-1]
        ks = _per_block(block_spatial_kernel, n_blocks, "block_spatial_kernel")
        kt = _per_block(block_temporal_kernel, n_blocks, "block_temporal_kernel")
        layers: list[nn.Module] = [
            InvertedResidualBlock3d(width, expansion=block_expansion, spatial_kernel=ks[i], temporal_kernel=kt[i])
            for i in range(n_blocks)
        ]
        layers += [
            CausalTemporalAttention(width, attn_heads, attn_window, attn_mlp_ratio, attn_ls_init, attn_dropout)
            for _ in range(attn_blocks)
        ]
        self.blocks = nn.Sequential(*layers)
        # Reference to the stem layers for helpers that look for a `features` attribute.
        self.features = self.stem.features
        self.gamma_input, self.gamma_temporal, self.gamma_in_sparse = gamma_input, gamma_temporal, gamma_in_sparse

    def forward(self, x: torch.Tensor, data_key: str | None = None) -> torch.Tensor:
        return self.blocks(self.stem(x))

    def regularizer(self) -> torch.Tensor:
        return self.stem.regularizer()

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        """Visualises the stem filters (the blocks' 1x1x1 and depthwise filters are not visualised)."""
        self.stem.save_weight_visualizations(folder_path, file_format, state_suffix)


def _per_block(value: int | list[int] | tuple[int, ...], n_blocks: int, name: str) -> list[int]:
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n_blocks
    if len(values) != n_blocks:
        raise ValueError(f"{name} needs {n_blocks} entries (one per block), got {values}")
    return values
