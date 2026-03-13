from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib.colors import Normalize


def _next_rng(rngs: nnx.Rngs | None, stream: str, fallback_seed: int) -> jax.Array:
    if rngs is None:
        return jax.random.PRNGKey(fallback_seed)
    elif isinstance(rngs, nnx.Rngs):
        stream_fn = getattr(rngs, stream)
        return stream_fn()
    else:
        raise ValueError(f"Could not retrieve rng stream '{stream}' from {type(rngs)}")


def _conv3d_ncthw(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    stride: int | tuple[int, int, int],
    padding: int | str | tuple[int, ...],
) -> jax.Array:
    if isinstance(stride, int):
        strides = (stride, stride, stride)
    else:
        if len(stride) != 3:
            raise ValueError(f"Expected stride with 3 values for 3D conv, got {stride}")
        strides = tuple(int(s) for s in stride)

    if isinstance(padding, str):
        normalized = padding.upper()
        if normalized == "SAME":
            conv_padding: str | tuple[tuple[int, int], ...] = "SAME"
        elif normalized == "VALID":
            conv_padding = "VALID"
        else:
            raise ValueError(f"Unsupported string padding '{padding}'")
    elif isinstance(padding, int):
        conv_padding = ((padding, padding), (padding, padding), (padding, padding))
    else:
        if len(padding) != 3:
            raise ValueError(f"Expected 3D padding tuple, got {padding}")
        conv_padding = tuple((int(p), int(p)) for p in padding)

    out = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=weight,
        window_strides=strides,
        padding=conv_padding,
        dimension_numbers=("NCTHW", "OITHW", "NCTHW"),
    )
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1, 1)
    return out

def compute_temporal_kernel(
    log_speed: jax.Array,
    sin_weights: jax.Array,
    cos_weights: jax.Array,
    length: int,
    subsampling_factor: int,
) -> jax.Array:
    stretches = jnp.exp(log_speed)
    sines, cosines = STSeparableBatchConv3d.temporal_basis(stretches, length, subsampling_factor)
    weights_temporal = jnp.sum(sin_weights[:, :, :, None] * sines[None, None, ...], axis=2) + jnp.sum(
        cos_weights[:, :, :, None] * cosines[None, None, ...], axis=2
    )
    return weights_temporal[..., None, None]


class STSeparableBatchConv3d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: int | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | str | tuple[int, ...] = 0,
        num_scans: int = 1,
        bias: bool = True,
        subsampling_factor: int = 3,
        rngs: nnx.Rngs | None = None,
    ):
        if log_speed_dict:
            raise NotImplementedError(
                "Session-specific log_speed_dict is not implemented for JAX STSeparableBatchConv3d in phase 1."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_kernel_size2 = spatial_kernel_size2 if spatial_kernel_size2 is not None else spatial_kernel_size
        self.stride = stride
        self.padding = padding
        self.num_scans = num_scans
        self.subsampling_factor = subsampling_factor

        k = temporal_kernel_size // subsampling_factor
        if temporal_kernel_size < subsampling_factor:
            raise ValueError(
                f"Length cannot be smaller than subsampling factor: {temporal_kernel_size=} {subsampling_factor=}"
            )


        key_sin = _next_rng(rngs, stream="params", fallback_seed=0)
        key_cos = _next_rng(rngs, stream="params", fallback_seed=1)
        key_space = _next_rng(rngs, stream="params", fallback_seed=2)

        self.sin_weights = nnx.Param(
            jax.random.normal(key_sin, (out_channels, in_channels, k), dtype=jnp.float32) * 0.01
        )
        self.cos_weights = nnx.Param(
            jax.random.normal(key_cos, (out_channels, in_channels, k), dtype=jnp.float32) * 0.01
        )
        self.weight_spatial = nnx.Param(
            jax.random.normal(
                key_space,
                (out_channels, in_channels, 1, self.spatial_kernel_size, self.spatial_kernel_size2),
                dtype=jnp.float32,
            )
            * 0.01
        )

        if bias:
            self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=jnp.float32))
            self._bias_scale = 1.0
        else:
            # Keep call path branch-free: bias is present but scaled to zero.
            self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=jnp.float32))
            self._bias_scale = 0.0
        self._log_speed_default = jnp.zeros((1,), dtype=jnp.float32)

    @nnx.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        weight_temporal = compute_temporal_kernel(
            self._log_speed_default,
            self.sin_weights.value,
            self.cos_weights.value,
            self.temporal_kernel_size,
            self.subsampling_factor,
        )


        weight = jnp.einsum(
            "oitxx,oixhw->oithw",
            weight_temporal,
            self.weight_spatial.value,
        )
        bias_value = self.bias.value * self._bias_scale
        return _conv3d_ncthw(x, weight, bias_value, self.stride, self.padding)

    def get_spatial_weight(self, in_channel: int, out_channel: int) -> np.ndarray:
        spatial_2d_tensor = self.weight_spatial.value[out_channel, in_channel, 0]
        return np.asarray(jax.device_get(spatial_2d_tensor))

    def get_temporal_weight(self, in_channel: int, out_channel: int) -> tuple[np.ndarray, float]:
        weight_temporal = compute_temporal_kernel(
            self._log_speed_default,
            self.sin_weights.value,
            self.cos_weights.value,
            self.temporal_kernel_size,
            self.subsampling_factor,
        )
        global_abs_max = float(np.asarray(jax.device_get(jnp.abs(weight_temporal).max())).item())
        temporal_trace_tensor = weight_temporal[out_channel, in_channel, :, 0, 0]
        temporal_trace_np = np.asarray(jax.device_get(temporal_trace_tensor))
        return temporal_trace_np, global_abs_max

    def get_sin_cos_weights(self, in_channel: int, out_channel: int) -> tuple[np.ndarray, np.ndarray]:
        sin_trace = np.asarray(jax.device_get(self.sin_weights.value[out_channel, in_channel]))
        cos_trace = np.asarray(jax.device_get(self.cos_weights.value[out_channel, in_channel]))
        return sin_trace, cos_trace

    def plot_weights(
        self,
        in_channel: int,
        out_channel: int,
        plot_log_speed: bool = False,
        add_titles: bool = True,
        remove_ticks: bool = False,
        spatial_weight_center_positive: bool = True,
    ) -> plt.Figure:
        ncols = 4 if plot_log_speed else 3
        fig, axes = plt.subplots(ncols=ncols, figsize=(ncols * 6, 6))
        axes = np.atleast_1d(axes)

        spatial_weight = self.get_spatial_weight(in_channel, out_channel)
        temporal_weight, temporal_abs_max = self.get_temporal_weight(in_channel, out_channel)
        sin_trace, cos_trace = self.get_sin_cos_weights(in_channel, out_channel)

        center_x, center_y = int(spatial_weight.shape[0] / 2), int(spatial_weight.shape[1] / 2)
        if (
            spatial_weight_center_positive
            and np.mean(spatial_weight[max(center_x - 3, 0) : center_x + 2, max(center_y - 3, 0) : center_y + 2]) < 0
        ):
            spatial_weight *= -1
            temporal_weight *= -1
            sin_trace *= -1
            cos_trace *= -1

        abs_max = float(np.asarray(jax.device_get(jnp.abs(self.weight_spatial.value).max())).item())
        im = axes[0].imshow(
            spatial_weight, interpolation="none", cmap="RdBu_r", norm=Normalize(vmin=-abs_max, vmax=abs_max)
        )
        color_bar = fig.colorbar(im, orientation="vertical")

        axes[1].set_ylim(-temporal_abs_max, temporal_abs_max)
        axes[1].plot(temporal_weight)

        trace_max = max(
            float(np.asarray(jax.device_get(jnp.abs(self.sin_weights.value).max())).item()),
            float(np.asarray(jax.device_get(jnp.abs(self.cos_weights.value).max())).item()),
        )
        trace_max *= 1.1
        axes[2].set_ylim(-trace_max, trace_max)
        axes[2].plot(sin_trace, label="sin")
        axes[2].plot(cos_trace, label="cos")
        axes[2].legend()

        if plot_log_speed:
            axes[3].bar(["default"], [float(np.asarray(jax.device_get(self._log_speed_default[0])).item())])

        if add_titles:
            fig.suptitle(f"Weights for in_channel={in_channel} out_channel={out_channel}")
            axes[0].set_title("Spatial Weight")
            axes[1].set_title("Temporal Weight")
            axes[2].set_title("Sin/Cos Weights")

        if remove_ticks:
            for ax in axes:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            color_bar.set_ticks([])

        return fig

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        for in_channel in range(self.in_channels):
            for out_channel in range(self.out_channels):
                plot_path = f"{folder_path}/{in_channel}_{out_channel}_{state_suffix}.{file_format}"
                fig = self.plot_weights(in_channel, out_channel)
                fig.savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
                fig.clf()
                plt.close(fig)

    @staticmethod
    def temporal_weights(
        length: int,
        num_channels: int,
        num_feat: int,
        scale: float = 0.01,
        subsampling_factor: int = 3,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if length < subsampling_factor:
            raise ValueError(f"Length cannot be smaller than subsampling factor: {length=} {subsampling_factor=}")

        k = length // subsampling_factor
        key_sin = _next_rng(rngs, stream="params", fallback_seed=10)
        key_cos = _next_rng(rngs, stream="params", fallback_seed=11)
        sin_weights = jax.random.normal(key_sin, (num_feat, num_channels, k), dtype=jnp.float32) * scale
        cos_weights = jax.random.normal(key_cos, (num_feat, num_channels, k), dtype=jnp.float32) * scale
        return sin_weights, cos_weights

    @staticmethod
    def temporal_basis(stretches: jax.Array, length: int, subsampling_factor: int) -> tuple[jax.Array, jax.Array]:

        big_k = length // subsampling_factor
        time = jnp.arange(length, dtype=jnp.float32) - length
        stretched = stretches * time
        freq = stretched * 2.0 * np.pi / float(length)
        mask = STSeparableBatchConv3d.mask_tf(time, stretches, length)

        sines = [mask * jnp.sin(freq * k) for k in range(big_k)]
        cosines = [mask * jnp.cos(freq * k) for k in range(big_k)]
        sines_stacked = jnp.stack(sines, axis=0)
        cosines_stacked = jnp.stack(cosines, axis=0)
        return sines_stacked, cosines_stacked

    @staticmethod
    def mask_tf(time: jax.Array, stretch: jax.Array, T: int) -> jax.Array:
        mask = 1.0 / (1.0 + jnp.exp(-time - float(int(T * 0.95)) / stretch))
        return mask.T if mask.ndim > 1 else mask


def temporal_smoothing(sin: jax.Array, cos: jax.Array) -> jax.Array:
    smoother = jnp.linspace(0.1, 0.9, sin.shape[2], dtype=sin.dtype)[None, None, :]
    features = float(sin.shape[0])
    reg = jnp.sum((smoother * sin) ** 2) / features
    reg += jnp.sum((smoother * cos) ** 2) / features
    return reg
