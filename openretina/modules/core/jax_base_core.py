from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openretina.modules.layers.jax_convolutions import STSeparableBatchConv3d, temporal_smoothing

EPS = 1e-6

LAPLACE_1D = jnp.array([-1.0, 2.0, -1.0], dtype=jnp.float32)[None, None, ...]
LAPLACE_3x3 = jnp.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32)[None, None, ...]


def _to_array(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _conv1d_ncw(x: jax.Array, kernel: jax.Array, padding_size: int) -> jax.Array:
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1,),
        padding=((padding_size, padding_size),),
        dimension_numbers=("NCW", "OIW", "NCW"),
    )


def _conv2d_nchw(x: jax.Array, kernel: jax.Array, padding_size: int) -> jax.Array:
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1, 1),
        padding=((padding_size, padding_size), (padding_size, padding_size)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def _max_pool3d_ncthw(x: jax.Array, window: tuple[int, int, int], stride: tuple[int, int, int]) -> jax.Array:
    return jax.lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, 1, *window),
        window_strides=(1, 1, *stride),
        padding="VALID",
    )


def _avg_pool3d_ncthw(x: jax.Array, kernel: tuple[int, int, int]) -> jax.Array:
    pooled = jax.lax.reduce_window(
        x,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, 1, *kernel),
        window_strides=(1, 1, *kernel),
        padding="VALID",
    )
    return pooled / float(np.prod(kernel))


class WeightedChannelSumLayer(nnx.Module):
    def __init__(self, init_channel_weights: tuple[float, ...], trainable: bool = False):
        init_weights = jnp.asarray(init_channel_weights, dtype=jnp.float32)
        self.channel_weights = nnx.Param(init_weights) if trainable else init_weights

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.shape[1] == 1:
            return x

        channel_weights = _to_array(self.channel_weights)
        weighted_input = x * channel_weights.reshape(1, -1, 1, 1, 1)
        return jnp.sum(weighted_input, axis=1, keepdims=True)


class Bias3DLayer(nnx.Module):
    def __init__(self, channels: int, initial: float = 0.0):
        self.bias = nnx.Param(jnp.full((1, channels, 1, 1, 1), initial, dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.bias.value


class BatchNorm3d(nnx.Module):
    def __init__(self, num_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=jnp.float32))
        self.running_mean = nnx.BatchStat(jnp.zeros((num_channels,), dtype=jnp.float32))
        self.running_var = nnx.BatchStat(jnp.ones((num_channels,), dtype=jnp.float32))

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if train:
            mean = jnp.mean(x, axis=(0, 2, 3, 4))
            var = jnp.var(x, axis=(0, 2, 3, 4))
            self.running_mean.value = (1.0 - self.momentum) * self.running_mean.value + self.momentum * mean
            self.running_var.value = (1.0 - self.momentum) * self.running_var.value + self.momentum * var
        else:
            mean = self.running_mean.value
            var = self.running_var.value

        mean = mean.reshape(1, -1, 1, 1, 1)
        var = var.reshape(1, -1, 1, 1, 1)
        scale = self.scale.value.reshape(1, -1, 1, 1, 1)
        bias = self.bias.value.reshape(1, -1, 1, 1, 1)
        return ((x - mean) / jnp.sqrt(var + self.eps)) * scale + bias


class Dropout3d(nnx.Module):
    def __init__(self, p: float, rngs: nnx.Rngs | None = None):
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.rngs = rngs

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        if (not train) or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        key = self.rngs.dropout() if (self.rngs is not None and hasattr(self.rngs, "dropout")) else jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)


class Laplace:
    def __init__(self, padding: int | None = None):
        self.filter = LAPLACE_3x3
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return _conv2d_nchw(x, self.filter, self.padding_size)


class Laplace1d:
    def __init__(self, padding: int | None):
        self.filter = LAPLACE_1D
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def __call__(self, x: jax.Array, avg: bool = False) -> jax.Array:
        agg_fn = jnp.mean if avg else jnp.sum
        return agg_fn(_conv1d_ncw(x, self.filter, self.padding_size))


class FlatLaplaceL23dnorm:
    def __init__(self, padding: int | None = None):
        self.laplace = Laplace(padding=padding)

    def __call__(self, x: jax.Array, avg: bool = False) -> jax.Array:
        agg_fn = jnp.mean if avg else jnp.sum
        oc, ic, k1, k2, k3 = x.shape
        if k1 != 1:
            raise ValueError("time dimension must be one")

        reshaped = x.reshape(oc * ic, 1, k2, k3)
        numerator = agg_fn(self.laplace(reshaped) ** 2)
        denominator = agg_fn(reshaped**2) + EPS
        return numerator / denominator


class _CoreLayer(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        padding: bool | int | str | tuple[int, int, int],
        dropout_rate: float,
        use_pool: bool,
        rngs: nnx.Rngs | None,
    ):
        self.conv = STSeparableBatchConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            log_speed_dict={},
            temporal_kernel_size=temporal_kernel_size,
            spatial_kernel_size=spatial_kernel_size,
            padding=padding,
            bias=False,
            rngs=rngs,
        )
        self.norm = BatchNorm3d(out_channels, momentum=0.1)
        self.bias = Bias3DLayer(out_channels)
        self.dropout = Dropout3d(dropout_rate, rngs=rngs) if dropout_rate > 0.0 else None
        self.use_pool = use_pool

    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x, train=train)
        x = self.bias(x)
        x = jax.nn.elu(x)

        if self.dropout is not None:
            x = self.dropout(x, train=train)

        if self.use_pool:
            x = _max_pool3d_ncthw(x, window=(1, 2, 2), stride=(1, 2, 2))

        return x


class Core(nnx.Module):
    def regularizer(self) -> jax.Array:
        warnings.warn(
            f"Regularizer not implemented for {self.__class__.__name__}", category=RuntimeWarning, stacklevel=2
        )
        return jnp.asarray(0.0, dtype=jnp.float32)

    def save_weight_visualizations(
        self,
        folder_path: str,
        file_format: str = "jpg",
        state_suffix: str = "",
    ) -> None:
        print(f"Save weight visualization of {self.__class__.__name__} not implemented.")


class SimpleCoreWrapper(Core):
    def __init__(
        self,
        channels: tuple[int, ...],
        temporal_kernel_sizes: tuple[int, ...],
        spatial_kernel_sizes: tuple[int, ...],
        gamma_input: float,
        gamma_temporal: float,
        gamma_in_sparse: float,
        gamma_hidden: float,
        dropout_rate: float = 0.0,
        cut_first_n_frames: int = 30,
        maxpool_every_n_layers: int | None = None,
        downsample_input_kernel_size: tuple[int, int, int] | None = None,
        input_padding: bool | int | str | tuple[int, int, int] = False,
        hidden_padding: bool | int | str | tuple[int, int, int] = True,
        color_squashing_weights: tuple[float, ...] | None = None,
        convolution_type: str = "custom_separable",
        n_neurons_dict: dict[str, int] | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        del n_neurons_dict  # compatibility arg to support Hydra config wiring

        if len(channels) < 2:
            raise ValueError(f"At least two channels required (input and output channel), {channels=}")
        if len(temporal_kernel_sizes) != len(channels) - 1:
            raise ValueError(
                f"{len(channels) - 1} layers, but only {len(temporal_kernel_sizes)} temporal kernel sizes."
            )
        if len(temporal_kernel_sizes) != len(spatial_kernel_sizes):
            raise ValueError("Temporal and spatial kernel sizes must have the same length.")
        if color_squashing_weights is not None and channels[0] != 1:
            raise ValueError(
                "Number of input channels must be 1 when squashing multi-channel input into single-channel."
            )

        if convolution_type != "custom_separable":
            raise NotImplementedError(
                "Phase 1 JAX core supports only convolution_type='custom_separable'."
            )

        self.convolution_type = convolution_type
        self.gamma_input = gamma_input
        self.gamma_temporal = gamma_temporal
        self.gamma_in_sparse = gamma_in_sparse
        self.gamma_hidden = gamma_hidden
        self._cut_first_n_frames = cut_first_n_frames
        self._downsample_input_kernel_size = (
            tuple(downsample_input_kernel_size) if downsample_input_kernel_size is not None else None
        )
        self.color_squashing_weights = color_squashing_weights

        if self._cut_first_n_frames and not input_padding:
            warnings.warn(
                (
                    "Cutting frames from the core output can lead to unexpected results if the input is not padded."
                    f"{self._cut_first_n_frames=}, {input_padding=}."
                ),
                UserWarning,
                stacklevel=2,
            )

        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=0)
        self._input_weights_regularizer_temporal = Laplace1d(padding=0)

        self.color_squashing_layer = (
            WeightedChannelSumLayer(self.color_squashing_weights)
            if self.color_squashing_weights is not None
            else None
        )

        layers: list[_CoreLayer] = []
        for layer_id, (num_in_channels, num_out_channels) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
            padding_to_use = input_padding if layer_id == 0 else hidden_padding
            if padding_to_use is True:
                padding: str | int | tuple[int, int, int] = "same"
            elif padding_to_use is False:
                padding = 0
            else:
                padding = padding_to_use

            layer = _CoreLayer(
                in_channels=num_in_channels,
                out_channels=num_out_channels,
                temporal_kernel_size=temporal_kernel_sizes[layer_id],
                spatial_kernel_size=spatial_kernel_sizes[layer_id],
                padding=padding,
                dropout_rate=dropout_rate,
                use_pool=maxpool_every_n_layers is not None and (layer_id % maxpool_every_n_layers) == 0,
                rngs=rngs,
            )
            setattr(self, f"layer{layer_id}", layer)
            layers.append(layer)

        self.features = tuple(layers)

    def _apply_features(self, x: jax.Array, train: bool = True) -> jax.Array:
        result = x
        for layer in self.features:
            result = layer(result, train=train)
        return result

    def __call__(self, input_: jax.Array, train: bool = True) -> jax.Array:
        if input_.ndim != 5:
            raise ValueError(f"Expected 5D input in NCTHW format, got shape {input_.shape}")

        if self.color_squashing_layer is not None:
            input_ = self.color_squashing_layer(input_)

        if self._downsample_input_kernel_size is not None:
            input_ = _avg_pool3d_ncthw(input_, self._downsample_input_kernel_size)

        res = self._apply_features(input_, train=train)
        return res[:, :, self._cut_first_n_frames :, :, :]

    def spatial_laplace(self) -> jax.Array:
        conv_obj = self.features[0].conv
        weight_spatial = conv_obj.weight_spatial.value
        return self._input_weights_regularizer_spatial(weight_spatial, avg=False)

    def temporal_laplace(self) -> jax.Array:
        raise NotImplementedError(
            "temporal_laplace is only used for convolution_type='separable', which is not part of phase 1."
        )

    def temporal_smoothness(self) -> jax.Array:
        if self.convolution_type == "custom_separable":
            results = [temporal_smoothing(x.conv.sin_weights.value, x.conv.cos_weights.value) for x in self.features]
            return jnp.sum(jnp.stack(results))

        raise ValueError(
            f"Temporal smoothness not supported for {self.convolution_type=}."
            "Set the temporal smoothness regularization weight to 0.0 to still use this conv type."
        )

    def group_sparsity_0(self) -> jax.Array:
        result_array: list[jax.Array] = []
        for layer in self.features:
            weight_spatial = layer.conv.weight_spatial.value
            numerator = jnp.sqrt(jnp.sum(weight_spatial**2, axis=(2, 3, 4))).sum(axis=1)
            denominator = jnp.sqrt(1e-8 + jnp.sum(weight_spatial**2, axis=(1, 2, 3, 4)))
            result_array.append(jnp.sum(numerator / denominator))

        return jnp.sum(jnp.stack(result_array))

    def group_sparsity(self) -> jax.Array:
        sparsities: list[jax.Array] = []
        for feat in self.features[1:]:
            weight_spatial = feat.conv.weight_spatial.value
            numerator = jnp.sqrt(jnp.sum(weight_spatial**2, axis=(2, 3, 4))).sum(axis=1)
            denominator = jnp.sqrt(1e-8 + jnp.sum(weight_spatial**2, axis=(1, 2, 3, 4)))
            sparsities.append(numerator / denominator)

        if len(sparsities) == 0:
            return jnp.asarray(0.0, dtype=jnp.float32)

        return jnp.sum(jnp.stack(sparsities))

    def regularizer(self) -> jax.Array:
        res = jnp.asarray(0.0, dtype=jnp.float32)
        for weight, reg_fn in [
            (self.gamma_input, self.spatial_laplace),
            (self.gamma_hidden, self.group_sparsity),
            (self.gamma_temporal, self.temporal_smoothness),
            (self.gamma_in_sparse, self.group_sparsity_0),
        ]:
            if weight != 0.0:
                res = res + weight * reg_fn()
        return res

    def plot_weight_visualization(self, layer: int, in_channel: int, out_channel: int) -> plt.Figure:
        if layer >= len(self.features):
            raise ValueError(f"Requested layer {layer}, but only {len(self.features)} layers present.")

        conv_obj = self.features[layer].conv
        return conv_obj.plot_weights(in_channel, out_channel)

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        for i, layer in enumerate(self.features):
            output_dir = os.path.join(folder_path, f"weights_layer_{i}")
            os.makedirs(output_dir, exist_ok=True)
            layer.conv.save_weight_visualizations(output_dir, file_format, state_suffix)
