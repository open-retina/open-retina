from builtins import int
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from openretina.modules.core.base_core import Core3d
from openretina.modules.layers import Bias3DLayer, Scale2DLayer, Scale3DLayer
from openretina.modules.layers.convolutions import (
    STSeparableBatchConv3d,
    TimeIndependentConv3D,
    TorchFullConv3D,
    TorchSTSeparableConv3D,
    compute_temporal_kernel,
    temporal_smoothing,
)
from openretina.modules.layers.gru import GRU_Module
from openretina.modules.layers.regularizers import FlatLaplaceL23dnorm, TimeLaplaceL23dnorm
from openretina.modules.layers.scaling import FiLM


class ConvGRUCore(Core3d, nn.Module):
    def __init__(
        self,
        n_neurons_dict: dict[str, int] | None = None,
        input_channels: int = 2,
        hidden_channels=(8,),
        temporal_kernel_size=(21,),
        spatial_kernel_size=(14,),
        layers: int = 1,
        gamma_hidden: float = 0.0,
        gamma_input: float = 0.0,
        gamma_in_sparse: float = 0.0,
        gamma_temporal: float = 0.0,
        final_nonlinearity: bool = True,
        bias: bool = True,
        input_padding: bool = False,
        hidden_padding: bool = True,
        batch_norm: bool = True,
        batch_norm_scale: bool = True,
        batch_norm_momentum: float = 0.1,
        laplace_padding: int | None = 0,
        batch_adaptation: bool = False,
        use_avg_reg: bool = False,
        nonlinearity: str = "ELU",
        conv_type: str = "custom_separable",
        use_gru: bool = False,
        use_projections: bool = False,
        gru_kwargs: dict[str, int | float] | None = None,
        **kwargs,
    ):
        super().__init__()
        # Set regularizers
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=laplace_padding)
        self._input_weights_regularizer_temporal = TimeLaplaceL23dnorm(padding=laplace_padding)

        # Get convolution class
        self.conv_class = self.get_conv_class(conv_type)

        if n_neurons_dict is None:
            n_neurons_dict = {}
            if batch_adaptation:
                raise ValueError(
                    "If batch_adaptation is True, n_neurons_dict must be provided to "
                    "learn the adaptation terms per session."
                )

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_in_sparse = gamma_in_sparse
        self.gamma_hidden = gamma_hidden
        self.gamma_temporal = gamma_temporal
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_avg_reg = use_avg_reg

        if not isinstance(hidden_channels, (list, tuple)):
            hidden_channels = [hidden_channels] * (self.layers)
        if not isinstance(temporal_kernel_size, (list, tuple)):
            temporal_kernel_size = [temporal_kernel_size] * (self.layers)
        if not isinstance(spatial_kernel_size, (list, tuple)):
            spatial_kernel_size = [spatial_kernel_size] * (self.layers)

        self.features = nn.Sequential()

        # Log speed dictionary
        log_speed_dict = self.generate_log_speed_dict(n_neurons_dict, batch_adaptation) if batch_adaptation else {}

        # Padding logic
        self.input_pad, self.hidden_pad = self.calculate_padding(input_padding, hidden_padding, spatial_kernel_size)

        # Initialize layers, including projection if applicable
        self.initialize_layers(
            input_channels,
            hidden_channels,
            temporal_kernel_size,
            spatial_kernel_size,
            log_speed_dict,
            batch_norm,
            batch_norm_momentum,
            bias,
            batch_norm_scale,
            final_nonlinearity,
            self.input_pad,
            self.hidden_pad,
            nonlinearity,
            use_projections,
        )

        self.apply(self.init_conv)

        # GRU integration
        if use_gru:
            print("Using GRU")
            self.features.add_module("gru", GRU_Module(**gru_kwargs))  # type: ignore

    def forward(self, input_, data_key=None):
        ret = []
        do_skip = False
        for layer_num, feat in enumerate(self.features):
            input_ = feat(
                (
                    torch.cat(ret[-min(self.skip, layer_num) :], dim=1) if do_skip else input_,
                    data_key,
                )
            )

        return input_

    def get_conv_class(self, conv_type: str) -> type[nn.Module]:
        if conv_type == "separable":
            return TorchSTSeparableConv3D
        elif conv_type == "custom_separable":
            return STSeparableBatchConv3d
        elif conv_type == "full":
            return TorchFullConv3D
        elif conv_type == "time_independent":
            return TimeIndependentConv3D
        else:
            raise ValueError(f"Un-implemented conv_type {conv_type}")

    def calculate_padding(self, input_padding, hidden_padding, spatial_kernel_size):
        if input_padding:
            input_pad = (0, spatial_kernel_size[0] // 2, spatial_kernel_size[0] // 2)
        else:
            input_pad = 0

        hidden_pad = [
            (0, spatial_kernel_size[x] // 2, spatial_kernel_size[x] // 2) if hidden_padding and x > 0 else 0
            for x in range(len(spatial_kernel_size))
        ]
        return input_pad, hidden_pad

    def initialize_layers(
        self,
        input_channels,
        hidden_channels,
        temporal_kernel_size,
        spatial_kernel_size,
        log_speed_dict,
        batch_norm,
        batch_norm_momentum,
        bias: bool,
        batch_norm_scale,
        final_nonlinearity,
        input_pad,
        hidden_pad,
        nonlinearity,
        use_projections: bool,
    ):
        layer: OrderedDict[str, Any] = OrderedDict()
        layer["conv"] = self.conv_class(
            input_channels,
            hidden_channels[0],
            log_speed_dict,
            temporal_kernel_size[0],
            spatial_kernel_size[0],
            bias=False,
            padding=input_pad,  # type: ignore
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm3d(
                hidden_channels[0],  # type: ignore
                momentum=batch_norm_momentum,
                affine=bias and batch_norm_scale,
            )  # ok or should we ensure same batch norm?
            if bias:
                if not batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels[0])
            elif batch_norm_scale:
                layer["scale"] = Scale3DLayer(hidden_channels[0])
        if final_nonlinearity:
            layer["nonlin"] = getattr(nn, nonlinearity)()
        self.features.add_module("layer0", nn.Sequential(layer))

        if use_projections:
            self.features.add_module(
                "projection",
                nn.Sequential(
                    TorchFullConv3D(
                        hidden_channels[0],
                        hidden_channels[0],
                        log_speed_dict,
                        1,
                        1,
                        bias=False,
                    ),
                    getattr(nn, nonlinearity)(),
                ),
            )

        # --- other layers

        for layer_num in range(1, self.layers):
            layer = OrderedDict()
            layer["conv"] = self.conv_class(
                hidden_channels[layer_num - 1],
                hidden_channels[layer_num],
                log_speed_dict,
                temporal_kernel_size[layer_num],
                spatial_kernel_size[layer_num],
                bias=False,
                padding=hidden_pad[layer_num - 1],  # type: ignore
            )
            if batch_norm:
                layer["norm"] = nn.BatchNorm3d(
                    hidden_channels[layer_num],
                    momentum=batch_norm_momentum,
                    affine=bias and batch_norm_scale,
                )
                if bias:
                    if not batch_norm_scale:
                        layer["bias"] = Bias3DLayer(hidden_channels[layer_num])
                elif batch_norm_scale:
                    layer["scale"] = Scale2DLayer(hidden_channels[layer_num])
            if final_nonlinearity or layer_num < self.layers - 1:
                layer["nonlin"] = getattr(nn, nonlinearity)()
            self.features.add_module(f"layer{layer_num}", nn.Sequential(layer))

    def generate_log_speed_dict(self, n_neurons_dict, batch_adaptation):
        log_speed_dict = {}
        for k in n_neurons_dict:
            var_name = "_".join(["log_speed", k])
            log_speed_val = torch.nn.Parameter(data=torch.zeros(1), requires_grad=batch_adaptation)
            setattr(self, var_name, log_speed_val)
            log_speed_dict[var_name] = log_speed_val
        return log_speed_dict

    def spatial_laplace(self):
        return self._input_weights_regularizer_spatial(self.features[0].conv.weight_spatial, avg=self.use_avg_reg)

    def group_sparsity(self):  # check if this is really what we want
        sparsity_loss = 0
        for layer in self.features:
            if hasattr(layer, "conv"):
                spatial_weight_layer = layer.conv.weight_spatial
                norm = spatial_weight_layer.pow(2).sum([2, 3, 4]).sqrt().sum(1)
                sparsity_loss_layer = (spatial_weight_layer.pow(2).sum([2, 3, 4]).sqrt().sum(1) / norm).sum()
                sparsity_loss += sparsity_loss_layer
            else:
                continue
        return sparsity_loss

    def group_sparsity0(self):  # check if this is really what we want
        weight_temporal = compute_temporal_kernel(
            torch.zeros(1, device=self.features[0].conv.sin_weights.device),
            self.features[0].conv.sin_weights,
            self.features[0].conv.cos_weights,
            self.features[0].conv.temporal_kernel_size,
        )
        # abc are dummy dimensions
        weight = torch.einsum("oitab,oichw->oithw", weight_temporal, self.features[0].conv.weight_spatial)
        return (weight.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(1e-8 + weight.pow(2).sum([1, 2, 3, 4]))).sum()

    def temporal_smoothness(self):
        return sum(
            temporal_smoothing(
                layer.conv.sin_weights,
                layer.conv.cos_weights,
            )
            for layer in self.features
            if hasattr(layer, "conv")
        )

    def regularizer(self):
        if self.conv_class == STSeparableBatchConv3d:
            return (
                self.group_sparsity() * self.gamma_hidden
                + self.gamma_input * self.spatial_laplace()
                + self.gamma_temporal * self.temporal_smoothness()
                + self.group_sparsity0() * self.gamma_in_sparse
            )
        else:
            return 0

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels[-1]


class ConditionedGRUCore(ConvGRUCore, nn.Module):
    def __init__(self, cond_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cond_dim = cond_dim

        j = 0
        for i, layer in enumerate(self.features):
            if hasattr(layer, "conv") and hasattr(layer, "norm"):
                setattr(self, f"film_{i}", FiLM(self.hidden_channels[j], self.cond_dim))
                # Manual instead of enumerate because of projection layers.
                j += 1

    def forward(self, x, conditioning=None, data_key=None):
        if conditioning is None:
            conditioning = torch.zeros(x.size(0), self.cond_dim, device=x.device)
        for layer_num, feat in enumerate(self.features):
            x = feat(
                (
                    x,
                    data_key,
                )
            )
            if hasattr(feat, "conv") and hasattr(feat, "norm"):
                x = getattr(self, f"film_{layer_num}")(x, conditioning)

        return x
