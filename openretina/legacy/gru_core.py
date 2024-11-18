from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from openretina.data_io.hoefling_2024 import (
    Core3d,
    FlatLaplaceL23dnorm,
    TimeIndependentConv3D,
    TorchFullConv3D,
    TorchSTSeparableConv3D,
    temporal_smoothing,
)
from openretina.modules.core.space_time_separable_conv import STSeparableBatchConv3d, compute_temporal_kernel
from openretina.modules.layers import Bias3DLayer, Scale2DLayer, Scale3DLayer
from openretina.modules.layers.laplace import TimeLaplaceL23dnorm


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    def __init__(
        self,
        input_channels,
        rec_channels,
        input_kern,
        rec_kern,
        groups=1,
        gamma_rec=0,
        pad_input=True,
        **kwargs,
    ):
        super().__init__()

        input_padding = input_kern // 2 if pad_input else 0
        rec_padding = rec_kern // 2

        self.rec_channels = rec_channels
        self._shrinkage = 0 if pad_input else input_kern - 1
        self.groups = groups

        self.gamma_rec = gamma_rec
        self.reset_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.reset_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.update_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.update_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.out_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.out_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.apply(self.init_conv)
        self.register_parameter("_prev_state", None)

    def init_state(self, input_):
        batch_size, _, *spatial_size = input_.data.size()
        state_size = [batch_size, self.rec_channels] + [s - self._shrinkage for s in spatial_size]
        prev_state = torch.zeros(*state_size)
        if input_.is_cuda:
            prev_state = prev_state.cuda()
        prev_state = nn.Parameter(prev_state)
        return prev_state

    def forward(self, input_, prev_state):
        # get batch and spatial sizes

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(input_)

        update_gate = self.update_gate_input(input_) + self.update_gate_hidden(prev_state)
        update_gate = F.sigmoid(update_gate)

        reset_gate = self.reset_gate_input(input_) + self.reset_gate_hidden(prev_state)
        reset_gate = F.sigmoid(reset_gate)

        out = self.out_gate_input(input_) + self.out_gate_hidden(prev_state * reset_gate)
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update_gate) + h_t * update_gate

        return new_state

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self):
        return (
            self.reset_gate_hidden.bias.abs().mean() / 3  # type: ignore
            + self.update_gate_hidden.weight.abs().mean() / 3
            + self.out_gate_hidden.bias.abs().mean() / 3  # type: ignore
        )

    def __repr__(self):
        s = super().__repr__()
        s += f" [{self.__class__.__name__} regularizers: "
        ret = [
            f"{attr} = {getattr(self, attr)}"
            for attr in filter(lambda x: not x.startswith("_") and "gamma" in x, dir(self))
        ]
        return s + "|".join(ret) + "]\n"

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight.data)
            if m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)


class GRU_Module(nn.Module):
    def __init__(
        self,
        input_channels,
        rec_channels,
        input_kern,
        rec_kern,
        groups=1,
        gamma_rec=0,
        pad_input=True,
        **kwargs,
    ):
        """
        A GRU module for video data to add between the core and the readout.
        Receives as input the output of a 3Dcore. Expected dimensions:
            - (Batch, Channels, Frames, Height, Width) or (Channels, Frames, Height, Width)
        The input is fed sequentially to a convolutional GRU cell, based on the frames channel.
        The output has the same dimensions as the input.
        """
        super().__init__()
        self.gru = ConvGRUCell(
            input_channels,
            rec_channels,
            input_kern,
            rec_kern,
            groups=groups,
            gamma_rec=gamma_rec,
            pad_input=pad_input,
        )

    def forward(self, input_):
        """
        Forward pass definition based on
        https://github.com/sinzlab/Sinz2018_NIPS/blob/3a99f7a6985ae8dec17a5f2c54f550c2cbf74263/nips2018/architectures/cores.py#L556
        Modified to also accept 4 dimensional inputs (assuming no batch dimension is provided).
        """
        x, data_key = input_
        if len(x.shape) not in [4, 5]:
            raise RuntimeError(
                f"Expected 4D (unbatched) or 5D (batched) input to ConvGRUCell, but got input of size: {x.shape}"
            )

        batch = True
        if len(x.shape) == 4:
            batch = False
            x = torch.unsqueeze(x, dim=0)

        states = []
        hidden = None
        frame_pos = 2

        for frame in range(x.shape[frame_pos]):
            slice_channel = [frame if frame_pos == i else slice(None) for i in range(len(x.shape))]
            hidden = self.gru(x[slice_channel], hidden)
            states.append(hidden)
        out = torch.stack(states, frame_pos)
        if not batch:
            out = torch.squeeze(out, dim=0)
        return out


class ConvGRUCore(Core3d, nn.Module):
    def __init__(
        self,
        n_neurons_dict: Optional[Dict[str, int]] = None,
        input_channels=2,
        hidden_channels=(8,),
        temporal_kernel_size=(21,),
        spatial_kernel_size=(14,),
        layers=1,
        gamma_hidden=0.0,
        gamma_input=0.0,
        gamma_in_sparse=0.0,
        gamma_temporal=0.0,
        final_nonlinearity=True,
        bias=True,
        input_padding=False,
        hidden_padding=True,
        batch_norm=True,
        batch_norm_scale=True,
        batch_norm_momentum=0.1,
        laplace_padding: Optional[int] = 0,
        batch_adaptation=False,
        use_avg_reg=False,
        nonlinearity="ELU",
        conv_type="custom_separable",
        use_gru=False,
        use_projections=False,
        gru_kwargs: Optional[Dict[str, int | float]] = None,
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
        use_projections,
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


class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) is a neural network module that applies
    conditional scaling and shifting to input features.

    This module takes input features and a conditioning tensor, computes scaling (gamma)
    and shifting (beta) parameters from the conditioning tensor, and applies these parameters to
    the input features. The result is a modulated output that can adapt based on the provided conditions.

    Args:
        num_features (int): The number of features in the input tensor.
        cond_dim (int): The dimensionality of the conditioning tensor.

    Returns:
        Tensor: The modulated output tensor after applying the scaling and shifting.
    """

    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim

        self.fc_gamma = nn.Linear(cond_dim, num_features)
        self.fc_beta = nn.Linear(cond_dim, num_features)

        # To avoid perturbations in early epochs, we set these defaults to match the identity function
        nn.init.normal_(self.fc_gamma.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_gamma.bias, 1.0)

        nn.init.normal_(self.fc_beta.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_beta.bias, 0.0)

    def forward(self, x, cond):
        # View the conditioning tensor to match the input tensor shape
        gamma = self.fc_gamma(cond).view(cond.size(0), self.num_features, *[1] * (x.dim() - 2))
        beta = self.fc_beta(cond).view(cond.size(0), self.num_features, *[1] * (x.dim() - 2))

        return gamma * x + beta


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
