# ruff: noqa: E501  # for long link in this file, it didn't work to put it at that specific line for some reason
from collections import OrderedDict
from collections.abc import Iterable
from operator import itemgetter
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralpredictors import regularizers
from neuralpredictors.layers.affine import Bias3DLayer, Scale2DLayer, Scale3DLayer
from neuralpredictors.layers.readouts import (
    FullGaussian2d,
    Gaussian3d,
    MultiReadoutBase,
)
from neuralpredictors.utils import get_module_output

from .dataloaders import get_dims_for_loader_dict
from .hoefling_2024.models import (
    Core3d,
    Encoder,
    FlatLaplaceL23dnorm,
    STSeparableBatchConv3d,
    TimeIndependentConv3D,
    TimeLaplaceL23dnorm,
    TorchFullConv3D,
    TorchSTSeparableConv3D,
    compute_temporal_kernel,
    temporal_smoothing,
)
from .utils.misc import set_seed

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class RNNCore:
    """
    RNN Core taken from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight.data)
            if m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith("_") and "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class ConvGRUCell(RNNCore, nn.Module):
    """
    Convolutional GRU cell taken from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
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

        update = self.update_gate_input(input_) + self.update_gate_hidden(prev_state)
        update = F.sigmoid(update)

        reset = self.reset_gate_input(input_) + self.reset_gate_hidden(prev_state)
        reset = F.sigmoid(reset)

        out = self.out_gate_input(input_) + self.out_gate_hidden(prev_state * reset)
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update) + h_t * update

        return new_state

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self):
        return (
            self.reset_gate_hidden.bias.abs().mean() / 3
            + self.update_gate_hidden.weight.abs().mean() / 3
            + self.out_gate_hidden.bias.abs().mean() / 3
        )


class GRUEnabledCore(Core3d, nn.Module):
    def __init__(
        self,
        n_neurons_dict,
        input_channels=2,
        num_scans=1,
        hidden_channels=[8],
        temporal_kernel_size=[21],
        spatial_kernel_size=[14],
        layers=1,
        gamma_hidden=0.0,
        gamma_input=0.0,
        gamma_in_sparse=0.0,
        gamma_temporal=0.0,
        final_nonlinearity=True,
        bias=True,
        momentum=0.1,
        input_padding=False,
        hidden_padding=True,
        batch_norm=True,
        batch_norm_scale=True,
        laplace_padding: Optional[int] = 0,
        stack=None,
        batch_adaptation=True,
        use_avg_reg=False,
        nonlinearity="ELU",
        conv_type="custom_separable",
        use_gru=False,
        device=DEVICE,
        gru_kwargs: Optional[Dict[str, int | float]] = None,
    ):
        super().__init__()
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=laplace_padding)
        self._input_weights_regularizer_temporal = TimeLaplaceL23dnorm(padding=laplace_padding)

        if conv_type == "separable":
            self.conv_class = TorchSTSeparableConv3D  # type: ignore
        elif conv_type == "custom_separable":
            self.conv_class = STSeparableBatchConv3d  # type: ignore
        elif conv_type == "full":
            self.conv_class = TorchFullConv3D  # type: ignore
        elif conv_type == "time_independent":
            self.conv_class = TimeIndependentConv3D  # type: ignore
        else:
            raise ValueError(f"Un-implemented conv_type {conv_type}")

        self.num_scans = num_scans
        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_in_sparse = gamma_in_sparse
        self.gamma_hidden = gamma_hidden
        self.gamma_temporal = gamma_temporal
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_avg_reg = use_avg_reg

        self.features = nn.Sequential()
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [range(self.layers)[stack]] if isinstance(stack, int) else stack

        log_speed_dict = dict()
        for k in n_neurons_dict:
            var_name = "_".join(["log_speed", k])
            log_speed_val = torch.nn.Parameter(data=torch.zeros(1), requires_grad=batch_adaptation)
            setattr(self, var_name, log_speed_val)
            log_speed_dict[var_name] = log_speed_val

        if input_padding:
            input_pad: Tuple[int, ...] | int = (0, spatial_kernel_size[0] // 2, spatial_kernel_size[0] // 2)
        else:
            input_pad = 0
        if hidden_padding & (len(spatial_kernel_size) > 1):
            hidden_pad: list[Tuple[int, ...] | int] = [
                (0, spatial_kernel_size[x] // 2, spatial_kernel_size[x] // 2)
                for x in range(1, len(spatial_kernel_size))
            ]
        else:
            hidden_pad = [0 for _ in range(1, len(spatial_kernel_size))]

        if not isinstance(hidden_channels, (list, tuple)):
            hidden_channels = [hidden_channels] * (self.layers)
        if not isinstance(temporal_kernel_size, (list, tuple)):
            temporal_kernel_size = [temporal_kernel_size] * (self.layers)
        if not isinstance(spatial_kernel_size, (list, tuple)):
            spatial_kernel_size = [spatial_kernel_size] * (self.layers)

        # --- first layer
        layer: OrderedDict[str, Any] = OrderedDict()
        layer["conv"] = self.conv_class(
            input_channels,
            hidden_channels[0],
            log_speed_dict,
            temporal_kernel_size[0],
            spatial_kernel_size[0],
            bias=False,
            padding=input_pad,  # type: ignore
            num_scans=self.num_scans,
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm3d(
                hidden_channels[0],  # type: ignore
                momentum=momentum,
                affine=bias and batch_norm_scale,
            )  # ok or should we ensure same batch norm?
            if bias:
                if not batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels[0])
            elif batch_norm_scale:
                layer["scale"] = Scale3DLayer(hidden_channels[0])
        if final_nonlinearity:
            layer["nonlin"] = getattr(nn, nonlinearity)()  # TODO add back in place if necessary
        self.features.add_module("layer0", nn.Sequential(layer))  # type: ignore

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
                num_scans=self.num_scans,
            )
            if batch_norm:
                layer["norm"] = nn.BatchNorm3d(
                    hidden_channels[layer_num],
                    momentum=momentum,
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

        self.apply(self.init_conv)

        if use_gru:
            print("Using GRU")
            self.features.add_module("gru", GRU_Module(**gru_kwargs))  # type: ignore

    def forward(self, input_, data_key=None):
        ret = []
        for layer_num, feat in enumerate(self.features):
            do_skip = False
            input_ = feat(
                (
                    input_ if not do_skip else torch.cat(ret[-min(self.skip, layer_num) :], dim=1),
                    data_key,
                )
            )
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def spatial_laplace(self):
        return self._input_weights_regularizer_spatial(self.features[0].conv.weight_spatial, avg=self.use_avg_reg)

    def group_sparsity(self):  # check if this is really what we want
        sparsity_loss = 0
        for layer_num in range(1, self.layers):
            spatial_weight_layer = self.features[layer_num].conv.weight_spatial
            norm = spatial_weight_layer.pow(2).sum([2, 3, 4]).sqrt().sum(1)
            sparsity_loss_layer = (spatial_weight_layer.pow(2).sum([2, 3, 4]).sqrt().sum(1) / norm).sum()
            sparsity_loss += sparsity_loss_layer
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
        ret = (weight.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(1e-8 + weight.pow(2).sum([1, 2, 3, 4]))).sum()
        return ret

    def temporal_smoothness(self):
        ret = 0
        for layer_num in range(self.layers):
            ret += temporal_smoothing(
                self.features[layer_num].conv.sin_weights, self.features[layer_num].conv.cos_weights
            )
        return ret

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


class MultiGaussian3d(MultiReadoutBase):
    def __init__(self, in_shape_dict, n_neurons_dict, **kwargs):
        super().__init__(in_shape_dict, n_neurons_dict, base_readout=Gaussian3d, **kwargs)

    def regularizer(self, data_key):
        return 0


class MultiGaussian2d(MultiReadoutBase):
    def __init__(self, in_shape_dict, n_neurons_dict, **kwargs):
        super().__init__(
            in_shape_dict,
            n_neurons_dict,
            base_readout=FullGaussian2d,
            gauss_type="full",
            **kwargs,
        )


class VideoEncoder(Encoder):
    def forward(self, x, data_key=None, detach_core=False, **kwargs):
        self.detach_core = detach_core
        if self.detach_core:
            for name, param in self.core.features.named_parameters():
                if name.find("speed") < 0:
                    param.requires_grad = False

        x = self.core(x, data_key=data_key)

        # Make time the second dimension again for the readout
        x = torch.transpose(x, 1, 2)

        # Get dims for later reshaping
        batch_size = x.shape[0]
        time_points = x.shape[1]

        # Treat time as an indipendent (batch) dimension for the readout
        x = x.reshape(((-1,) + x.size()[2:]))
        x = self.readout(x, data_key=data_key, **kwargs)

        # Reshape back to the correct dimensions before returning
        x = x.reshape(((batch_size, time_points) + x.size()[1:]))

        # Add softplus as gaussian readout does not assure positive values
        x = F.softplus(x)
        return x


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


# Gaussian readout model
def SFB3d_core_gaussian_readout(
    dataloaders,
    seed,
    hidden_channels: Tuple[int] = (8,),  # core args
    temporal_kernel_size: Tuple[int] = (21,),
    spatial_kernel_size: Tuple[int] = (11,),
    layers: int = 1,
    gamma_hidden: float = 0,
    gamma_input: float = 0.1,
    gamma_temporal: float = 0.1,
    gamma_in_sparse=0.0,
    final_nonlinearity: bool = True,
    core_bias: bool = False,
    momentum: float = 0.1,
    input_padding: bool = False,
    hidden_padding: bool = True,
    batch_norm: bool = True,
    batch_norm_scale: bool = False,
    laplace_padding=None,
    batch_adaptation: bool = True,
    readout_scale: bool = False,
    readout_bias: bool = True,
    gaussian_masks: bool = False,  # readout args,
    gamma_readout: float = 0.1,
    gamma_masks: float = 0,
    gaussian_mean_scale: float = 1e0,
    gaussian_var_scale: float = 1e0,
    initialize_from_roi_masks: bool = False,
    readout_positive: bool = False,
    stack=None,
    readout_reg_avg: bool = False,
    use_avg_reg: bool = False,
    data_info: Optional[dict] = None,
    nonlinearity: str = "ELU",
    conv_type: Literal["full", "separable", "custom_separable", "time_independent"] = "custom_separable",
    device=DEVICE,
    use_gru: bool = False,
    gru_kwargs: dict = {},
):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per sessionin the format:
            {'train': {'session1': dataloader1, 'session2': dataloader2, ...},
             'validation': {'session1': dataloader1, 'session2': dataloader2, ...},
             'test': {'session1': dataloader1, 'session2': dataloader2, ...}}
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts
    Returns: An initialized model which consists of model.core and model.readout
    """

    # make sure trainloader is being used
    if data_info is not None:
        in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
        input_channels = [v["input_channels"] for k, v in data_info.items()]
        n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    else:
        dataloaders = dataloaders.get("train", dataloaders)

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name, *_ = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        print(session_shape_dict)
        n_neurons_dict = {
            k: v[out_name][-1] for k, v in session_shape_dict.items()
        }  # dictionary containing # neurons per session
        in_shapes_dict = {
            k: v[in_name] for k, v in session_shape_dict.items()
        }  # dictionary containing input shapes per session
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]  # gets the # of input channels
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    set_seed(seed)

    # get a stacked factorized 3d core from below
    core = GRUEnabledCore(
        n_neurons_dict=n_neurons_dict,
        input_channels=input_channels[0],
        num_scans=len(n_neurons_dict.keys()),
        hidden_channels=hidden_channels,
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        layers=layers,
        gamma_hidden=gamma_hidden,
        gamma_input=gamma_input,
        gamma_in_sparse=gamma_in_sparse,
        gamma_temporal=gamma_temporal,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        input_padding=input_padding,
        hidden_padding=hidden_padding,
        batch_norm=batch_norm,
        batch_norm_scale=batch_norm_scale,
        laplace_padding=laplace_padding,
        stack=stack,
        batch_adaptation=batch_adaptation,
        use_avg_reg=use_avg_reg,
        nonlinearity=nonlinearity,
        conv_type=conv_type,
        device=device,
        use_gru=use_gru,
        gru_kwargs=gru_kwargs,
    )

    in_shapes_readout = {}
    subselect = itemgetter(0, 2, 3)
    for k in n_neurons_dict:  # iterate over sessions
        in_shapes_readout[k] = subselect(tuple(get_module_output(core, in_shapes_dict[k])[1:]))

    readout = MultiGaussian2d(
        in_shape_dict=in_shapes_readout,
        n_neurons_dict=n_neurons_dict,
        scale=readout_scale,
        bias=readout_bias,
        feature_reg_weights=gamma_readout,
    )

    # initializing readout bias to mean response
    if readout_bias is True:
        if data_info is None:
            for k in dataloaders:
                readout[k].bias.data = dataloaders[k].dataset[:]._asdict()[out_name].mean(0)
        else:
            for k in data_info.keys():
                readout[k].bias.data = torch.from_numpy(data_info[k]["mean_response"])

    model = VideoEncoder(
        core,
        readout,
    )

    return model


# Baseline LNP:


class DummyCore(nn.Module):
    """
    A dummy core that does nothing. Used for readout only models, like the LNP model.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, data_key=None, **kwargs):
        return x

    def regularizer(self):
        return 0


class LNP(nn.Module):
    def __init__(
        self,
        in_shape,
        outdims,
        smooth_weight=0.0,
        sparse_weight=0.0,
        smooth_regularizer="LaplaceL2norm",
        laplace_padding=None,
        nonlinearity="exp",
        **kwargs,
    ):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight
        self.kernel_size = list(in_shape[3:])
        self.in_channels = in_shape[1]
        self.n_neurons = outdims
        self.nonlinearity = torch.__dict__[nonlinearity]

        self.inner_product = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_neurons,
            kernel_size=(1, *self.kernel_size),  # Not using time
            bias=False,
            stride=1,
        )

        if smooth_regularizer == "GaussianLaplaceL2":
            regularizer_config = dict(padding=laplace_padding, kernel=self.kernel_size)
        else:
            regularizer_config = dict(padding=laplace_padding)

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )

        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)

    def forward(self, x, data_key=None, **kwargs):
        x = self.inner_product(x)
        x = self.nonlinearity(x)
        x = torch.squeeze(x, dim=(3, 4))
        return x.transpose(1, 2)

    def weights_l1(self, average=True):
        """Returns l1 regularization across all weight dimensions

        Args:
            average (bool, optional): use mean of weights instad of sum. Defaults to True.
        """
        if average:
            return self.inner_product.weight.abs().mean()
        else:
            return self.inner_product.weight.abs().sum()

    def laplace(self):
        # Squeezing out the empty time dimension so we can use 2D regularizers
        return self._smooth_reg_fn(self.inner_product.weight.squeeze(2))

    def regularizer(self, **kwargs):
        return self.smooth_weight * self.laplace() + self.sparse_weight * self.weights_l1()

    def initialize(self, *args, **kwargs):
        pass


class MultipleLNP(Encoder):
    def __init__(self, in_shape_dict, n_neurons_dict, **kwargs):
        # The multiple LNP model acts like a readout reading directly from the videos
        readout = MultiReadoutBase(in_shape_dict, n_neurons_dict, base_readout=LNP, **kwargs)
        super().__init__(
            core=DummyCore(),
            readout=readout,
        )


class DenseReadout(nn.Module):
    """
    Fully connected readout layer.
    """

    def __init__(self, in_shape, outdims, bias=True, init_noise=1e-3, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.init_noise = init_noise
        c, w, h = in_shape

        self.linear = torch.nn.Linear(in_features=c * w * h, out_features=outdims, bias=False)
        if bias:
            bias = torch.nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def features(self):
        return next(iter(self.linear.parameters()))

    def feature_l1(self, average=False):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def regularizer(self, reduction="sum", average=False):
        return 0

    def initialize(self, *args, **kwargs):
        self.features.data.normal_(0, self.init_noise)

    def forward(self, x):
        b, c, w, h = x.shape

        x = x.view(b, c * w * h)
        y = self.linear(x)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return self.__class__.__name__ + " (" + "{} x {} x {}".format(*self.in_shape) + " -> " + str(self.outdims) + ")"


class MultipleDense(MultiReadoutBase):
    def __init__(
        self,
        in_shape_dict,
        n_neurons_dict,
        bias,
        init_noise,
    ):
        super().__init__(
            in_shape_dict,
            n_neurons_dict,
            base_readout=DenseReadout,
            bias=bias,
            init_noise=init_noise,
        )
