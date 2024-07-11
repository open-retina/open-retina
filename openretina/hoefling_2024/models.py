from collections import OrderedDict
from collections.abc import Iterable
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralpredictors.layers.affine import Bias3DLayer, Scale2DLayer, Scale3DLayer
from neuralpredictors.regularizers import Laplace, Laplace1d
from neuralpredictors.utils import get_module_output

from ..dataloaders import get_dims_for_loader_dict
from ..utils.misc import set_seed

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Core(nn.Module):
    def initialize(self) -> None:
        raise NotImplementedError("Not initializing")

    def __repr__(self) -> str:
        s = f"{repr(super())} [{self.__class__.__name__} regularizers: "
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append(f"{attr} = {getattr(self, attr)}")
        return s + "|".join(ret) + "]\n"


class Core3d(Core):
    def initialize(self) -> None:
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


class ParametricFactorizedBatchConv3dCore(Core3d):
    def __init__(
        self,
        n_neurons_dict,
        input_channels: int = 2,
        num_scans: int = 1,
        hidden_channels=[8],
        temporal_kernel_size=[21],
        spatial_kernel_size=[14],
        layers: int = 1,
        gamma_hidden: float = 0.0,
        gamma_input: float = 0.0,
        gamma_in_sparse: float = 0.0,
        gamma_temporal: float = 0.0,
        final_nonlinearity: bool = True,
        bias: bool = True,
        momentum: float = 0.1,
        input_padding: bool = False,
        hidden_padding: bool = True,
        batch_norm: bool = True,
        batch_norm_scale: bool = True,
        laplace_padding: int = 0,
        stack=None,
        batch_adaptation: bool = True,
        use_avg_reg: bool = False,
        nonlinearity: str = "ELU",
        conv_type: str = "custom_separable",
    ):
        super().__init__()
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=laplace_padding)
        self._input_weights_regularizer_temporal = TimeLaplaceL23dnorm(padding=laplace_padding)

        if conv_type == "separable":
            self.conv_class: type[torch.nn.Module] = TorchSTSeparableConv3D
        elif conv_type == "custom_separable":
            self.conv_class = STSeparableBatchConv3d
        elif conv_type == "full":
            self.conv_class = TorchFullConv3D
        elif conv_type == "time_independent":
            self.conv_class = TimeIndependentConv3D
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
            var_name = "_".join(["log_speed", str(k)])
            log_speed_val = torch.nn.Parameter(data=torch.zeros(1), requires_grad=batch_adaptation)
            setattr(self, var_name, log_speed_val)
            log_speed_dict[var_name] = log_speed_val

        if input_padding:
            input_pad: tuple[int, int, int] | int = (0, spatial_kernel_size[0] // 2,
                                                     spatial_kernel_size[0] // 2)
        else:
            input_pad = 0
        if hidden_padding & (len(spatial_kernel_size) > 1):
            hidden_pad: list[tuple[int, int, int] | int] = [
                (0, spatial_kernel_size[x] // 2, spatial_kernel_size[x] // 2)
                for x in range(1, len(spatial_kernel_size))
            ]
        else:
            hidden_pad = [0 for _ in range(1, len(spatial_kernel_size))]

        if not isinstance(hidden_channels, Iterable):
            hidden_channels = [hidden_channels] * self.layers
        if not isinstance(temporal_kernel_size, Iterable):
            temporal_kernel_size = [temporal_kernel_size] * self.layers
        if not isinstance(spatial_kernel_size, Iterable):
            spatial_kernel_size = [spatial_kernel_size] * self.layers

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = self.conv_class(
            input_channels,
            hidden_channels[0],
            log_speed_dict,
            temporal_kernel_size[0],
            spatial_kernel_size[0],
            bias=False,
            padding=input_pad,
            num_scans=self.num_scans,
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm3d(
                hidden_channels[0], momentum=momentum, affine=bias and batch_norm_scale
            )  # ok or should we ensure same batch norm?
            if bias:
                if not batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels[0])
            elif batch_norm_scale:
                layer["scale"] = Scale3DLayer(hidden_channels[0])
        if final_nonlinearity:
            layer["nonlin"] = getattr(nn, nonlinearity)()
        self.features.add_module("layer0", nn.Sequential(layer))

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
                padding=hidden_pad[layer_num - 1],
                num_scans=self.num_scans,
            )
            if batch_norm:
                layer["norm"] = nn.BatchNorm3d(hidden_channels[layer_num], momentum=momentum,
                                               affine=bias and batch_norm_scale)
                if bias:
                    if not batch_norm_scale:
                        layer["bias"] = Bias3DLayer(hidden_channels[layer_num])
                elif batch_norm_scale:
                    layer["scale"] = Scale2DLayer(hidden_channels[layer_num])
            if final_nonlinearity or layer_num < self.layers - 1:
                layer["nonlin"] = getattr(nn, nonlinearity)()
            self.features.add_module("layer{}".format(layer_num), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_, data_key=None):
        ret = []
        do_skip = False
        for layer_num, feat in enumerate(self.features):
            input_ = feat((torch.cat(ret[-min(self.skip, layer_num):], dim=1)
                           if do_skip else input_, data_key))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def spatial_laplace(self):
        return self._input_weights_regularizer_spatial(self.features[0].conv.weight_spatial, avg=self.use_avg_reg)

    def group_sparsity(self):  # check if this is really what we want
        ret = 0
        for layer_num in range(1, self.layers):
            ret = (
                ret
                + (
                    self.features[layer_num].conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1)
                    / torch.sqrt(1e-8 + self.features[layer_num].conv.weight_spatial.pow(2).sum([1, 2, 3, 4]))
                ).sum()
            )
        return ret

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
        for layer_norm in range(self.layers):
            ret += temporal_smoothing(self.features[layer_norm].conv.sin_weights,
                                      self.features[layer_norm].conv.cos_weights)
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
    def outchannels(self) -> list[int]:
        return len(self.features) * self.hidden_channels[-1]


class SpatialXFeature3dReadout(nn.ModuleDict):
    def __init__(
        self,
        core: Core,
        in_shape_dict,
        n_neurons_dict,
        scale,
        bias,
        gaussian_masks,
        gaussian_mean_scale,
        gaussian_var_scale,
        positive,
        initialize_from_roi_masks,
        roi_masks,
        gamma_readout,
        gamma_masks=0.0,
        readout_reg_avg=False,
    ):
        super().__init__()
        for k in n_neurons_dict:  # iterate over sessions
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                str(k),
                SpatialXFeature3d(  # add a readout for each session
                    in_shape,
                    n_neurons,
                    gaussian_masks=gaussian_masks,
                    gaussian_mean_scale=gaussian_mean_scale,
                    gaussian_var_scale=gaussian_var_scale,
                    initialize_from_roi_masks=initialize_from_roi_masks,
                    roi_mask=roi_masks[k],
                    positive=positive,
                    scale=scale,
                    bias=bias,
                ),
            )

        self.gamma_readout = gamma_readout
        self.gamma_masks = gamma_masks
        self.gaussian_masks = gaussian_masks
        self.readout_reg_avg = readout_reg_avg

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        ret = self[data_key].feature_l1(average=self.readout_reg_avg) * self.gamma_readout
        ret = ret + self[data_key].mask_l1(average=self.readout_reg_avg) * self.gamma_masks
        return ret


class SpatialXFeature3d(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        outdims: int,
        gaussian_masks: bool = False,
        gaussian_mean_scale: float = 1e0,
        gaussian_var_scale: float = 1e0,
        initialize_from_roi_masks: bool = False,
        roi_mask=[],
        positive: bool = False,
        scale: bool = False,
        bias: bool = True,
        nonlinearity: bool = True,
    ):
        """
        TODO write docstring

        Args:
            in_shape (tuple): The shape of the input tensor (c, t, w, h).
            outdims (int): The number of output dimensions (usually the number of neurons in the session).
            gaussian_masks (bool, optional): Whether to use Gaussian masks. Defaults to False.
            gaussian_mean_scale (float, optional): The scale factor for the Gaussian mean. Defaults to 1e0.
            gaussian_var_scale (float, optional): The scale factor for the Gaussian variance. Defaults to 1e0.
            initialize_from_roi_masks (bool, optional): Whether to initialize from ROI masks. Defaults to False.
            roi_mask (list, optional): The ROI mask. Defaults to an empty list.
            positive (bool, optional): Whether the output should be positive. Defaults to False.
            scale (bool, optional): Whether to include a scale parameter. Defaults to False.
            bias (bool, optional): Whether to include a bias parameter. Defaults to True.
            nonlinearity (bool, optional): Whether to include a nonlinearity. Defaults to True.
        """
        super().__init__()
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.gaussian_masks = gaussian_masks
        self.gaussian_mean_scale = gaussian_mean_scale
        self.gaussian_var_scale = gaussian_var_scale
        self.positive = positive
        self.nonlinearity = nonlinearity
        self.initialize_from_roi_masks = initialize_from_roi_masks

        if gaussian_masks:
            """we train on the log var and transform to var in a separate step"""
            self.mask_mean = torch.nn.Parameter(data=torch.zeros(self.outdims, 2), requires_grad=True)
            self.mask_log_var = torch.nn.Parameter(data=torch.zeros(self.outdims), requires_grad=True)
            self.grid = torch.nn.Parameter(data=self.make_mask_grid(w, h), requires_grad=False)
            self.masks = self.normal_pdf().permute(1, 2, 0)
        else:
            if initialize_from_roi_masks:
                self.mask_mean = torch.nn.Parameter(data=roi_mask, requires_grad=False)
                self.mask_var = torch.nn.Parameter(data=torch.ones(self.outdims) * 0.01, requires_grad=False)
                self.grid = torch.nn.Parameter(data=self.make_mask_grid(w, h), requires_grad=False)
                self.masks = nn.Parameter(self.get_normal_pdf_from_roi_mask().permute(1, 2, 0))
            else:
                self.masks = nn.Parameter(torch.Tensor(w, h, outdims))

        self.features = nn.Parameter(torch.Tensor(1, c, 1, outdims))

        if scale:
            scale_param = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("scale", scale_param)
        else:
            self.register_parameter("scale", None)

        if bias:
            bias_param = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias_param)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    def initialize(self) -> None:
        if (not self.gaussian_masks) and (not self.initialize_from_roi_masks):
            self.masks.data.normal_(0.0, 0.01)
        self.features.data.normal_(0.0, 0.01)
        if self.scale is not None:
            self.scale.data.normal_(1.0, 0.01)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average: bool = False, subs_idx=None) -> torch.Tensor:
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return self.features[..., subs_idx].abs().mean()
        else:
            return self.features[..., subs_idx].abs().sum()

    def mask_l1(self, average: bool = False, subs_idx=None) -> torch.Tensor:
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if self.gaussian_masks:
            if average:
                return (
                    torch.exp(self.mask_log_var * self.gaussian_var_scale)[..., subs_idx].mean()
                    + (self.mask_mean[..., subs_idx] * self.gaussian_mean_scale).pow(2).mean()
                )
            else:
                return (
                    torch.exp(self.mask_log_var * self.gaussian_var_scale)[..., subs_idx].sum()
                    + (self.mask_mean[..., subs_idx] * self.gaussian_mean_scale).pow(2).sum()
                )
        else:
            if average:
                return self.masks[..., subs_idx].abs().mean()
            else:
                return self.masks[..., subs_idx].abs().sum()

    def make_mask_grid(self, w: int, h: int) -> torch.Tensor:
        """Actually mixed up: w (width) is height, and vice versa"""
        grid_w = torch.linspace(-1 * w / max(w, h), 1 * w / max(w, h), w)
        grid_h = torch.linspace(-1 * h / max(w, h), 1 * h / max(w, h), h)
        xx, yy = torch.meshgrid([grid_w, grid_h], indexing="ij")
        grid = torch.stack([xx, yy], 2)[None, ...]
        return grid.repeat([self.outdims, 1, 1, 1])

    def normal_pdf(self) -> torch.Tensor:
        """Gets the actual mask values in terms of a PDF from the mean and SD"""
        # self.mask_var_ = torch.exp(self.mask_log_var * self.gaussian_var_scale).view(-1, 1, 1)
        scaled_log_var = self.mask_log_var * self.gaussian_var_scale
        self.mask_var_ = torch.exp(torch.clamp(scaled_log_var, min=-20, max=20)).view(-1, 1, 1)
        pdf = self.grid - self.mask_mean.view(self.outdims, 1, 1, -1) * self.gaussian_mean_scale
        pdf = torch.sum(pdf**2, dim=-1) / (self.mask_var_ + 1e-8)
        pdf = torch.exp(-0.5 * torch.clamp(pdf, max=20))
        normalisation = torch.sum(pdf, dim=(1, 2), keepdim=True)
        pdf = torch.nan_to_num(pdf / normalisation)
        return pdf

    def forward(self, x: torch.Tensor, shift=None, subs_idx=None) -> torch.Tensor:
        if self.gaussian_masks:
            self.masks = self.normal_pdf().permute(1, 2, 0)
        else:
            self.masks.data.abs_()

        if self.positive:
            self.features.data.clamp_(0)

        N, c, t, w, h = x.size()
        if subs_idx is not None:
            feat = self.features[..., subs_idx]
            masks = self.masks[..., subs_idx]

        else:
            feat = self.features
            masks = self.masks

        y = torch.einsum("nctwh,whd->nctd", x, masks)
        y = (y * feat).sum(1)

        if self.scale is not None:
            y = y * self.scale
        if self.bias is not None:
            if subs_idx is None:
                y = y + self.bias
            else:
                y = y + self.bias[subs_idx]
        if self.nonlinearity:
            y = F.softplus(y)
        return y

    def __repr__(self) -> str:
        c, _, w, h = self.in_shape
        res_array: list[str] = []
        r = f"{self.__class__.__name__} ( {c} x {w} x {h} -> {str(self.outdims)})"
        if self.bias is not None:
            r += " with bias"
        res_array.append(r)

        for ch in self.children():
            r += "  -> " + ch.__repr__()
            res_array.append(r)
        return "\n".join(res_array)


class Encoder(nn.Module):
    """
    puts together all parts of model (core, readouts) and defines a forward
    function which will return the output of the model; PyTorch then allows
    to call .backward() on the Encoder which will compute the gradients
    """

    def __init__(
        self,
        core: Core,
        readout,
    ):
        super().__init__()
        self.core = core
        self.readout = readout
        self.detach_core = False

    def forward(self, x: torch.Tensor, data_key: str | None = None, detach_core: bool = False, **kwargs):
        self.detach_core = detach_core
        x = self.core(x, data_key=data_key)
        if self.detach_core:
            x = x.detach()
        x = self.readout(x, data_key=data_key)
        return x


def compute_temporal_kernel(log_speed, sin_weights, cos_weights, length: int) -> torch.Tensor:
    """
    Computes the temporal kernel for the convolution.

    Args:
        log_speed (torch.nn.Parameter): Logarithm of the speed factor.
        sin_weights (torch.nn.Parameter): Sinusoidal weights.
        cos_weights (torch.nn.Parameter): Cosine weights.
        length (int): Length of the temporal kernel.

    Returns:
        torch.Tensor: The temporal kernel.
    """
    stretches = torch.exp(log_speed)
    sines, cosines = STSeparableBatchConv3d.temporal_basis(stretches, length)
    weights_temporal = torch.sum(sin_weights[:, :, :, None] * sines[None, None, ...], dim=2) + torch.sum(
        cos_weights[:, :, :, None] * cosines[None, None, ...], dim=2
    )
    return weights_temporal[..., None, None]


class TorchFullConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        num_scans=1,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        self._log_speed_default = torch.nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (temporal_kernel_size, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x, data_key = input_

        # Compute temporal kernel based on the provided data key
        # TODO implement log speed use in full conv
        # if data_key is None:
        #    log_speed = self._log_speed_default
        # else:
        #    log_speed = getattr(self, "_".join(["log_speed", data_key]))

        return self.conv(x)


class TimeIndependentConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_):
        x, data_key = input_
        return self.conv(x)


class TorchSTSeparableConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        num_scans=1,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        self._log_speed_default = torch.nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.space_conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.time_conv = nn.Conv3d(
            out_channels, out_channels, (temporal_kernel_size, 1, 1), stride=stride, padding=padding, bias=bias
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x, data_key = input_

        # Compute temporal kernel based on the provided data key
        if data_key is None:
            log_speed = self._log_speed_default
        else:
            log_speed = getattr(self, "_".join(["log_speed", data_key]))

        space_conv = self.space_conv(x)

        return torch.exp(log_speed) * self.time_conv(space_conv)


class STSeparableBatchConv3d(nn.Module):
    """
    Spatio-temporal separable convolution layer for processing 3D data.

    This layer applies convolution separately in the spatial and temporal dimensions,
    which is efficient for spatio-temporal data like video or medical images.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        temporal_kernel_size (int): Size of the kernel in the temporal dimension.
        spatial_kernel_size (int): Size of the kernel in the spatial dimensions.
        spatial_kernel_size2 (int): Size of the kernel in the second spatial dimension.
        stride (int): Stride of the convolution.
        padding (int): Padding added to all sides of the input.
        num_scans (int): Number of scans for batch processing.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: int | None = None,
        stride: int = 1,
        padding: int = 0,
        num_scans: int = 1,
        bias: bool = True,
    ):
        """
        Initializes the STSeparableBatchConv3d layer.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            log_speed_dict (dict): Dictionary mapping data keys to log speeds.
            temporal_kernel_size (int): Size of the temporal kernel.
            spatial_kernel_size (int): Size of the spatial kernel.
            spatial_kernel_size2 (int, optional): Size of the second spatial dimension of the kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to all sides of the input. Defaults to 0.
            num_scans (int, optional): Number of scans to process in batch. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_kernel_size2 = spatial_kernel_size2 if spatial_kernel_size2 is not None else spatial_kernel_size
        self.stride = stride
        self.padding = padding
        self.num_scans = num_scans

        # Initialize temporal weights
        self.sin_weights, self.cos_weights = self.temporal_weights(temporal_kernel_size, in_channels, out_channels)

        # Initialize spatial weights
        self.weight_spatial = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, self.spatial_kernel_size, self.spatial_kernel_size2) * 0.01
        )

        # Initialize bias if required
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._log_speed_default = torch.nn.Parameter(data=torch.zeros(1), requires_grad=False)

        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

    def forward(self, input_: tuple[torch.Tensor, str]) -> torch.Tensor:
        """
        Forward pass of the STSeparableBatchConv3d layer.

        Args:
            input_ (tuple): Tuple containing the input tensor and the data key.

        Returns:
            torch.Tensor: The output of the convolution.
        """
        x, data_key = input_

        # Compute temporal kernel based on the provided data key
        if data_key is None:
            log_speed = self._log_speed_default
        else:
            log_speed = getattr(self, "_".join(["log_speed", data_key]))
        self.weight_temporal = compute_temporal_kernel(
            log_speed, self.sin_weights, self.cos_weights, self.temporal_kernel_size
        )

        # Assemble the complete weight tensor for convolution
        # o - output channels, i - input channels, t - temporal kernel size
        # x - empty dimension, h - spatial kernel size, w - second spatial kernel size
        self.weight = torch.einsum("oitxx,oixhw->oithw", self.weight_temporal, self.weight_spatial)

        # Perform the convolution
        self.conv = F.conv3d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return self.conv

    @staticmethod
    def temporal_weights(length: int, num_channels: int, num_feat: int, scale: float = 0.01
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates initial weights for the temporal components of the convolution.

        Args:
            length (int): Length of the temporal kernel.
            num_channels (int): Number of input channels.
            num_feat (int): Number of output features.
            scale (float, optional): Scaling factor for weight initialization. Defaults to 0.01.

        Returns:
            tuple: Tuple containing sin and cos weights.
        """
        K = length // 3
        sin_weights = torch.nn.Parameter(data=torch.randn(num_feat, num_channels, K) * scale, requires_grad=True)
        cos_weights = torch.nn.Parameter(data=torch.randn(num_feat, num_channels, K) * scale, requires_grad=True)
        return sin_weights, cos_weights

    @staticmethod
    def temporal_basis(stretches: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the basis for the temporal component of the convolution.

        Args:
            stretches (torch.Tensor): Temporal stretches per ROI.
            T (int): Length of the temporal kernel.

        Returns:
            tuple: Tuple containing sines and cosines tensors.
        """
        K = T // 3
        time = torch.arange(T, dtype=torch.float, device=stretches.device) - T
        stretched = stretches * time
        freq = stretched * 2 * np.pi / T
        mask = STSeparableBatchConv3d.mask_tf(time, stretches, T)
        sines, cosines = [], []
        for k in range(K):
            sines.append(mask * torch.sin(freq * k))
            cosines.append(mask * torch.cos(freq * k))
        sines_stacked = torch.stack(sines, 0)
        cosines_stacked = torch.stack(cosines, 0)
        return sines_stacked, cosines_stacked

    @staticmethod
    def mask_tf(time: torch.Tensor, stretch: torch.Tensor, T: int) -> torch.Tensor:
        """
        Generates a mask for the temporal basis functions.

        Args:
            time (torch.Tensor): Time tensor.
            stretch (torch.Tensor): Stretch tensor.
            T (int): Length of the temporal kernel.

        Returns:
            torch.Tensor: The mask tensor.
        """
        mask = 1 / (1 + torch.exp(-time - int(T * 0.95) / stretch))
        return mask.T


class TimeLaplaceL23dnorm(nn.Module):
    """
    Normalized Laplace regularizer for the temporal component of a separable 3D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.laplace = Laplace1d(padding=padding)

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2, k3 = x.size()
        assert (k2, k3) == (1, 1), "space dimensions must be one"
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1)).pow(2)) / agg_fn(x.view(oc * ic, 1, k1).pow(2))


class FlatLaplaceL23dnorm(nn.Module):
    """
    Normalized Laplace regularizer for the spatial component of a separable 3D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2, k3 = x.size()
        assert k1 == 1, "time dimension must be one"
        return agg_fn(self.laplace(x.view(oc * ic, 1, k2, k3)).pow(2)) / agg_fn(x.view(oc * ic, 1, k2, k3).pow(2))


def temporal_smoothing(sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    smoother = torch.linspace(0.1, 0.9, sin.shape[2], device=sin.device)[None, None, :]
    F = float(sin.shape[0])
    reg = torch.sum((smoother * sin) ** 2) / F
    reg += torch.sum((smoother * cos) ** 2) / F
    return reg


class LocalEncoder(Encoder):

    def forward(
            self,
            x: torch.Tensor,
            data_key: str | None = None,
            detach_core: bool = False,
            **kwargs
    ) -> torch.Tensor:
        self.detach_core = detach_core
        if self.detach_core:
            for name, param in self.core.features.named_parameters():
                if name.find("speed") < 0:
                    param.requires_grad = False
        x = self.core(x, data_key=data_key)
        x = self.readout(x, data_key=data_key)
        return x


# Batch adaption model
def SFB3d_core_SxF3d_readout(
    dataloaders,
    seed: int | None = None,
    hidden_channels: Tuple[int] = (8,),  # core args
    temporal_kernel_size: Tuple[int] = (21,),
    spatial_kernel_size: Tuple[int] = (11,),
    layers: int = 1,
    gamma_hidden: float = 0,
    gamma_input: float = 0.1,
    gamma_temporal: float = 0.1,
    gamma_in_sparse: float = 0.0,
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
) -> torch.nn.Module:
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
        roi_masks = {k: torch.tensor(v["roi_coords"]) for k, v in data_info.items()}
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
        roi_masks = {k: dataloaders[k].dataset.roi_coords for k in dataloaders.keys()}
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    set_seed(seed)

    # get a stacked factorized 3d core from below
    core = ParametricFactorizedBatchConv3dCore(
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
    )

    readout = SpatialXFeature3dReadout(
        core,
        in_shape_dict=in_shapes_dict,
        n_neurons_dict=n_neurons_dict,
        scale=readout_scale,
        bias=readout_bias,
        gaussian_masks=gaussian_masks,
        gaussian_mean_scale=gaussian_mean_scale,
        gaussian_var_scale=gaussian_var_scale,
        positive=readout_positive,
        initialize_from_roi_masks=initialize_from_roi_masks,
        roi_masks=roi_masks,
        gamma_readout=gamma_readout,
        gamma_masks=gamma_masks,
        readout_reg_avg=readout_reg_avg,
    )

    # initializing readout bias to mean response
    if readout_bias:
        if data_info is None:
            for k in dataloaders:
                readout[k].bias.data = dataloaders[k].dataset.mean_response
        else:
            for k in data_info.keys():
                readout[k].bias.data = torch.from_numpy(data_info[k]["mean_response"])

    model = LocalEncoder(
        core,
        readout,
    )

    return model
