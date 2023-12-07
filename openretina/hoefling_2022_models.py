from collections import OrderedDict
from collections.abc import Iterable
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralpredictors.utils import get_module_output

from .dataloaders import get_dims_for_loader_dict
from .misc import set_seed


# Batch adapation model
def SFB3d_core_SxF3d_readout(
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
    data_info: dict = None,
):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
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

    class LocalEncoder(Encoder):
        def forward(self, x, data_key=None, detach_core=False, **kwargs):
            self.detach_core = detach_core
            if self.detach_core:
                for name, param in self.core.features.named_parameters():
                    if name.find("speed") < 0:
                        param.requires_grad = False
            x = self.core(x, data_key=data_key)
            x = self.readout(x, data_key=data_key)
            return x

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
    if readout_bias == True:
        if data_info is None:
            for k in dataloaders:
                readout[k].bias.data = dataloaders[k].dataset[:]._asdict()[out_name].mean(0)
        else:
            for k in data_info.keys():
                readout[k].bias.data = torch.from_numpy(data_info[k]["mean_response"])

    model = LocalEncoder(
        core,
        readout,
    )

    return model


class ParametricFactorizedBatchConv3dCore(Core3d, nn.Module):
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
        laplace_padding=0,
        stack=None,
        batch_adaptation=True,
        use_avg_reg=False,
    ):
        super().__init__()
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=laplace_padding)
        self._input_weights_regularizer_temporal = TimeLaplaceL23dnorm(padding=laplace_padding)

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

        # make log_speeds a list/dict/whatever for every scan
        # self.log_speeds = torch.nn.ParameterDict({})
        log_speed_dict = dict()
        for k in n_neurons_dict:
            # self.log_speeds[k] = torch.nn.Parameter(data=torch.zeros(1),
            #                                  requires_grad=batch_adaptation)
            var_name = "_".join(["log_speed", k])
            log_speed_val = torch.nn.Parameter(data=torch.zeros(1), requires_grad=batch_adaptation)
            setattr(self, var_name, log_speed_val)
            log_speed_dict[var_name] = log_speed_val

        if input_padding:
            input_pad = (0, spatial_kernel_size[0] // 2, spatial_kernel_size[0] // 2)
        else:
            input_pad = 0
        if hidden_padding & (len(spatial_kernel_size) > 1):
            hidden_pad = [
                (0, spatial_kernel_size[l] // 2, spatial_kernel_size[l] // 2)
                for l in range(1, len(spatial_kernel_size))
            ]
        else:
            hidden_pad = [0 for l in range(1, len(spatial_kernel_size))]

        if not isinstance(hidden_channels, Iterable):
            hidden_channels = [hidden_channels] * (self.layers)
        if not isinstance(temporal_kernel_size, Iterable):
            temporal_kernel_size = [temporal_kernel_size] * (self.layers)
        if not isinstance(spatial_kernel_size, Iterable):
            spatial_kernel_size = [spatial_kernel_size] * (self.layers)

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = STSeparableBatchConv3d(
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
            #             layer["norm"] = KeywordIgnoringBatchNorm(hidden_channels[0], momentum=momentum, bias=bias, batch_norm_scale=batch_norm_scale)
            layer["norm"] = nn.BatchNorm3d(
                hidden_channels[0], momentum=momentum, affine=bias and batch_norm_scale
            )  # ok or should we ensure same batch norm?
            if bias:
                if not batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels[0])
            elif batch_norm_scale:
                layer["scale"] = Scale3DLayer(hidden_channels[0])
        if final_nonlinearity:
            #             layer["nonlin"] = KeywordIgnoringELU(inplace=True) #test ReLU instead of ELU
            layer["nonlin"] = nn.ELU(inplace=True)
        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers

        for l in range(1, self.layers):
            layer = OrderedDict()
            layer["conv"] = STSeparableBatchConv3d(
                hidden_channels[l - 1],
                hidden_channels[l],
                log_speed_dict,
                temporal_kernel_size[l],
                spatial_kernel_size[l],
                bias=False,
                padding=hidden_pad[l - 1],
                num_scans=self.num_scans,
            )
            if batch_norm:
                #                 layer["norm"] = KeywordIgnoringBatchNorm(hidden_channels[l], momentum=momentum, bias=bias, batch_norm_scale=batch_norm_scale)
                layer["norm"] = nn.BatchNorm3d(hidden_channels[l], momentum=momentum, affine=bias and batch_norm_scale)
                if bias:
                    if not batch_norm_scale:
                        layer["bias"] = Bias3DLayer(hidden_channels[l])
                elif batch_norm_scale:
                    layer["scale"] = Scale2DLayer(hidden_channels[l])
            if final_nonlinearity or l < self.layers - 1:
                #                 layer["nonlin"] = KeywordIgnoringELU(inplace=True)
                layer["nonlin"] = nn.ELU(inplace=True)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_, data_key=None):
        ret = []
        # input_ = input_.repeat(1, self.num_scans, 1, 1, 1) #BS*CDHW
        for l, feat in enumerate(self.features):
            #             do_skip = l >= 1 and self.skip > 1
            do_skip = False
            input_ = feat((input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1), data_key))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def spatial_laplace(self):
        return self._input_weights_regularizer_spatial(self.features[0].conv.weight_spatial, avg=self.use_avg_reg)

    def group_sparsity(self):  # check if this is really what we want
        ret = 0
        for l in range(1, self.layers):
            #             ret = ret + (self.features[l].conv.weight_temporal.pow(2).sum([2,3,4]).sqrt().sum(1) /
            #                          torch.sqrt(1e-8 + self.features[l].conv.weight_temporal.pow(2).sum([1,2,3,4]))).sum()
            ret = (
                ret
                + (
                    self.features[l].conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1)
                    / torch.sqrt(1e-8 + self.features[l].conv.weight_spatial.pow(2).sum([1, 2, 3, 4]))
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
        weight = torch.einsum("oidab,oichw->oidhw", weight_temporal, self.features[0].conv.weight_spatial)
        ret = (weight.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(1e-8 + weight.pow(2).sum([1, 2, 3, 4]))).sum()
        return ret

    def temporal_smoothness(self):
        ret = 0
        for l in range(self.layers):
            ret = ret + temporal_smoothing(self.features[l].conv.sin_weights, self.features[l].conv.cos_weights)
        return ret

    def regularizer(self):
        return (
            self.group_sparsity() * self.gamma_hidden
            + self.gamma_input * self.spatial_laplace()
            + self.gamma_temporal * self.temporal_smoothness()
            + self.group_sparsity0() * self.gamma_in_sparse
        )

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels[-1]


class SpatialXFeature3dReadout(nn.ModuleDict):
    def __init__(
        self,
        core,
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
        for k in n_neurons_dict:  # iterate over sessions (?)
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(
                k,
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
        in_shape,
        outdims,
        gaussian_masks=False,
        gaussian_mean_scale=1e0,
        gaussian_var_scale=1e0,
        initialize_from_roi_masks=False,
        roi_mask=[],
        positive=False,
        scale=False,
        bias=True,
        nonlinearity=True,
        #         stop_grad=False
    ):
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
        # roi_mask = roi_mask.to(device="cuda")

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
            scale = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("scale", scale)
        else:
            self.register_parameter("scale", None)

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    #         self.stop_grad = stop_grad

    def initialize(self, init_noise=1e-3, grid=True):
        if (not self.gaussian_masks) and (not self.initialize_from_roi_masks):
            self.masks.data.normal_(0.0, 0.01)
        self.features.data.normal_(0.0, 0.01)
        if self.scale is not None:
            self.scale.data.normal_(1.0, 0.01)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=False, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return self.features[..., subs_idx].abs().mean()
        else:
            return self.features[..., subs_idx].abs().sum()

    def mask_l1(self, average=False, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if self.gaussian_masks:
            if average:
                return (
                    torch.exp(self.mask_log_var * self.gaussian_var_scale)[..., subs_idx].mean()
                    + (self.mask_mean[..., subs_idx] * self.gaussian_mean_scale).pow(2).mean()
                )
            #                 return self.mask_var_[..., subs_idx].mean() + self.mask_mean[..., subs_idx].pow(2).mean()
            else:
                return (
                    torch.exp(self.mask_log_var * self.gaussian_var_scale)[..., subs_idx].sum()
                    + (self.mask_mean[..., subs_idx] * self.gaussian_mean_scale).pow(2).sum()
                )
        #                 return self.mask_var_[..., subs_idx].sum() + self.mask_mean[..., subs_idx].pow(2).sum()
        else:
            if average:
                return self.masks[..., subs_idx].abs().mean()
            else:
                return self.masks[..., subs_idx].abs().sum()

    def make_mask_grid(self, w, h):
        """Actually mixed up: w (width) is height, and vice versa"""
        grid_w = torch.linspace(-1 * w / max(w, h), 1 * w / max(w, h), w)
        grid_h = torch.linspace(-1 * h / max(w, h), 1 * h / max(w, h), h)
        xx, yy = torch.meshgrid([grid_w, grid_h], indexing="ij")
        grid = torch.stack([xx, yy], 2)[None, ...]
        return grid.repeat([self.outdims, 1, 1, 1])

    def normal_pdf(self):
        """Gets the actual mask values in terms of a PDF from the mean and SD (?)"""
        self.mask_var_ = torch.exp(self.mask_log_var * self.gaussian_var_scale).view(-1, 1, 1)
        pdf = self.grid - self.mask_mean.view(self.outdims, 1, 1, -1) * self.gaussian_mean_scale
        pdf = torch.sum(pdf**2, dim=-1) / self.mask_var_
        pdf = torch.exp(-0.5 * pdf)
        pdf = pdf / torch.sum(pdf, dim=(1, 2), keepdim=True)
        return pdf

    def get_normal_pdf_from_roi_mask(self):
        self.mask_var_ = self.mask_var.view(-1, 1, 1)
        pdf = self.grid - self.mask_mean.view(self.outdims, 1, 1, -1)
        # print("mask var_ shape: {}".format(self.mask_var_.shape))
        # print("grid shape: {}".format(self.grid.shape))
        # print("mask mean shape: {}".format(self.mask_mean.view(
        #     self.outdims, 1, 1, -1).shape))
        pdf = torch.sum(pdf**2, dim=-1) / self.mask_var_
        pdf = torch.exp(-0.5 * pdf)
        pdf = pdf / torch.sum(pdf, dim=(1, 2), keepdim=True)
        return pdf

    def forward(self, x, shift=None, subs_idx=None):
        #         if self.stop_grad:
        #             x = x.detach()
        if self.gaussian_masks:
            self.masks = self.normal_pdf().permute(1, 2, 0)
        else:
            self.masks.data.abs_()

        if self.positive:
            self.features.data.clamp_(0)

        N, c, t, w, h = x.size()
        if subs_idx is not None:
            feat = self.features[..., subs_idx]  # .contiguous()
            masks = self.masks[..., subs_idx]  # .contiguous()
            outdims = feat.size(-1)
            # feat = feat.view(1, c, outdims)
            # masks = masks.view(w, h, outdims)
        else:
            feat = self.features  # .view(1, c, self.outdims)
            masks = self.masks
            outdims = self.outdims

        y = torch.tensordot(x, masks, dims=([3, 4], [0, 1]))
        y = (y * feat).sum(1)

        if self.scale is not None:
            y = y * self.scale
        if self.bias is not None:
            if subs_idx is None:
                y = y + self.bias
            else:
                y = y + self.bias[subs_idx]
        if self.nonlinearity:
            y = torch.log(torch.exp(y) + 1)
        return y

    def __repr__(self):
        c, _, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        #         if self.stop_grad:
        #             r += ", stop_grad=True"
        r += "\n"

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


class Encoder(nn.Module):
    """
    puts together all parts of model (core, readouts) and defines a forward
    function which will return the output of the model; PyTorch then allows
    to call .backward() on the Encoder which will compute the gradients
    """

    def __init__(
        self,
        core,
        readout,
    ):
        super().__init__()
        self.core = core
        self.readout = readout
        self.detach_core = False

    def forward(self, x, data_key=None, detach_core=False, **kwargs):
        self.detach_core = detach_core
        x = self.core(x, data_key=data_key)
        if self.detach_core:
            x = x.detach()
        x = self.readout(x, data_key=data_key)
        return x
