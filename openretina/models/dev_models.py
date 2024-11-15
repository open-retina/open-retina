# ruff: noqa: E501  # for long link in this file, it didn't work to put it at that specific line for some reason
"""
This file contatins models under development. They are subject to change and are not guaranteed to work as expected.
"""

from operator import itemgetter
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from neuralpredictors import regularizers  # type: ignore
from neuralpredictors.layers.readouts import (  # type: ignore
    FullGaussian2d,
    Gaussian3d,
    MultiReadoutBase,
)
from neuralpredictors.utils import get_module_output  # type: ignore

from openretina.dataloaders import get_dims_for_loader_dict
from openretina.hoefling_2024.models import Encoder
from openretina.utils.misc import set_seed

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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

    model = PhotorecVideoEncoder(
        core,
        readout,
    )

    return model


# Baseline NLP:


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


class DynamicLayerNorm(nn.Module):
    """
    A flexible LayerNorm implementation that:
    1. Supports normalization along any axis/axes
    2. Does not use learned affine parameters
    3. Can handle varying sizes of normalization dimensions
    4. Uses einops for clean tensor manipulation
    """

    def __init__(self, norm_axes: Union[int, List[int], Tuple[int, ...]], eps: float = 1e-5):
        """
        Args:
            norm_axes: Axis or axes along which to normalize. Can be single int or list/tuple of ints.
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.norm_axes = [norm_axes] if isinstance(norm_axes, int) else list(norm_axes)
        self.eps = eps

    def _build_reduction_pattern(self, x: torch.Tensor) -> Tuple[str, str, str]:
        """
        Builds the einops reduction pattern based on input shape and normalization axes.

        Returns:
            Tuple of (source_pattern, reduction_pattern, broadcast_pattern)
        """
        # Build dimension names: b0, b1, b2, etc.
        ndim = len(x.shape)
        dim_names = [f"b{i}" for i in range(ndim)]

        # Mark which dimensions to reduce
        reduce_dims = []
        kept_dims = []
        for i in range(ndim):
            if i in self.norm_axes:
                reduce_dims.append(dim_names[i])
            else:
                kept_dims.append(dim_names[i])

        # Create patterns
        source_pattern = " ".join(dim_names)
        reduction_pattern = " ".join(kept_dims)
        broadcast_pattern = " ".join("1" if i in self.norm_axes else f"b{i}" for i in range(ndim))

        return source_pattern, reduction_pattern, broadcast_pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate number of elements being normalized over
        n = torch.prod(torch.tensor([x.shape[i] for i in self.norm_axes]))

        src_pattern, red_pattern, broadcast_pattern = self._build_reduction_pattern(x)

        # Calculate mean and subtract
        mean = reduce(x, f"{src_pattern} -> {broadcast_pattern}", "mean")
        x_centered = x - mean

        # Calculate variance with Bessel's correction and normalize
        var = reduce(x_centered**2, f"{src_pattern} -> {broadcast_pattern}", "mean")
        var = var * (n / (n - 1))  # Apply Bessel's correction
        return x_centered / torch.sqrt(var + self.eps)
