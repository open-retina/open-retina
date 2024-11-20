"""
Models in this folder are under development. They can depend on the legacy codebase or on neuralpredictors.
"""

import torch
import torch.nn as nn
from neuralpredictors import regularizers
from neuralpredictors.layers.readouts import FullGaussian2d, Gaussian3d, MultiReadoutBase


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
    # Linear nonlinear Poisson
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


class Encoder(nn.Module):
    """
    puts together all parts of model (core, readouts) and defines a forward
    function which will return the output of the model; PyTorch then allows
    to call .backward() on the Encoder which will compute the gradients
    """

    def __init__(
        self,
        core: nn.Module,
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

    def readout_keys(self) -> list[str]:
        return self.readout.readout_keys()


class MultipleLNP(Encoder):
    def __init__(self, in_shape_dict, n_neurons_dict, **kwargs):
        # The multiple LNP model acts like a readout reading directly from the videos
        readout = MultiReadoutBase(in_shape_dict, n_neurons_dict, base_readout=LNP, **kwargs)
        super().__init__(
            core=DummyCore(),
            readout=readout,
        )


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
