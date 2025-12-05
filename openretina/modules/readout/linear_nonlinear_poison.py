import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int

from openretina.modules.layers import regularizers
from openretina.modules.readout.multi_readout import Readout


class LNPReadout(Readout):
    # Linear nonlinear Poisson
    def __init__(
        self,
        in_shape: Int[tuple, "channel time height width"],
        outdims: int,
        smooth_weight: float = 0.0,
        sparse_weight: float = 0.0,
        smooth_regularizer: str = "LaplaceL2norm",
        laplace_padding=None,
        nonlinearity: str = "exp",
        **kwargs,
    ):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight
        self.kernel_size = tuple(in_shape[2:])
        self.in_channels = in_shape[0]
        self.n_neurons = outdims
        self.nonlinearity = torch.exp if nonlinearity == "exp" else F.__dict__[nonlinearity]

        self.inner_product_kernel = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.n_neurons,  # Each neuron gets its own kernel
            kernel_size=(1, *self.kernel_size),  # Not using time
            bias=False,
            stride=1,
        )

        nn.init.xavier_normal_(self.inner_product_kernel.weight.data)

        regularizer_config = (
            dict(padding=laplace_padding, kernel=self.kernel_size)
            if smooth_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )

        self._smooth_reg_fn = regularizers.__dict__[smooth_regularizer](**regularizer_config)

    def forward(self, x: Float[torch.Tensor, "batch channels t h w"], data_key=None, **kwargs):
        out = self.inner_product_kernel(x)
        out = self.nonlinearity(out)
        out = rearrange(out, "batch neurons t 1 1 -> batch t neurons")
        return out

    def weights_l1(self, average: bool = True):
        """Returns l1 regularization across all weight dimensions

        Args:
            average (bool, optional): use mean of weights instead of sum. Defaults to True.
        """
        if average:
            return self.inner_product_kernel.weight.abs().mean()
        else:
            return self.inner_product_kernel.weight.abs().sum()

    def laplace(self):
        # Squeezing out the empty time dimension, so we can use 2D regularizers
        return self._smooth_reg_fn(self.inner_product_kernel.weight.squeeze(2))

    def regularizer(self, **kwargs):
        return self.smooth_weight * self.laplace() + self.sparse_weight * self.weights_l1()

    def initialize(self, *args, **kwargs):
        pass