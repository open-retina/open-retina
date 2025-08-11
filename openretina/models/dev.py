"""
Models in this folder are under development. They can depend on the legacy codebase or on neuralpredictors.
"""

import torch
import torch.nn as nn
from neuralpredictors.layers.readouts import Gaussian3d, MultiReadoutBase

from openretina.modules.readout.gaussian import FullGaussian2d


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
