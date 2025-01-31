import os
from typing import Literal

import torch
import torch.nn as nn

from openretina.modules.readout.base import ClonedReadout, Readout
from openretina.modules.readout.factorised_gaussian import SimpleSpatialXFeature3d


class MultiGaussianReadoutWrapper(nn.ModuleDict):
    """
    Multiple Sessions version of the SimpleSpatialXFeature3d factorised gaussian readout.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        scale: bool,
        bias: bool,
        gaussian_masks: bool,
        gaussian_mean_scale: float,
        gaussian_var_scale: float,
        positive: bool,
        gamma_readout: float,
        gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
    ):
        super().__init__()
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert len(in_shape) == 4
            self.add_module(
                k,
                SimpleSpatialXFeature3d(  # add a readout for each session
                    in_shape,
                    n_neurons,
                    gaussian_mean_scale=gaussian_mean_scale,
                    gaussian_var_scale=gaussian_var_scale,
                    positive=positive,
                    scale=scale,
                    bias=bias,
                ),
            )

        self.gamma_readout = gamma_readout
        self.gamma_masks = gamma_masks
        self.gaussian_masks = gaussian_masks
        self.readout_reg_avg = readout_reg_avg

    def forward(self, *args, data_key: str | None, **kwargs) -> torch.Tensor:
        if data_key is None:
            readout_responses = []
            for readout_key in self.readout_keys():
                resp = self[readout_key](*args, **kwargs)
                readout_responses.append(resp)
            response = torch.concatenate(readout_responses, dim=-1)
        else:
            response = self[data_key](*args, **kwargs)
        return response

    def regularizer(self, data_key: str) -> torch.Tensor:
        feature_loss = self[data_key].feature_l1(average=self.readout_reg_avg) * self.gamma_readout
        mask_loss = self[data_key].mask_l1(average=self.readout_reg_avg) * self.gamma_masks
        return feature_loss + mask_loss

    def readout_keys(self) -> list[str]:
        return sorted(self._modules.keys())

    def save_weight_visualizations(self, folder_path: str) -> None:
        for key in self.readout_keys():
            readout_folder = os.path.join(folder_path, key)
            os.makedirs(readout_folder, exist_ok=True)
            self._modules[key].save_weight_visualizations(readout_folder)  # type: ignore

    @property
    def sessions(self) -> list[str]:
        return self.readout_keys()


class MultiReadoutBase(nn.ModuleDict):
    """
    Base class for MultiReadouts. It is a dictionary of data keys and readouts to the corresponding datasets.

    Adapted from neuralpredictors. Original code at:
    https://github.com/sinzlab/neuralpredictors/blob/v0.3.0.pre/neuralpredictors/layers/readouts/multi_readout.py


    Args:
        in_shape_dict (dict): dictionary of data_key and the corresponding dataset's shape as an output of the core.

        n_neurons_dict (dict): dictionary of data_key and the corresponding dataset's number of neurons

        base_readout (torch.nn.Module): base readout class. If None, self._base_readout must be set manually in the
                                        inheriting class's definition.

        mean_activity_dict (dict): dictionary of data_key and the corresponding dataset's mean responses.
                                    Used to initialize the readout bias with.
                                    If None, the bias is initialized with 0.

        clone_readout (bool): whether to clone the first data_key's readout to all other readouts, only allowing for a
                                scale and offset. This is a rather simple method to enforce parameter-sharing
                                between readouts.

        gamma_readout (float): regularization strength

        **kwargs: additional keyword arguments to be passed to the base_readout's constructor
    """

    _base_readout = None

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        base_readout: Readout | None = None,
        mean_activity_dict: dict[str, float] | None = None,
        clone_readout=False,
        **kwargs,
    ):
        # The `base_readout` can be overridden only if the static property `_base_readout` is not set
        if self._base_readout is None:
            self._base_readout = base_readout

        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set")
        super().__init__()

        for i, data_key in enumerate(n_neurons_dict):
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None

            if i == 0 or clone_readout is False:
                self.add_module(
                    data_key,
                    self._base_readout(
                        in_shape=in_shape,
                        outdims=n_neurons_dict[data_key],
                        mean_activity=mean_activity,
                        **kwargs,
                    ),
                )
                original_readout = data_key
            elif i > 0 and clone_readout is True:
                self.add_module(data_key, ClonedReadout(self[original_readout]))

        self.initialize(mean_activity_dict)

    def forward(self, *args, data_key: str | None = None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def initialize(self, mean_activity_dict: dict[str, float] | None = None):
        for data_key, readout in self.items():
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None
            readout.initialize(mean_activity)

    def regularizer(self, data_key: str | None = None, reduction: Literal["sum", "mean", None] = "sum"):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key].regularizer(reduction=reduction)
