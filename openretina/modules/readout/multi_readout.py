import os
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.nn as nn

from openretina.modules.readout.base import ClonedReadout, Readout
from openretina.modules.readout.factorised_gaussian import SimpleSpatialXFeature3d
from openretina.modules.readout.gaussian import FullGaussian2d
from openretina.modules.readout.klindt_readout import KlindtReadoutWrapper3D


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
        self.session_init_args = {
            "in_shape": in_shape,
            "gaussian_mean_scale": gaussian_mean_scale,
            "gaussian_var_scale": gaussian_var_scale,
            "positive": positive,
            "scale": scale,
            "bias": bias,
        }

        self.add_sessions(n_neurons_dict)

        self.gamma_readout = gamma_readout
        self.gamma_masks = gamma_masks
        self.gaussian_masks = gaussian_masks
        self.readout_reg_avg = readout_reg_avg

    def add_sessions(self, n_neurons_dict: dict[str, int]) -> None:
        """Adds new sessions to the readout wrapper.
        Can be called to add new sessions to an existing readout wrapper."""

        if any(key in self.keys() for key in n_neurons_dict):
            duplicate_session_names = set(self.keys()).intersection(n_neurons_dict.keys())
            raise ValueError(
                f"Found duplicate sessions in n_neurons_dict:  {duplicate_session_names=}. \
                    Make sure to use different session names for each session."
            )
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert len(self.session_init_args["in_shape"]) == 4  # type: ignore
            self.add_module(
                k,
                SimpleSpatialXFeature3d(  # add a readout for each session
                    outdims=n_neurons,
                    **self.session_init_args,  # type: ignore
                ),
            )

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

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        for key in self.readout_keys():
            readout_folder = os.path.join(folder_path, key)
            os.makedirs(readout_folder, exist_ok=True)
            self._modules[key].save_weight_visualizations(readout_folder, file_format, state_suffix)  # type: ignore

    @property
    def sessions(self) -> list[str]:
        return self.readout_keys()


class MultiKlindtReadoutWrapper(nn.ModuleDict):
    """
    Multiple Sessions version of the SimpleSpatialXFeature3d factorised gaussian readout.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        mask_l1_reg: float,
        weights_l1_reg: float,
        laplace_mask_reg: float,
        readout_bias: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[Iterable[Iterable[float]]] = None,
    ):
        super().__init__()
        # set kernels and mask size based on input shape
        num_kernels = [in_shape[0]]
        mask_size = in_shape[2:]
        self.session_init_args = {
            "num_kernels": num_kernels,
            "mask_l1_reg": mask_l1_reg,
            "weights_l1_reg": weights_l1_reg,
            "laplace_mask_reg": laplace_mask_reg,
            "mask_size": mask_size,
            "readout_bias": readout_bias,
            "weights_constraint": weights_constraint,
            "mask_constraint": mask_constraint,
            "init_mask": init_mask,
            "init_weights": init_weights,
            "init_scales": init_scales,
        }

        self.add_sessions(n_neurons_dict)

        self.gamma_readout = weights_l1_reg
        self.gamma_masks = mask_l1_reg
        self.gamma_laplace_masks = laplace_mask_reg

    def add_sessions(self, n_neurons_dict: dict[str, int]) -> None:
        """Adds new sessions to the readout wrapper.
        Can be called to add new sessions to an existing readout wrapper."""

        if any(key in self.keys() for key in n_neurons_dict):
            duplicate_session_names = set(self.keys()).intersection(n_neurons_dict.keys())
            raise ValueError(
                f"Found duplicate sessions in n_neurons_dict:  {duplicate_session_names=}. \
                    Make sure to use different session names for each session."
            )
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert len(self.session_init_args["mask_size"]) == 2  # type: ignore
            self.add_module(
                k,
                KlindtReadoutWrapper3D(  # add a readout for each session
                    num_neurons=n_neurons,
                    **self.session_init_args,  # type: ignore
                ),
            )

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
        reg_loss = self[data_key].regularizer()
        return reg_loss

    def readout_keys(self) -> list[str]:
        return sorted(self._modules.keys())

    def save_weight_visualizations(self, folder_path: str, file_format, state_suffix: str = "") -> None:
        for key in self.readout_keys():
            readout_folder = os.path.join(folder_path, key)
            os.makedirs(readout_folder, exist_ok=True)
            self._modules[key].save_weight_visualizations(readout_folder, file_format, state_suffix)  # type: ignore

    @property
    def sessions(self) -> list[str]:
        return self.readout_keys()


class MultiSampledGaussianReadoutWrapper(nn.ModuleDict):
    """
    Multiple Sessions version of the sampling gaussian readout.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        bias: bool,
        init_mu_range: float,
        init_sigma_range: float,
        batch_sample: bool = True,
        align_corners: bool = True,
        gauss_type: Literal["full", "iso"] = "full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        init_grid=None,
        mean_activity=None,
        gamma_readout: float = 1.0,
        readout_reg_avg: bool = False,
        nonlinearity_function: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.softplus,
    ):
        super().__init__()
        self.session_init_args = {
            "in_shape": in_shape,
            "init_mu_range": init_mu_range,
            "init_simga": init_sigma_range,
            "batch_sample": batch_sample,
            "align_corners": align_corners,
            "gauss_type": gauss_type,
            "grid_mean_predictor": grid_mean_predictor,
            "shared_features": shared_features,
            "shared_grid": shared_grid,
            "init_grid": init_grid,
            "mean_activity": mean_activity,
            "bias": bias,
        }

        self.add_sessions(n_neurons_dict)

        self.gamma_readout = gamma_readout
        self.readout_reg_avg = readout_reg_avg
        self.nonlinearity = nonlinearity_function

    def add_sessions(self, n_neurons_dict: dict[str, int]) -> None:
        """Adds new sessions to the readout wrapper.
        Can be called to add new sessions to an existing readout wrapper."""

        if any(key in self.keys() for key in n_neurons_dict):
            duplicate_session_names = set(self.keys()).intersection(n_neurons_dict.keys())
            raise ValueError(
                f"Found duplicate sessions in n_neurons_dict:  {duplicate_session_names=}. \
                    Make sure to use different session names for each session."
            )
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert len(self.session_init_args["in_shape"]) == 4  # type: ignore
            self.add_module(
                k,
                FullGaussian2d(  # add a readout for each session
                    outdims=n_neurons,
                    **self.session_init_args,  # type: ignore
                ),
            )

    def forward(self, *args, data_key: str | None, **kwargs) -> torch.Tensor:
        if data_key is None:
            readout_responses = []
            for readout_key in self.readout_keys():
                out_core = torch.transpose(args[0], 1, 2)
                out_core = out_core.reshape(((-1,) + out_core.size()[2:]))
                resp = self[readout_key](out_core, **kwargs)
                resp = resp.reshape((args[0].size(0), -1, resp.size(-1)))
                resp = self.nonlinearity(resp)
                readout_responses.append(resp)

            response = torch.concatenate(readout_responses, dim=-1)
        else:
            out_core = torch.transpose(args[0], 1, 2)
            out_core = out_core.reshape(((-1,) + out_core.size()[2:]))
            response = self[data_key](out_core, **kwargs)
            response = response.reshape((args[0].size(0), -1, response.size(-1)))
            response = self.nonlinearity(response)

        return response

    def regularizer(self, data_key: str) -> torch.Tensor:
        feature_loss = self[data_key].feature_l1() * self.gamma_readout
        return feature_loss

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
