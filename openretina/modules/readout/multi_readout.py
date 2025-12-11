import os
import warnings
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from openretina.modules.readout.base import ClonedReadout, Readout
from openretina.modules.readout.factorized import FactorizedReadout
from openretina.modules.readout.factorized_gaussian import GaussianMaskReadout
from openretina.modules.readout.gaussian import PointGaussianReadout
from openretina.modules.readout.linear_nonlinear_poison import LNPReadout


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

    _base_readout_cls: type[Readout] | None = None

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        base_readout: type[Readout] | None = None,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
        clone_readout=False,
        readout_reg_avg: bool = False,
        **kwargs,
    ):
        # The `base_readout` can be overridden only if the static property `_base_readout_cls` is not set
        if self._base_readout_cls is None:
            assert base_readout is not None, (
                "Argument `base_readout` must be provided if the class variable `_base_readout_cls` is not set"
            )
            self._base_readout_cls = base_readout

        self._readout_kwargs = kwargs
        self._in_shape = in_shape
        self.readout_reg_avg = readout_reg_avg
        self.readout_reg_reduction: Literal["mean", "sum"] = "mean" if readout_reg_avg else "sum"
        super().__init__()

        for i, data_key in enumerate(n_neurons_dict):
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None

            if i == 0 or clone_readout is False:
                self.add_module(
                    data_key,
                    self._base_readout_cls(
                        in_shape=in_shape,
                        outdims=n_neurons_dict[data_key],
                        mean_activity=mean_activity,
                        **kwargs,
                    ),
                )
                original_readout = data_key
            elif i > 0 and clone_readout is True:
                original_readout_object: Readout = self[original_readout]  # type: ignore
                self.add_module(data_key, ClonedReadout(original_readout_object))

        self.initialize(mean_activity_dict)

    def add_sessions(
        self,
        n_neurons_dict: dict[str, int],
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ) -> None:
        """Wrapper method to add new sessions to the readout wrapper.
        Can be called to add new sessions to an existing readout wrapper.
        Individual readouts should override this method to add additional checks.
        """
        self._add_sessions(n_neurons_dict, mean_activity_dict)

    def _add_sessions(
        self,
        n_neurons_dict: dict[str, int],
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ) -> None:
        """Base method to add new sessions to the readout wrapper.
        Can be called to add new sessions to an existing readout wrapper."""

        if any(key in self.keys() for key in n_neurons_dict):
            duplicate_session_names = set(self.keys()).intersection(n_neurons_dict.keys())
            raise ValueError(
                f"Found duplicate sessions in n_neurons_dict:  {duplicate_session_names=}. \
                    Make sure to use different session names for each session."
            )
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert self._base_readout_cls is not None
            self.add_module(
                k,
                self._base_readout_cls(
                    in_shape=self._in_shape,
                    outdims=n_neurons,
                    mean_activity=mean_activity_dict[k] if mean_activity_dict is not None else None,
                    **self._readout_kwargs,
                ),
            )

    def forward(self, *args, data_key: str | None = None, **kwargs) -> torch.Tensor:
        if data_key is None:
            warnings.warn(
                "No data key provided, returning concatenated responses from all readouts",
                stacklevel=2,
                category=UserWarning,
            )
            readout_responses = []
            for readout_key in self.readout_keys():
                resp = self[readout_key](*args, **kwargs)
                readout_responses.append(resp)
            response = torch.cat(readout_responses, dim=-1)
        else:
            response = self[data_key](*args, **kwargs)
        return response

    def initialize(self, mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None):
        for data_key, readout in self.items():
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict is not None else None
            assert isinstance(readout, Readout)
            readout.initialize(mean_activity)

    def regularizer(self, data_key: str | None = None, reduction: Literal["sum", "mean"] | None = None):
        if reduction is None:
            reduction = self.readout_reg_reduction
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        elif data_key is None:
            raise ValueError("data_key is required when there are multiple sessions")
        return self[data_key].regularizer(reduction=reduction)

    def readout_keys(self) -> list[str]:
        return sorted(self._modules.keys())

    def __getitem__(self, key: str) -> Readout:
        """For type checking purposes"""
        res = self._modules[key]
        assert isinstance(res, Readout)
        return res

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        """To ensure we only add Readout objects to the module dictionary"""
        assert isinstance(module, Readout)
        self.add_module(key, module)

    @property
    def sessions(self) -> list[str]:
        return self.readout_keys()

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        for key in self.readout_keys():
            readout_folder = os.path.join(folder_path, key)
            os.makedirs(readout_folder, exist_ok=True)
            self._modules[key].save_weight_visualizations(readout_folder, file_format, state_suffix)  # type: ignore


class MultiGaussianMaskReadout(MultiReadoutBase):
    """
    Multiple Sessions version of the GaussianMaskReadout factorised gaussian readout.
    """

    _base_readout_cls = GaussianMaskReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        scale: bool,
        bias: bool,
        gaussian_mean_scale: float,
        gaussian_var_scale: float,
        positive: bool,
        mask_l1_reg: float = 1.0,
        feature_weights_l1_reg: float = 1.0,
        readout_reg_avg: bool = False,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            mean_activity_dict=mean_activity_dict,
            scale=scale,
            bias=bias,
            gaussian_mean_scale=gaussian_mean_scale,
            gaussian_var_scale=gaussian_var_scale,
            positive=positive,
            mask_l1_reg=mask_l1_reg,
            feature_weights_l1_reg=feature_weights_l1_reg,
            readout_reg_avg=readout_reg_avg,
        )


class MultiFactorizedReadout(MultiReadoutBase):
    """
    Multiple Sessions version of the classic factorized readout.
    """

    _base_readout_cls = FactorizedReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        mask_l1_reg: float,
        weights_l1_reg: float,
        laplace_mask_reg: float,
        readout_bias: bool = False,
        weights_constraint: Literal["abs", "norm", "absnorm"] | None = None,
        mask_constraint: Literal["abs"] | None = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[Iterable[Iterable[float]]] = None,
        readout_reg_avg: bool = False,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ):
        mask_size = in_shape[2:]
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            mask_size=mask_size,
            mask_l1_reg=mask_l1_reg,
            weights_l1_reg=weights_l1_reg,
            laplace_mask_reg=laplace_mask_reg,
            readout_bias=readout_bias,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_mask=init_mask,
            init_weights=init_weights,
            init_scales=init_scales,
            readout_reg_avg=readout_reg_avg,
            mean_activity_dict=mean_activity_dict,
        )


class MultiSampledGaussianReadout(MultiReadoutBase):
    """
    Multiple Sessions version of the sampled point gaussian readout.
    """

    _base_readout_cls = PointGaussianReadout

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
        gamma: float = 1.0,
        reg_avg: bool = False,
        nonlinearity_function: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.softplus,
        mean_activity_dict: dict[str, Float[torch.Tensor, " neurons"]] | None = None,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            bias=bias,
            init_mu_range=init_mu_range,
            init_sigma_range=init_sigma_range,
            batch_sample=batch_sample,
            align_corners=align_corners,
            gauss_type=gauss_type,
            grid_mean_predictor=grid_mean_predictor,
            shared_features=shared_features,
            shared_grid=shared_grid,
            init_grid=init_grid,
            gamma_readout=gamma,
            readout_reg_avg=reg_avg,
            mean_activity_dict=mean_activity_dict,
        )

        self.nonlinearity = nonlinearity_function

    def forward(self, *args, data_key: str | None = None, **kwargs) -> torch.Tensor:
        if data_key is None:
            readout_responses = []
            for readout_key in self.readout_keys():
                out_core = rearrange(args[0], "batch channels time height width -> (batch time) channels height width")
                resp = self[readout_key](out_core, **kwargs)
                resp = rearrange(resp, "(batch time) neurons -> batch time neurons", batch=args[0].size(0))
                resp = self.nonlinearity(resp)
                readout_responses.append(resp)

            response = torch.concatenate(readout_responses, dim=-1)
        else:
            out_core = rearrange(args[0], "batch channels time height width -> (batch time) channels height width")
            response = self[data_key](out_core, **kwargs)
            response = rearrange(response, "(batch time) neurons -> batch time neurons", batch=args[0].size(0))
            response = self.nonlinearity(response)

        return response


class MultipleLNPReadout(MultiReadoutBase):
    """ Multiple Linear Nonlinear Poisson Readout (LNP)
        For use as an LNP Model use this readout with a DummyCore that passes the input through.
    """
    _base_readout_cls = LNPReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        **kwargs,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            **kwargs,
        )
