"""
Adapted from neuralpredictors:
https://github.com/sinzlab/neuralpredictors/blob/v0.3.0.pre/neuralpredictors/layers/readouts/base.py
"""

import warnings
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float


class Readout(nn.Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    features: nn.Parameter
    bias: nn.Parameter

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(
        self,
        reduction: Literal["sum", "mean", None] = "sum",
    ) -> torch.Tensor:
        raise NotImplementedError("regularizer is not implemented for ", self.__class__.__name__)

    def apply_reduction(self, x: torch.Tensor, reduction: Literal["sum", "mean", None] = "mean") -> torch.Tensor:
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x: output of the regularizer
            reduction: method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].

        Returns: reduced value of the regularizer
        """

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def initialize_bias(self, mean_activity: Optional[Float[torch.Tensor, " n_neurons"]] = None) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)

    def save_weight_visualizations(
        self, folder_path: str, file_format: str = "jpg", state_suffix: str = "", *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError("save_weight_visualizations is not implemented for ", self.__class__.__name__)


class ClonedReadout(Readout):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout: Readout, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]

        self._source = original_readout
        self.alpha = nn.Parameter(torch.ones(self._source.features.shape[-1]))
        self.beta = nn.Parameter(torch.zeros(self._source.features.shape[-1]))

    def forward(self, x: torch.Tensor, **kwarg: Any) -> torch.Tensor:
        x = self._source(x) * self.alpha + self.beta
        return x

    def feature_l1(self, average: bool = True) -> torch.Tensor:
        """Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self, **kwargs: Any) -> None:
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
