"""
Adapted from neuralpredictors:
https://github.com/sinzlab/neuralpredictors/blob/v0.3.0.pre/neuralpredictors/layers/readouts/base.py
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from jaxtyping import Float

MIN_MEAN_RATE = 1e-6


def sanitize_mean_rate(mean_rate: torch.Tensor, minimum_rate: float = MIN_MEAN_RATE) -> torch.Tensor:
    """Return finite, positive rates suitable for an inverse output link."""
    if not mean_rate.is_floating_point():
        mean_rate = mean_rate.float()
    return torch.nan_to_num(
        mean_rate,
        nan=minimum_rate,
        posinf=minimum_rate,
        neginf=minimum_rate,
    ).clamp_min(minimum_rate)


def softplus_inverse(mean_rate: torch.Tensor, minimum_rate: float = MIN_MEAN_RATE) -> torch.Tensor:
    """Numerically stable inverse softplus for non-negative rate targets."""
    safe_rate = sanitize_mean_rate(mean_rate, minimum_rate=minimum_rate)
    return safe_rate + torch.log(-torch.expm1(-safe_rate))


def initialize_bias_from_mean_rate(
    bias: torch.Tensor,
    mean_rate: torch.Tensor,
    inverse_link: Callable[[torch.Tensor], torch.Tensor] = softplus_inverse,
) -> None:
    """Initialize a preactivation bias without aliasing the statistics tensor."""
    safe_rate = sanitize_mean_rate(mean_rate.to(device=bias.device, dtype=bias.dtype))
    preactivation_bias = inverse_link(safe_rate)
    if not torch.isfinite(preactivation_bias).all():
        raise ValueError("Inverse output link produced a non-finite readout bias")
    with torch.no_grad():
        bias.copy_(preactivation_bias)


class Readout(nn.Module, ABC):
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

    def initialize_bias(
        self,
        mean_activity: Optional[Float[torch.Tensor, " n_neurons"]] = None,
        inverse_link: Callable[[torch.Tensor], torch.Tensor] = softplus_inverse,
    ) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.
            inverse_link: Maps output rates to readout preactivations.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            with torch.no_grad():
                self.bias.fill_(0)
        else:
            initialize_bias_from_mean_rate(self.bias, mean_activity, inverse_link=inverse_link)

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)

    def plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes] | None = None,
        remove_readout_ticks: bool = False,
        add_titles: bool = True,
    ) -> plt.Figure:
        """Visualize the weights contributing to a single neuron."""
        if neuron_id < 0 or neuron_id >= self.number_of_neurons():
            raise IndexError(f"neuron_id={neuron_id} is out of bounds for {self.number_of_neurons()} neurons")
        if axes is None:
            fig, (ax_readout, ax_features) = plt.subplots(ncols=2, figsize=(12, 6))
        else:
            ax_readout, ax_features = axes
        self._plot_weight_for_neuron(neuron_id, axes=(ax_readout, ax_features), add_titles=add_titles)
        if remove_readout_ticks:
            ax_features.axes.get_xaxis().set_ticks([])
            ax_features.axes.get_yaxis().set_ticks([])
        return ax_readout.figure

    @abstractmethod
    def _plot_weight_for_neuron(self, neuron_id: int, axes: tuple[plt.Axes, plt.Axes], add_titles: bool) -> None:
        """Visualize the weights contributing to a single neuron."""

    @abstractmethod
    def number_of_neurons(self) -> int:
        """Return the number of neurons represented by this readout."""

    def save_weight_visualizations(
        self,
        folder_path: str,
        file_format: str = "jpg",
        state_suffix: str = "",
        cell_indices: list[int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        os.makedirs(folder_path, exist_ok=True)
        suffix = f"_{state_suffix}" if state_suffix else ""

        if cell_indices is None:
            indices_to_plot = list(range(self.number_of_neurons()))
        else:
            indices_to_plot = cell_indices
        for neuron_id in indices_to_plot:
            fig = self.plot_weight_for_neuron(neuron_id, *args, **kwargs)
            fig.tight_layout()
            plot_path = os.path.join(folder_path, f"neuron_{neuron_id}{suffix}.{file_format}")
            fig.savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig.clf()
            plt.close(fig)


class ClonedReadout(Readout):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout: Readout, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]

        self._source = original_readout
        self.alpha = nn.Parameter(torch.ones(self._source.number_of_neurons()))
        self.beta = nn.Parameter(torch.zeros(self._source.number_of_neurons()))

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

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        fig = self._source.plot_weight_for_neuron(
            neuron_id,
            axes=axes,
            add_titles=add_titles,
        )
        fig.suptitle(
            f"Cloned readout: alpha={self.alpha[neuron_id].item():.3g}, beta={self.beta[neuron_id].item():.3g}",
            fontsize=10,
        )

    def number_of_neurons(self) -> int:
        return self._source.number_of_neurons()
