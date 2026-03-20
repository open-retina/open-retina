import os
from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import Normalize

from openretina.modules.layers import FlatLaplaceL23dnorm
from openretina.modules.readout.base import Readout


class FactorizedReadout(Readout):
    """
    The canonical factorized readout module: each neuron's output is the dot product
    of a learned 2D spatial mask and feature weights.

    This module implements the general Factorized (a.k.a. "Klindt") Readout, where—for
    each neuron—the spatial integration is performed via a freeform, unconstrained (but sparse) mask,
    and the stimulus dimensions (e.g., features, channels, time) are combined by a separate
    learned vector of feature weights. The spatial mask is independently learned for every neuron
    without any restriction to a particular functional form, but with sparsity penalties.

    First introduced in Klindt et al., 2017:
    https://doi.org/10.48550/arXiv.1711.02653

    Key notes:
        - Unlike parametric-masked readouts (see `GaussianMaskReadout`), this class allows the spatial mask
        to take any shape, offering maximum expressive power for fitting the spatial receptive field.

        - Typical regularizations include sparsity (L1), Laplace smoothness penalties, and optional
        constraints (non-negativity, normalization) on the mask or weights.

    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        outdims: int,
        mask_l1_reg: float,
        weights_l1_reg: float,
        laplace_mask_reg: float,
        mask_size: int | tuple[int, int],
        readout_bias: bool = False,
        weights_constraint: Literal["abs", "norm", "absnorm"] | None = None,
        mask_constraint: Literal["abs"] | None = None,
        init_mask: torch.Tensor | None = None,
        init_weights: torch.Tensor | None = None,
        init_scales: Sequence[tuple[float, float]] | None = None,
        mean_activity: Float[torch.Tensor, " outdims"] | None = None,
    ):
        """
        Initializes the FactorizedReadout module : (2d spatial mask + feature weights) / cell.
        Args:
            in_shape: The shape of the input tensor (c, t, w, h).
            outdims (int): Number of output neurons.
            mask_l1_reg (float): L1 regularization strength for mask.
            weights_l1_reg (float): L1 regularization strength for weights.
            laplace_mask_reg (float): Laplace regularization strength for mask.
            mask_size (int | Tuple[int, int]): Size of the mask (height, width) or (height).
            readout_bias (bool, optional): If True, includes bias in readout. Defaults to False.
            weights_constraint (Optional[str], optional): Constraint for weights. Defaults to None.
            mask_constraint (Optional[str], optional): Constraint for mask. Defaults to None.
            init_mask (Optional[torch.Tensor], optional): Initial mask tensor. Defaults to None.
            init_weights (Optional[torch.Tensor], optional): Initial weights tensor. Defaults to None.
            init_scales (Optional[Sequence[Tuple[float, float]]], optional): Initialization scales for mask
            and weights. Defaults to None.
            mean_activity (Float[torch.Tensor, " outdims"] | None): Mean activity of neurons. Defaults to None.
        Raises:
            ValueError: If neither init_mask nor init_scales is provided.
        """
        super().__init__()

        self.outdims = outdims
        self.in_shape = in_shape
        channels, _, _, _ = in_shape
        self.reg = [mask_l1_reg, weights_l1_reg, laplace_mask_reg]
        self.readout_bias = readout_bias
        self.weights_constraint = weights_constraint
        self.mask_constraint = mask_constraint
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=0)
        num_neurons = outdims

        if isinstance(mask_size, int):
            num_mask_pixels = mask_size**2
            self.mask_size = (mask_size, mask_size)
        else:
            h, w = mask_size
            num_mask_pixels = h * w
            self.mask_size = mask_size

        if init_mask is not None:
            assert num_neurons == init_mask.shape[0], "Number of neurons in init_mask does not match outdims"
            h, w = self.mask_size
            H, W = init_mask.shape[2], init_mask.shape[3]
            h_offset = (H - h) // 2
            w_offset = (W - w) // 2

            # Crop center region
            cropped = init_mask[:, :, h_offset : h_offset + h, w_offset : w_offset + w]  # shape: (num_neurons, 1, h, w)

            # Reshape to (num_mask_pixels, num_neurons)
            reshaped = cropped.reshape(num_neurons, -1).T  # shape: (h*w, num_neurons)

            # Convert to tensor and register as parameter
            self.mask_weights = nn.Parameter(torch.tensor(reshaped, dtype=torch.float32))
        else:
            if init_scales is None:
                raise ValueError("Either init_mask or init_scales must be provided")
            mean, std = init_scales[0]
            self.mask_weights = nn.Parameter(torch.normal(mean=mean, std=std, size=(num_mask_pixels, num_neurons)))

        if init_weights is not None:
            self.feature_weights = nn.Parameter(init_weights)
        else:
            assert init_scales is not None
            mean, std = init_scales[1]
            self.feature_weights = nn.Parameter(torch.normal(mean=mean, std=std, size=(channels, num_neurons)))

        if readout_bias:
            self.bias = nn.Parameter(torch.zeros(outdims))
        else:
            self.register_buffer("bias", torch.zeros(outdims))  # Non-trainable

        self.initialize(mean_activity)

    def apply_constraints(self):
        if self.mask_constraint == "abs":
            with torch.no_grad():
                self.mask_weights.data.abs_()
        if self.weights_constraint == "abs":
            with torch.no_grad():
                self.feature_weights.data.abs_()
        elif self.weights_constraint == "norm":
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.feature_weights**2, dim=0, keepdim=True) + 1e-5)
                self.feature_weights.data /= norm
        elif self.weights_constraint == "absnorm":
            with torch.no_grad():
                self.feature_weights.data.abs_()
                norm = torch.sqrt(torch.sum(self.feature_weights**2, dim=0, keepdim=True) + 1e-5)
                self.feature_weights.data /= norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()
        B, C, T, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w
        x_flat = x.view(B, C, T, -1).permute(0, 2, 1, 3)
        masked = torch.matmul(x_flat, self.mask_weights)
        masked = masked.permute(0, 1, 3, 2)
        out = (masked * self.feature_weights.T.unsqueeze(0).unsqueeze(0)).sum(dim=3)
        if self.readout_bias:
            out = out + self.bias
        return F.softplus(out)

    def regularizer(self, reduction: Optional[Literal["sum", "mean"]] = None) -> torch.Tensor:
        mask_r = self.reg[0] * torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0))
        wt_r = self.reg[1] * torch.mean(torch.sum(torch.abs(self.feature_weights), dim=0))
        reshaped_masked_weights = self.mask_weights.reshape(-1, 1, 1, self.mask_size[0], self.mask_size[1])
        laplace_mask_r = self.reg[2] * self._input_weights_regularizer_spatial(reshaped_masked_weights, avg=False)
        if reduction == "mean":
            return mask_r + wt_r + laplace_mask_r / 3
        else:
            return mask_r + wt_r + laplace_mask_r

    def _spatial_masks(self) -> torch.Tensor:
        h, w = self.mask_size
        return self.mask_weights.T.view(-1, h, w)

    def _plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes],
        add_titles: bool = True,
    ) -> None:
        ax_readout, ax_features = axes

        masks = self._spatial_masks().detach().cpu().numpy()
        features = self.feature_weights.detach().cpu().numpy()
        mask_abs_max = float(np.abs(masks).max()) or 1.0
        feature_abs_max = float(np.abs(features).max()) or 1.0

        ax_readout.imshow(
            masks[neuron_id],
            interpolation="none",
            cmap="RdBu_r",
            norm=Normalize(-mask_abs_max, mask_abs_max),
        )

        features_neuron = features[:, neuron_id]
        ax_features.bar(range(features_neuron.shape[0]), features_neuron)
        ax_features.set_ylim((-feature_abs_max, feature_abs_max))

        if add_titles:
            ax_readout.set_title("Readout Mask")
            ax_features.set_title("Readout Feature Weights")

    def number_of_neurons(self) -> int:
        return self.outdims

    def initialize(self, mean_activity: Float[torch.Tensor, " outdims"] | None = None) -> None:
        if mean_activity is not None:
            self.initialize_bias(mean_activity)


class KlindtReadoutWrapper3D(FactorizedReadout):
    pass
