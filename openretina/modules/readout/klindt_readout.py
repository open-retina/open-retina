import os
from math import ceil, sqrt
from typing import Any, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from openretina.modules.layers import FlatLaplaceL23dnorm
from openretina.modules.readout.base import Readout


class KlindtReadoutWrapper3D(Readout):
    mask_size: Tuple[int, int]

    def __init__(
        self,
        num_kernels: Sequence[int],
        num_neurons: int,
        mask_l1_reg: float,
        weights_l1_reg: float,
        laplace_mask_reg: float,
        mask_size: int | Tuple[int, int],
        readout_bias: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        """
        Initializes the Klindt readout module : (2d spatial mask + feature weights) / cell.
        Args:
            num_kernels (Sequence[int]): Number of kernels per layer.
            num_neurons (int): Number of output neurons.
            mask_l1_reg (float): L1 regularization strength for mask.
            weights_l1_reg (float): L1 regularization strength for weights.
            laplace_mask_reg (float): Laplace regularization strength for mask.
            mask_size (int | Tuple[int, int]): Size of the mask (height, width).
            readout_bias (bool, optional): If True, includes bias in readout. Defaults to False.
            weights_constraint (Optional[str], optional): Constraint for weights. Defaults to None.
            mask_constraint (Optional[str], optional): Constraint for mask. Defaults to None.
            init_mask (Optional[torch.Tensor], optional): Initial mask tensor. Defaults to None.
            init_weights (Optional[torch.Tensor], optional): Initial weights tensor. Defaults to None.
            init_scales (Optional[Sequence[Tuple[float, float]]], optional): Initialization scales for mask and weights. Defaults to None.
        Raises:
            ValueError: If neither init_mask nor init_scales is provided.
        """
        super().__init__()

        self.num_neurons = num_neurons
        self.reg = [mask_l1_reg, weights_l1_reg, laplace_mask_reg]
        self.readout_bias = readout_bias
        self.weights_constraint = weights_constraint
        self.mask_constraint = mask_constraint
        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=0)
        self.outdims = num_neurons

        if isinstance(mask_size, int):
            num_mask_pixels = mask_size**2
            self.mask_size = (mask_size, mask_size)
        else:
            h, w = mask_size
            num_mask_pixels = h * w
            self.mask_size = mask_size

        if init_mask is not None:
            num_neurons = init_mask.shape[0]
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
            self.readout_weights = nn.Parameter(init_weights)
        else:
            assert init_scales is not None
            mean, std = init_scales[1]
            self.readout_weights = nn.Parameter(torch.normal(mean=mean, std=std, size=(num_kernels[-1], num_neurons)))

        init_bias = 0.5 if self.readout_bias else 0.0
        self.bias = nn.Parameter(torch.full((num_neurons,), init_bias))

    def apply_constraints(self):
        if self.mask_constraint == "abs":
            with torch.no_grad():
                self.mask_weights.data.abs_()
        if self.weights_constraint == "abs":
            with torch.no_grad():
                self.readout_weights.data.abs_()
        elif self.weights_constraint == "norm":
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.readout_weights**2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data /= norm
        elif self.weights_constraint == "absnorm":
            with torch.no_grad():
                self.readout_weights.data.abs_()
                norm = torch.sqrt(torch.sum(self.readout_weights**2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data /= norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()
        B, C, T, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w
        x_flat = x.view(B, C, T, -1).permute(0, 2, 1, 3)
        masked = torch.matmul(x_flat, self.mask_weights)
        masked = masked.permute(0, 1, 3, 2)
        out = (masked * self.readout_weights.T.unsqueeze(0).unsqueeze(0)).sum(dim=3)
        if self.readout_bias:
            out = out + self.bias
        return F.softplus(out)

    def regularizer(self, reduction: Optional[Literal["sum", "mean"]] = None) -> torch.Tensor:
        mask_r = self.reg[0] * torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0))
        wt_r = self.reg[1] * torch.mean(torch.sum(torch.abs(self.readout_weights), dim=0))
        reshaped_masked_weights = self.mask_weights.reshape(-1, 1, 1, self.mask_size[0], self.mask_size[1])
        laplace_mask_r = self.reg[2] * self._input_weights_regularizer_spatial(reshaped_masked_weights, avg=False)
        if reduction == "mean":
            return mask_r + wt_r + laplace_mask_r / 3
        else:
            return mask_r + wt_r + laplace_mask_r

    def save_weight_visualizations(self, readout_folder, file_format, state_suffix=None, cell_indices=None):
        state_suffix = str(state_suffix)
        if hasattr(self, "mask_size"):
            H, W = self.mask_size
        else:
            raise AttributeError("Model does not have attribute 'mask_size'.")

        readout = self
        if hasattr(readout, "mask_weights"):  # nn.Parameter & matmul
            weights = dict(readout.named_parameters())["mask_weights"].detach().cpu()
            weights = weights.T.view(-1, H, W)  # [num_neurons, H, W]
        elif hasattr(readout, "mask"):  # nn.Linear
            weights = dict(readout.named_parameters())["mask.weight"].detach().cpu()
            weights = weights.view(-1, H, W)
        else:
            raise AttributeError("Model does not have 'mask_weights' or 'readout_weights'.")

        total_neurons = weights.shape[0]
        if not cell_indices:
            cell_indices = list(range(total_neurons))
        else:
            cell_indices = [i for i in cell_indices if 0 <= i < total_neurons]

        selected_weights = weights[cell_indices]
        n_cells = selected_weights.shape[0]

        grid_cols = min(8, ceil(sqrt(n_cells)))
        grid_rows = ceil(n_cells / grid_cols)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(1.5 * grid_cols, 1.5 * grid_rows))

        if n_cells == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        vmin = torch.min(selected_weights).item()
        vmax = torch.max(selected_weights).item()

        for i, ax in enumerate(axes):
            if i < n_cells:
                ax.imshow(selected_weights[i], interpolation="bicubic", cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

        fig.suptitle("Spatial Masks", fontsize=14)
        fig.tight_layout()
        fig.savefig(
            os.path.join(readout_folder, "readout_masks_" + state_suffix + f".{file_format}"),
            bbox_inches="tight",
            facecolor="w",
            dpi=300,
        )
        fig.clf()
        plt.close()

        # Feature weights
        weights = readout.readout_weights.detach().cpu()
        weights = weights.numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(weights, aspect="auto", cmap="gray")
        ax.set_title("Feature Weights (channels × neurons)")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Feature Channel")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        fig.savefig(
            os.path.join(readout_folder, "feature_weights_" + state_suffix + f".{file_format}"),
            bbox_inches="tight",
            facecolor="w",
            dpi=300,
        )
        fig.clf()
        plt.close()
