from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import Normalize
from torch import nn

from openretina.modules.readout.base import Readout


class GaussianMaskReadout(Readout):
    """
    A readout module that computes each neuron's output as a weighted sum (dot product) across
    the spatial extent of the core feature map, using a 2D Gaussian mask per neuron.
    It can be considered as an extension of the classic FactorisedReadout, where the spatial
    mask is enforced to have a Gaussian shape.

    First introduced in Hoefling et al., 2024:
    https://doi.org/10.7554/eLife.86860

    Key notes:
        - Unlike point-based Gaussian readouts (see `PointGaussianReadout`), this class
        produces a full spatial mask for each neuron, effectively performing spatial integration (weighted by
        a Gaussian) across the entire input feature map in the spatial dimensions.

        - Each neuron has a single mask_log_var scalar (not per-axis), used as variance for both x and y, so the
        receptive field is circular (in the normalized grid), axis-aligned, and isotropic.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        outdims: int,
        mean_activity: Float[torch.Tensor, " outdims"] | None = None,
        gaussian_mean_scale: float = 1e0,
        gaussian_var_scale: float = 1e0,
        positive: bool = False,
        scale: bool = False,
        bias: bool = True,
        nonlinearity_function=torch.nn.functional.softplus,
        mask_l1_reg: float = 1.0,
        feature_weights_l1_reg: float = 1.0,
    ):
        """
        Args:
            in_shape: The shape of the input tensor (c, t, w, h).
            outdims: The number of output dimensions (usually the number of neurons in the session).
            mean_activity: The mean activity of the neurons, used to initialize the bias. Defaults to None.
            gaussian_mean_scale: The scale factor for the Gaussian mask mean. Defaults to 1e0.
            gaussian_var_scale: The scale factor for the Gaussian mask variance. Defaults to 1e0.
            positive: Whether the output should be positive. Defaults to False.
            scale: Whether to include a scale parameter. Defaults to False.
            bias: Whether to include a bias parameter. Defaults to True.
            nonlinearity_function: torch nonlinearity function , e.g. nn.functional.softplus
            mask_l1_reg: The regularization strength for the sparsity of the spatial mask. Defaults to 1.0.
            feature_weights_l1_reg: The regularization strength for the sparsity of feature weights. Defaults to 1.0.
        """
        super().__init__()
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.gaussian_mean_scale = gaussian_mean_scale
        self.gaussian_var_scale = gaussian_var_scale
        self.positive = positive
        self.nonlinearity_function = nonlinearity_function
        self.mask_l1_reg = mask_l1_reg
        self.feature_weights_l1_reg = feature_weights_l1_reg

        """we train on the log var and transform to var in a separate step"""
        self.mask_mean = torch.nn.Parameter(data=torch.zeros(self.outdims, 2), requires_grad=True)
        self.mask_log_var = torch.nn.Parameter(data=torch.zeros(self.outdims), requires_grad=True)

        # Grid is fixed and untrainable, so we register it as a buffer
        self.register_buffer("grid", self.make_mask_grid(outdims, w, h))

        self.features = nn.Parameter(torch.Tensor(1, c, 1, outdims))
        self.features.data.normal_(1.0 / c, 0.01)

        if scale:
            self.scale_param = nn.Parameter(torch.ones(outdims))
            self.scale_param.data.normal_(1.0, 0.01)
        else:
            self.register_buffer("scale_param", torch.ones(outdims))  # Non-trainable

        if bias:
            self.bias = nn.Parameter(torch.zeros(outdims))
        else:
            self.register_buffer("bias", torch.zeros(outdims))  # Non-trainable

        self.initialize(mean_activity)

    def initialize(self, mean_activity: torch.Tensor | None = None) -> None:
        """
        Initialize bias using base helper to optionally use mean activity.
        """
        self.initialize_bias(mean_activity)

    def feature_l1(self, average: bool = False) -> torch.Tensor:
        features_abs = self.features.abs()
        if average:
            return features_abs.mean()
        else:
            return features_abs.sum()

    def mask_l1(self, average: bool = False) -> torch.Tensor:
        if average:
            return (
                torch.exp(self.mask_log_var * self.gaussian_var_scale).mean()
                + (self.mask_mean * self.gaussian_mean_scale).pow(2).mean()
            )
        else:
            return (
                torch.exp(self.mask_log_var * self.gaussian_var_scale).sum()
                + (self.mask_mean * self.gaussian_mean_scale).pow(2).sum()
            )

    def regularizer(self, reduction: Literal["sum", "mean", None] = "sum") -> torch.Tensor:
        reg = (
            self.mask_l1(average=reduction == "mean") * self.mask_l1_reg
            + self.feature_l1(average=reduction == "mean") * self.feature_weights_l1_reg
        )
        return reg

    @staticmethod
    def make_mask_grid(outdims: int, w: int, h: int) -> torch.Tensor:
        """Actually mixed up: w (width) is height, and vice versa"""
        grid_w = torch.linspace(-1 * w / max(w, h), 1 * w / max(w, h), w)
        grid_h = torch.linspace(-1 * h / max(w, h), 1 * h / max(w, h), h)
        xx, yy = torch.meshgrid([grid_w, grid_h], indexing="ij")
        grid = torch.stack([xx, yy], 2)[None, ...]
        return grid.repeat([outdims, 1, 1, 1])

    def get_mask(self) -> torch.Tensor:
        """Gets the actual mask values in terms of a PDF from the mean and SD"""
        scaled_log_var = self.mask_log_var * self.gaussian_var_scale
        mask_var_ = torch.exp(torch.clamp(scaled_log_var, min=-20, max=20)).view(-1, 1, 1)
        pdf = self.grid - self.mask_mean.view(self.outdims, 1, 1, -1) * self.gaussian_mean_scale
        pdf = torch.sum(pdf**2, dim=-1) / (mask_var_ + 1e-8)
        pdf = torch.exp(-0.5 * torch.clamp(pdf, max=20))
        normalisation = torch.sum(pdf, dim=(1, 2), keepdim=True)
        pdf = torch.nan_to_num(pdf / normalisation)
        return pdf

    @property
    def masks(self) -> torch.Tensor:
        return self.get_mask().permute(1, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masks = self.masks
        y = torch.einsum("nctwh,whd->nctd", x, masks)

        if self.positive:
            self.features.data.clamp_(0)
        y = (y * self.features).sum(1)

        y = self.nonlinearity_function(y * self.scale_param + self.bias)
        return y

    def __repr__(self) -> str:
        c, _, w, h = self.in_shape
        res_array: list[str] = []
        r = f"{self.__class__.__name__} ( {c} x {w} x {h} -> {str(self.outdims)})"
        if not self.bias.requires_grad:
            r += " with bias"
        res_array.append(r)

        children_string = "".join([f" -> {ch.__repr__}" for ch in self.children()])
        res_array.append(children_string)
        return "\n".join(res_array)

    def plot_weight_for_neuron(
        self,
        neuron_id: int,
        axes: tuple[plt.Axes, plt.Axes] | None = None,
        add_titles: bool = True,
        remove_readout_ticks: bool = False,
    ) -> plt.Figure:
        if axes is None:
            fig_axes_tuple = plt.subplots(ncols=2, figsize=(2 * 6, 6))
            ax_readout, ax_features = fig_axes_tuple[1]  # type: ignore
        else:
            ax_readout, ax_features = axes

        masks = self.get_mask().detach().cpu().numpy()
        features = self.features.detach().cpu().numpy()
        mask_abs_max = np.absolute(masks).max()
        mask_neuron = masks[neuron_id, :, :]

        ax_readout.imshow(mask_neuron, interpolation="none", cmap="RdBu_r", norm=Normalize(-mask_abs_max, mask_abs_max))
        ax_readout.axes.get_xaxis().set_ticks([])
        ax_readout.axes.get_yaxis().set_ticks([])

        features_neuron = features[0, :, 0, neuron_id]
        ax_features.bar(range(features_neuron.shape[0]), features_neuron)
        ax_features.set_ylim((features.min(), features.max()))
        if remove_readout_ticks:
            ax_features.axes.get_xaxis().set_ticks([])
            ax_features.axes.get_yaxis().set_ticks([])

        if add_titles:
            ax_readout.set_title("Readout Mask")
            ax_features.set_title("Readout Feature Weights")

        return plt.gcf()

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg", state_suffix: str = "") -> None:
        for neuron_id in range(self.outdims):
            fig_axes_tuple = plt.subplots(ncols=2, figsize=(2 * 6, 6))
            axes: tuple[plt.Axes, plt.Axes] = fig_axes_tuple[1]  # type: ignore
            self.plot_weight_for_neuron(neuron_id, axes)

            plot_path = f"{folder_path}/neuron_{neuron_id}_{state_suffix}.{file_format}"
            fig_axes_tuple[0].savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig_axes_tuple[0].clf()
            plt.close()


# Alias for backward compatibility
class SimpleSpatialXFeature3d(GaussianMaskReadout):
    pass
