import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch import nn


class SimpleSpatialXFeature3d(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        outdims: int,
        gaussian_mean_scale: float = 1e0,
        gaussian_var_scale: float = 1e0,
        positive: bool = False,
        scale: bool = False,
        bias: bool = True,
        nonlinearity_function=torch.nn.functional.softplus,
    ):
        """
        Args:
            in_shape: The shape of the input tensor (c, t, w, h).
            outdims: The number of output dimensions (usually the number of neurons in the session).
            gaussian_mean_scale: The scale factor for the Gaussian mean. Defaults to 1e0.
            gaussian_var_scale: The scale factor for the Gaussian variance. Defaults to 1e0.
            positive: Whether the output should be positive. Defaults to False.
            scale: Whether to include a scale parameter. Defaults to False.
            bias: Whether to include a bias parameter. Defaults to True.
            nonlinearity_function: torch nonlinearity function , e.g. nn.functional.softplus
        """
        super().__init__()
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.gaussian_mean_scale = gaussian_mean_scale
        self.gaussian_var_scale = gaussian_var_scale
        self.positive = positive
        self.nonlinearity_function = nonlinearity_function

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
            self.bias_param = nn.Parameter(torch.zeros(outdims))
        else:
            self.register_buffer("bias_param", torch.zeros(outdims))  # Non-trainable

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

        y = self.nonlinearity_function(y * self.scale_param + self.bias_param)
        return y

    def __repr__(self) -> str:
        c, _, w, h = self.in_shape
        res_array: list[str] = []
        r = f"{self.__class__.__name__} ( {c} x {w} x {h} -> {str(self.outdims)})"
        if not self.bias_param.requires_grad:
            r += " with bias"
        res_array.append(r)

        children_string = "".join([f" -> {ch.__repr__}" for ch in self.children()])
        res_array.append(children_string)
        return "\n".join(res_array)

    def save_weight_visualizations(self, folder_path: str) -> None:
        masks = self.get_mask().detach().cpu().numpy()
        mask_abs_max = np.abs(masks).max()
        features = self.features.detach().cpu().numpy()
        features_min = float(features.min())
        features_max = float(features.max())
        for neuron_id in range(masks.shape[0]):
            mask_neuron = masks[neuron_id, :, :]
            fig_axes_tuple = plt.subplots(ncols=2, figsize=(2 * 6, 6))
            axes: list[plt.Axes] = fig_axes_tuple[1]  # type: ignore

            axes[0].set_title("Readout Mask")
            axes[0].imshow(
                mask_neuron, interpolation="none", cmap="RdBu_r", norm=Normalize(-mask_abs_max, mask_abs_max)
            )

            features_neuron = features[0, :, 0, neuron_id]
            axes[1].set_title("Readout feature weights")
            axes[1].bar(range(features_neuron.shape[0]), features_neuron)
            axes[1].set_ylim((features_min, features_max))

            plot_path = f"{folder_path}/neuron_{neuron_id}.pdf"
            fig_axes_tuple[0].savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig_axes_tuple[0].clf()
            plt.close()
