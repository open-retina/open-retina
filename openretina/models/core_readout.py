from collections import OrderedDict
from typing import Iterable
import os

import numpy as np
import torch
from torch import nn
import lightning
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from openretina.measures import PoissonLoss3d, CorrelationLoss3d
from openretina.dataloaders import DataPoint
from openretina.hoefling_2024.models import STSeparableBatchConv3d, Bias3DLayer, temporal_smoothing


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
        self.grid = torch.nn.Parameter(data=self.make_mask_grid(outdims, w, h), requires_grad=False)

        self.features = nn.Parameter(torch.Tensor(1, c, 1, outdims))
        self.features.data.normal_(1.0 / c, 0.01)
        self.scale_param = nn.Parameter(torch.ones(outdims), requires_grad=scale)
        if scale:
            self.scale_param.data.normal_(1.0, 0.01)
        self.bias_param = nn.Parameter(torch.zeros(outdims), requires_grad=bias)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masks = self.get_mask().permute(1, 2, 0)
        # Drastic downscaling!
        # x = torch.nn.functional.max_pool3d(x, [1, 12, 25])
        y = torch.einsum("nctwh,whd->nctd", x, masks)
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
            fig_axes_tuple = plt.subplots(ncols=2, figsize=(2*6, 6))
            axes: list[plt.Axes] = fig_axes_tuple[1]  # type: ignore

            axes[0].set_title("Readout Mask")
            axes[0].imshow(mask_neuron, interpolation="none", cmap="RdBu_r",
                           norm=Normalize(-mask_abs_max, mask_abs_max))

            features_neuron = features[0, :, 0, neuron_id]
            axes[1].set_title("Readout feature weights")
            axes[1].bar(range(features_neuron.shape[0]), features_neuron)
            axes[1].set_ylim((features_min, features_max))

            plot_path = f"{folder_path}/neuron_{neuron_id}.pdf"
            fig_axes_tuple[0].savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
            fig_axes_tuple[0].clf()
            plt.close()


class ReadoutWrapper(torch.nn.ModuleDict):
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
            response = torch.concatenate(readout_responses, dim=0)
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
            self._modules[key].save_weight_visualizations(readout_folder)


class CoreWrapper(torch.nn.Module):
    def __init__(self,
                 channels: tuple[int, ...],
                 temporal_kernel_sizes: tuple[int, ...],
                 spatial_kernel_sizes: tuple[int, ...],
                 gamma_input: float = 0.3,
                 gamma_temporal: float = 40.0,
                 gamma_in_sparse: float = 1.0,
                 cut_first_n_temporal_frames: int = 30,
                 ):
        # Input validation
        if len(channels) < 2:
            raise ValueError(f"At least two channels required (input and output channel), {channels=}")
        if len(temporal_kernel_sizes) != len(channels) - 1:
            raise ValueError(f"{len(channels) - 1} layers, but only {len(temporal_kernel_sizes)} "
                             f"temporal kernel sizes. {channels=} {temporal_kernel_sizes=}")
        if len(temporal_kernel_sizes) != len(spatial_kernel_sizes):
            raise ValueError(f"Temporal and spatial kernel sizes must have the same length."
                             f"{temporal_kernel_sizes=} {spatial_kernel_sizes=}")

        super().__init__()
        self.gamma_input = gamma_input
        self.gamma_temporal = gamma_temporal
        self.gamma_in_sparse = gamma_in_sparse
        self._cut_first_n_temporal_frames = cut_first_n_temporal_frames

        self.features = torch.nn.Sequential()
        for layer_id, (num_in_channels, num_out_channels) in enumerate(
                zip(channels[:-1], channels[1:], strict=True)):
            layer: dict[str, torch.nn.Module] = OrderedDict()
            padding = "same"  # ((temporal_kernel_sizes[layer_id] - 1) // 2, (spatial_kernel_sizes[layer_id] - 1) // 2, (spatial_kernel_sizes[layer_id] - 1) // 2)
            layer["conv"] = STSeparableBatchConv3d(
                num_in_channels,
                num_out_channels,
                log_speed_dict={},
                temporal_kernel_size=temporal_kernel_sizes[layer_id],
                spatial_kernel_size=spatial_kernel_sizes[layer_id],
                bias=False,
                padding=padding,
            )
            layer["norm"] = nn.BatchNorm3d(num_out_channels, momentum=0.1, affine=True)
            layer["bias"] = Bias3DLayer(num_out_channels)
            layer["nonlin"] = torch.nn.ELU()
            # layer["dropout"] = torch.nn.Dropout(p=0.5)
            # layer["dropout"] = torch.nn.Dropout3d(p=0.5)
            if layer_id % 2 == 0:
                layer["pool"] = torch.nn.MaxPool3d((1, 2, 2))
            self.features.add_module(f"layer{layer_id}", nn.Sequential(layer))  # type: ignore

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        res = self.features(input_)
        # To keep compatibility with hoefling model scores
        res_cut = res[:, :, self._cut_first_n_temporal_frames:, :, :]
        return res_cut

    def spatial_laplace(self) -> torch.Tensor:
        return 0.0  # type: ignore

    def temporal_smoothness(self) -> torch.Tensor:
        results = [temporal_smoothing(x.conv.sin_weights, x.conv.cos_weights) for x in self.features]
        return torch.sum(torch.stack(results))

    def group_sparsity_0(self) -> torch.Tensor:
        result_array = []
        for layer in self.features:
            result = (layer.conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1) /
                      torch.sqrt(1e-8 + layer.conv.weight_spatial.pow(2).sum([1, 2, 3, 4])))
            result_array.append(result.sum())

        return torch.sum(torch.stack(result_array))

    def regularizer(self) -> torch.Tensor:
        res = self.spatial_laplace() * self.gamma_input
        res += self.temporal_smoothness() * self.gamma_temporal
        res += self.group_sparsity_0() * self.gamma_in_sparse
        return res

    def save_weight_visualizations(self, folder_path: str) -> None:
        for i, layer in enumerate(self.features):
            output_dir = os.path.join(folder_path, f"weights_layer_{i}")
            os.makedirs(output_dir, exist_ok=True)
            layer.conv.save_weight_visualizations(output_dir)
            print(f"Saved weight visualization at path {output_dir}")


class CoreReadout(lightning.LightningModule):
    def __init__(
            self,
            in_channels: int,
            features_core: Iterable[int],
            temporal_kernel_sizes: Iterable[int],
            spatial_kernel_sizes: Iterable[int],
            in_shape: Iterable[int],
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
            learning_rate: float = 0.01,
            device: str = "cuda",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.core = CoreWrapper(
            channels=(in_channels, ) + tuple(features_core),
            temporal_kernel_sizes=tuple(temporal_kernel_sizes),
            spatial_kernel_sizes=tuple(spatial_kernel_sizes),
        )
        # Run one forward path to determine output shape of core
        example_input = torch.zeros((1, ) + tuple(in_shape))
        core_test_output = self.core.to(device).forward(example_input.to(device))
        in_shape_readout: tuple[int, int, int, int] = core_test_output.shape[1:]  # type: ignore
        print(f"{example_input.shape[1:]=} {in_shape_readout=}")

        self.readout = ReadoutWrapper(
            in_shape_readout, n_neurons_dict, scale, bias, gaussian_masks, gaussian_mean_scale, gaussian_var_scale,
            positive, gamma_readout, gamma_masks, readout_reg_avg
        )
        self.learning_rate = learning_rate
        self.loss = PoissonLoss3d()
        self.correlation_loss = CorrelationLoss3d(avg=True)

    def forward(self, x: torch.Tensor, data_key: str) -> torch.Tensor:
        # first experiment, downpool to 18x16
        # x = torch.nn.functional.avg_pool3d(x, kernel_size=[1, 4, 4])
        output_core = self.core(x)
        output_readout = self.readout(output_core, data_key=data_key)
        return output_readout

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets)
        regularization_loss_core = self.core.regularizer()
        regularization_loss_readout = self.readout.regularizer(session_id)
        self.log("loss", loss)
        self.log("regularization_loss_core", regularization_loss_core)
        self.log("regularization_loss_readout", regularization_loss_readout)
        total_loss = loss + regularization_loss_core + regularization_loss_readout

        return total_loss

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)
        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[str, DataPoint], batch_idx: int, dataloader_idx) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(data_point.inputs, session_id)
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)
        self.log_dict({
            "test_loss": loss,
            "test_correlation": correlation,
        })

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_decay_factor = 0.3
        patience = 5
        tolerance = 0.0005
        min_lr = self.learning_rate * (lr_decay_factor ** 3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay_factor,
            patience=patience,
            threshold=tolerance,
            verbose=True,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_correlation",
                "frequency": 1,
            },
        }

    def save_weight_visualizations(self, folder_path: str) -> None:
        self.core.save_weight_visualizations(os.path.join(folder_path, "core"))
        self.readout.save_weight_visualizations(os.path.join(folder_path, "readout"))
