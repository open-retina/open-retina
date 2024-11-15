import os
from collections import OrderedDict
from typing import Iterable, Optional

import lightning
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from openretina.hoefling_2024.models import Bias3DLayer, STSeparableBatchConv3d, temporal_smoothing
from openretina.measures import CorrelationLoss3d, PoissonLoss3d
from openretina.models.core_readout import CoreReadout, ReadoutWrapper
from openretina.models.dev_models import DynamicLayerNorm
from openretina.models.photoreceptor_layer import PhotoreceptorLayer


class PhotorReceptorCoreReadout(CoreReadout):
    def __init__(
        self,
        in_channels: int,
        features_core: Iterable[int],
        temporal_kernel_sizes: Iterable[int],
        spatial_kernel_sizes: Iterable[int],
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        readout_scale: bool,
        readout_bias: bool,
        readout_gaussian_masks: bool,
        readout_gaussian_mean_scale: float,
        readout_gaussian_var_scale: float,
        readout_positive: bool,
        readout_gamma: float,
        readout_gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
        learning_rate: float = 0.01,
        cut_first_n_frames_in_core: int = 30,
        dropout_rate: float = 0.0,
        maxpool_every_n_layers: Optional[int] = None,
        downsample_input_kernel_size: Optional[tuple[int, int, int]] = None,
        photoreceptor_layer_params: Optional[dict[str, float]] = None,
    ):
        # Want methods from CoreReadout, but with different init (same as base lightning module)
        lightning.LightningModule.__init__(self)

        self.save_hyperparameters()
        self.core = PhotoreceptorCoreWrapper(  # type: ignore
            photoreceptor_params=photoreceptor_layer_params,
            in_shape=in_shape,
            channels=(in_channels,) + tuple(features_core),
            temporal_kernel_sizes=tuple(temporal_kernel_sizes),
            spatial_kernel_sizes=tuple(spatial_kernel_sizes),
            cut_first_n_frames=cut_first_n_frames_in_core,
            dropout_rate=dropout_rate,
            maxpool_every_n_layers=maxpool_every_n_layers,
            downsample_input_kernel_size=downsample_input_kernel_size,
        )
        # Run one forward pass to determine output shape of core
        core_test_output = self.core.forward(torch.zeros((1,) + tuple(in_shape)))
        in_shape_readout: tuple[int, int, int, int] = core_test_output.shape[1:]  # type: ignore
        print(f"{in_shape_readout=}")

        self.readout = ReadoutWrapper(
            in_shape_readout,
            n_neurons_dict,
            readout_scale,
            readout_bias,
            readout_gaussian_masks,
            readout_gaussian_mean_scale,
            readout_gaussian_var_scale,
            readout_positive,
            readout_gamma,
            readout_gamma_masks,
            readout_reg_avg,
        )
        self.learning_rate = learning_rate
        self.loss = PoissonLoss3d()
        self.correlation_loss = CorrelationLoss3d(avg=True)


class PhotoreceptorCoreWrapper(torch.nn.Module):
    def __init__(
        self,
        photoreceptor_params: Optional[dict[str, float]],
        in_shape: tuple[int, int, int, int],
        channels: tuple[int, ...],
        temporal_kernel_sizes: tuple[int, ...],
        spatial_kernel_sizes: tuple[int, ...],
        gamma_input: float = 0.3,
        gamma_temporal: float = 40.0,
        gamma_in_sparse: float = 1.0,
        dropout_rate: float = 0.0,
        cut_first_n_frames: int = 30,
        maxpool_every_n_layers: Optional[int] = None,
        downsample_input_kernel_size: Optional[tuple[int, int, int]] = None,
    ):
        # Input validation
        if len(channels) < 2:
            raise ValueError(f"At least two channels required (input and output channel), {channels=}")
        if len(temporal_kernel_sizes) != len(channels) - 1:
            raise ValueError(
                f"{len(channels) - 1} layers, but only {len(temporal_kernel_sizes)} "
                f"temporal kernel sizes. {channels=} {temporal_kernel_sizes=}"
            )
        if len(temporal_kernel_sizes) != len(spatial_kernel_sizes):
            raise ValueError(
                f"Temporal and spatial kernel sizes must have the same length."
                f"{temporal_kernel_sizes=} {spatial_kernel_sizes=}"
            )

        super().__init__()
        self.gamma_input = gamma_input
        self.gamma_temporal = gamma_temporal
        self.gamma_in_sparse = gamma_in_sparse
        self._cut_first_n_frames = cut_first_n_frames
        self._downsample_input_kernel_size = (
            list(downsample_input_kernel_size) if downsample_input_kernel_size is not None else None
        )

        self.features = torch.nn.Sequential()
        height, width = in_shape[-2], in_shape[-1]
        # Add photoreceptor layer before anything else
        self.features.add_module(
            "photoreceptor_layer",
            nn.Sequential(
                Rearrange("b c t h w -> b c t (h w)"),
                PhotoreceptorLayer(pr_params=photoreceptor_params, units=channels[0]),
                Rearrange("b c t (h w) -> b c t h w", h=height, w=width),
                DynamicLayerNorm(norm_axes=[2]),
            ),
        )

        for layer_id, (num_in_channels, num_out_channels) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
            layer: dict[str, torch.nn.Module] = OrderedDict()
            padding = "same"  # ((temporal_kernel_sizes[layer_id] - 1) // 2,
            # (spatial_kernel_sizes[layer_id] - 1) // 2, (spatial_kernel_sizes[layer_id] - 1) // 2)
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
            if dropout_rate > 0.0:
                layer["dropout"] = torch.nn.Dropout3d(p=dropout_rate)
            if maxpool_every_n_layers is not None and (layer_id % maxpool_every_n_layers) == 0:
                layer["pool"] = torch.nn.MaxPool3d((1, 2, 2))
            self.features.add_module(f"layer{layer_id}", nn.Sequential(layer))  # type: ignore

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self._downsample_input_kernel_size is not None:
            input_ = torch.nn.functional.avg_pool3d(input_, kernel_size=self._downsample_input_kernel_size)  # type: ignore
        res = self.features(input_)
        # To keep compatibility with hoefling model scores
        res_cut = res[:, :, self._cut_first_n_frames :, :, :]
        return res_cut

    def spatial_laplace(self) -> torch.Tensor:
        return 0.0  # type: ignore

    def temporal_smoothness(self) -> torch.Tensor:
        results = [
            temporal_smoothing(x.conv.sin_weights, x.conv.cos_weights) for x in self.features if hasattr(x, "conv")
        ]
        return torch.sum(torch.stack(results))

    def group_sparsity_0(self) -> torch.Tensor:
        result_array = []
        for layer in self.features:
            if not hasattr(layer, "conv"):
                continue
            result = layer.conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(
                1e-8 + layer.conv.weight_spatial.pow(2).sum([1, 2, 3, 4])
            )
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
