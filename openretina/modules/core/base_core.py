import os
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from openretina.modules.layers import FlatLaplaceL23dnorm
from openretina.modules.layers.convolutions import STSeparableBatchConv3d, TorchSTSeparableConv3D
from openretina.modules.layers.regularizers import Laplace1d
from openretina.modules.layers.scaling import Bias3DLayer


def temporal_smoothing(sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    smoother = torch.linspace(0.1, 0.9, sin.shape[2], device=sin.device)[None, None, :]
    size = float(sin.shape[0])
    reg = torch.sum((smoother * sin) ** 2) / size
    reg += torch.sum((smoother * cos) ** 2) / size
    return reg


class Core(nn.Module):
    def initialize(self) -> None:
        pass

    def regularizer(self) -> torch.Tensor | float:
        warnings.warn(
            f"Regularizer not implemented for {self.__class__.__name__}", category=RuntimeWarning, stacklevel=2
        )
        return 0.0

    def __repr__(self) -> str:
        s = f"{repr(super())} [{self.__class__.__name__} regularizers: "
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append(f"{attr} = {getattr(self, attr)}")
        return s + "|".join(ret) + "]\n"

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg") -> None:
        print(f"Save weight visualization of {self.__class__.__name__} not implemented.")


class Core3d(Core):
    def initialize(self) -> None:
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


class SimpleCoreWrapper(Core):
    def __init__(
        self,
        channels: tuple[int, ...],
        temporal_kernel_sizes: tuple[int, ...],
        spatial_kernel_sizes: tuple[int, ...],
        gamma_input: float,
        gamma_temporal: float,
        gamma_in_sparse: float,
        gamma_hidden: float,
        dropout_rate: float = 0.0,
        cut_first_n_frames: int = 30,
        maxpool_every_n_layers: int | None = None,
        downsample_input_kernel_size: tuple[int, int, int] | None = None,
        input_padding: bool | int | str | tuple[int, int, int] = False,
        hidden_padding: bool | int | str | tuple[int, int, int] = True,
        convolution_type: str = "sin_cos",
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
        self.convolution_type = convolution_type
        self.gamma_input = gamma_input
        self.gamma_temporal = gamma_temporal
        self.gamma_in_sparse = gamma_in_sparse
        self.gamma_hidden = gamma_hidden
        self._cut_first_n_frames = cut_first_n_frames
        self._downsample_input_kernel_size = (
            list(downsample_input_kernel_size) if downsample_input_kernel_size is not None else None
        )
        if self._cut_first_n_frames and not input_padding:
            warnings.warn(
                (
                    "Cutting frames from the core output can lead to unexpected results if the input is not padded."
                    f"{self._cut_first_n_frames=}, {input_padding=}. Double check the core output shape."
                ),
                UserWarning,
                stacklevel=2,
            )

        self._input_weights_regularizer_spatial = FlatLaplaceL23dnorm(padding=0)
        self._input_weights_regularizer_temporal = Laplace1d(padding=0, persistent_buffer=False)

        self.features = torch.nn.Sequential()
        for layer_id, (num_in_channels, num_out_channels) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
            layer: dict[str, torch.nn.Module] = OrderedDict()
            padding_to_use = input_padding if layer_id == 0 else hidden_padding
            # explictily check against bools as the type can also be an int or a tuple
            if padding_to_use is True:
                padding: str | int | tuple[int, int, int] = "same"
            elif padding_to_use is False:
                padding = 0
            else:
                padding = padding_to_use

            if self.convolution_type == "sin_cos":
                layer["conv"] = STSeparableBatchConv3d(
                    num_in_channels,
                    num_out_channels,
                    log_speed_dict={},
                    temporal_kernel_size=temporal_kernel_sizes[layer_id],
                    spatial_kernel_size=spatial_kernel_sizes[layer_id],
                    bias=False,
                    padding=padding,
                )

                layer["norm"] = torch.nn.BatchNorm3d(num_out_channels, momentum=0.1, affine=True)
                layer["bias"] = Bias3DLayer(num_out_channels)
                layer["nonlin"] = torch.nn.ELU()
                if dropout_rate > 0.0:
                    layer["dropout"] = torch.nn.Dropout3d(p=dropout_rate)
                if maxpool_every_n_layers is not None and (layer_id % maxpool_every_n_layers) == 0:
                    layer["pool"] = torch.nn.MaxPool3d((1, 2, 2))
                self.features.add_module(f"layer{layer_id}", torch.nn.Sequential(layer))  # type: ignore

            elif self.convolution_type == "torch":
                layer["conv"] = TorchSTSeparableConv3D(
                    num_in_channels,
                    num_out_channels,
                    log_speed_dict={},
                    temporal_kernel_size=temporal_kernel_sizes[layer_id],
                    spatial_kernel_size=spatial_kernel_sizes[layer_id],
                    bias=False,
                    padding=padding,
                )
            else:
                raise ValueError(
                    f"Unknown type {convolution_type}. Supported convolution types are 'sin_cos' and 'torch'."
                )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self._downsample_input_kernel_size is not None:
            input_ = torch.nn.functional.avg_pool3d(input_, kernel_size=self._downsample_input_kernel_size)  # type: ignore
        res = self.features(input_)
        # To keep compatibility with hoefling model scores
        res_cut = res[:, :, self._cut_first_n_frames :, :, :]
        return res_cut

    def spatial_laplace(self) -> torch.Tensor:
        return self._input_weights_regularizer_spatial(self.features[0].conv.weight_spatial, avg=False)

    def temporal_laplace(self) -> torch.Tensor:
        ch_in, ch_out, t, h, w = self.features[0].conv.time_conv.weight.shape
        return self._input_weights_regularizer_temporal(
            self.features[0].conv.time_conv.weight.view(ch_in * ch_out, 1, t), avg=False
        )

    def temporal_smoothness(self) -> torch.Tensor:
        if self.convolution_type == "torch":
            return self.temporal_laplace()
        else:
            results = [temporal_smoothing(x.conv.sin_weights, x.conv.cos_weights) for x in self.features]
            return torch.sum(torch.stack(results))

    def group_sparsity_0(self) -> torch.Tensor:
        result_array = []
        for layer in self.features:
            result = layer.conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(
                1e-8 + layer.conv.weight_spatial.pow(2).sum([1, 2, 3, 4])
            )
            result_array.append(result.sum())

        return torch.sum(torch.stack(result_array))

    def group_sparsity(self) -> torch.Tensor:
        sparsities: list[torch.Tensor] = []
        for feat in self.features[1:]:
            val = feat.conv.weight_spatial.pow(2).sum([2, 3, 4]).sqrt().sum(1) / torch.sqrt(
                1e-8 + feat.conv.weight_spatial.pow(2).sum([1, 2, 3, 4])
            )
            sparsities.append(val)
        return torch.sum(torch.stack(sparsities))

    def regularizer(self) -> torch.Tensor:
        res: torch.Tensor = 0.0  # type: ignore
        for weight, reg_fn in [
            (self.gamma_input, self.spatial_laplace),
            (self.gamma_hidden, self.group_sparsity),
            (self.gamma_temporal, self.temporal_smoothness),
            (self.gamma_in_sparse, self.group_sparsity_0),
        ]:
            # lazy calculation of regularization functions
            if weight != 0.0:
                res += weight * reg_fn()
        return res

    def plot_weight_visualization(self, layer: int, in_channel: int, out_channel: int) -> plt.Figure:
        if layer >= len(self.features):
            raise ValueError(f"Requested layer {layer}, but only {len(self.features)} layers present.")
        conv_obj = self.features[layer].conv
        fig = conv_obj.plot_weights(in_channel, out_channel)
        return fig

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg") -> None:
        for i, layer in enumerate(self.features):
            output_dir = os.path.join(folder_path, f"weights_layer_{i}")
            os.makedirs(output_dir, exist_ok=True)
            layer.conv.save_weight_visualizations(output_dir, file_format)
            print(f"Saved weight visualization at path {output_dir}")


class DummyCore(Core):
    """
    A dummy core that does nothing. Used for readout only models, like the LNP model.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, data_key=None, **kwargs):
        return x

    def regularizer(self):
        return 0
