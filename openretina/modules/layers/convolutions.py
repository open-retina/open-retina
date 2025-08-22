from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize


class TorchFullConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        num_scans=1,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        # Initialize default log speed (batch adaptation term)
        self.register_buffer("_log_speed_default", torch.zeros(1))

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (temporal_kernel_size, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_: torch.Tensor | tuple[torch.Tensor, str]) -> torch.Tensor:
        if type(input_) is torch.Tensor:
            x = input_
            data_key: str | None = None
        else:
            x, data_key = input_

        # Compute temporal kernel based on the provided data key
        # TODO implement log speed use in full conv
        # if data_key is None:
        #    log_speed = self._log_speed_default
        # else:
        #    log_speed = getattr(self, "_".join(["log_speed", data_key]))

        return self.conv(x)


class TorchSTSeparableConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        bias: bool = True,
        num_scans=1,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        # Initialize default log speed (batch adaptation term)
        self.register_buffer("_log_speed_default", torch.zeros(1))

        self.space_conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.time_conv = nn.Conv3d(
            out_channels, out_channels, (temporal_kernel_size, 1, 1), stride=stride, padding=padding, bias=bias
        )

    def forward(self, input_: torch.Tensor | tuple[torch.Tensor, str | None]) -> torch.Tensor:
        if type(input_) is torch.Tensor:
            x = input_
            data_key: str | None = None
        else:
            x, data_key = input_

        # Compute temporal kernel based on the provided data key
        if data_key is None:
            log_speed = self._log_speed_default
        else:
            log_speed = getattr(self, "_".join(["log_speed", data_key]))

        space_conv = self.space_conv(x)
        exp_log_speed = torch.exp(log_speed)  # type: ignore
        return exp_log_speed * self.time_conv(space_conv)


class TimeIndependentConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

        if spatial_kernel_size2 is None:
            spatial_kernel_size2 = spatial_kernel_size

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, spatial_kernel_size, spatial_kernel_size2),
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_):
        if type(input_) is torch.Tensor:
            x = input_
            data_key: str | None = None
        else:
            x, data_key = input_
        return self.conv(x)


def compute_temporal_kernel(log_speed, sin_weights, cos_weights, length: int) -> torch.Tensor:
    """
    Computes the temporal kernel for the convolution.

    Args:
        log_speed (torch.nn.Parameter): Logarithm of the speed factor.
        sin_weights (torch.nn.Parameter): Sinusoidal weights.
        cos_weights (torch.nn.Parameter): Cosine weights.
        length (int): Length of the temporal kernel.

    Returns:
        torch.Tensor: The temporal kernel.
    """
    stretches = torch.exp(log_speed)
    sines, cosines = STSeparableBatchConv3d.temporal_basis(stretches, length)
    weights_temporal = torch.sum(sin_weights[:, :, :, None] * sines[None, None, ...], dim=2) + torch.sum(
        cos_weights[:, :, :, None] * cosines[None, None, ...], dim=2
    )
    return weights_temporal[..., None, None]


class STSeparableBatchConv3d(nn.Module):
    """
    Spatio-temporal separable convolution layer for processing 3D data.

    This layer applies convolution separately in the spatial and temporal dimensions,
    which is efficient for spatio-temporal data like video or medical images.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        temporal_kernel_size (int): Size of the kernel in the temporal dimension.
        spatial_kernel_size (int): Size of the kernel in the spatial dimensions.
        spatial_kernel_size2 (int): Size of the kernel in the second spatial dimension.
        stride (int): Stride of the convolution.
        padding (int): Padding added to all sides of the input.
        num_scans (int): Number of scans for batch processing.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        log_speed_dict: dict,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_kernel_size2: int | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | str | tuple[int, ...] = 0,
        num_scans: int = 1,
        bias: bool = True,
    ):
        """
        Initializes the STSeparableBatchConv3d layer.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            log_speed_dict (dict): Dictionary mapping data keys to log speeds.
            temporal_kernel_size (int): Size of the temporal kernel.
            spatial_kernel_size (int): Size of the spatial kernel.
            spatial_kernel_size2 (int, optional): Size of the second spatial dimension of the kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to all sides of the input. Defaults to 0.
            num_scans (int, optional): Number of scans to process in batch. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_kernel_size2 = spatial_kernel_size2 if spatial_kernel_size2 is not None else spatial_kernel_size
        self.stride = stride
        self.padding = padding
        self.num_scans = num_scans

        # Initialize temporal weights
        self.sin_weights, self.cos_weights = self.temporal_weights(temporal_kernel_size, in_channels, out_channels)

        # Initialize spatial weights
        self.weight_spatial = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, self.spatial_kernel_size, self.spatial_kernel_size2) * 0.01
        )

        # Initialize bias if required
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # Initialize default log speed (batch adaptation term)
        self.register_buffer("_log_speed_default", torch.zeros(1))

        # Store log speeds for each data key
        for key, val in log_speed_dict.items():
            setattr(self, key, val)

    def forward(self, input_: tuple[torch.Tensor, str] | torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STSeparableBatchConv3d layer.

        Args:
            input_ (tuple): Tuple containing the input tensor and the data key.

        Returns:
            torch.Tensor: The output of the convolution.
        """
        if type(input_) is torch.Tensor:
            x = input_
            data_key: str | None = None
        else:
            x, data_key = input_

        # Compute temporal kernel based on the provided data key
        if data_key is None:
            log_speed = self._log_speed_default
        else:
            log_speed = getattr(self, "_".join(["log_speed", data_key]))
        self.weight_temporal = compute_temporal_kernel(
            log_speed, self.sin_weights, self.cos_weights, self.temporal_kernel_size
        )

        # Assemble the complete weight tensor for convolution
        # o - output channels, i - input channels, t - temporal kernel size
        # x - empty dimension, h - spatial kernel size, w - second spatial kernel size
        self.weight = torch.einsum("oitxx,oixhw->oithw", self.weight_temporal, self.weight_spatial)

        # Perform the convolution
        self.conv = F.conv3d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return self.conv

    def get_spatial_weight(self, in_channel: int, out_channel: int) -> np.ndarray:
        spatial_2d_tensor = self.weight_spatial[out_channel, in_channel, 0]
        spatial_2d_np = spatial_2d_tensor.detach().cpu().numpy()
        return spatial_2d_np

    def get_temporal_weight(self, in_channel: int, out_channel: int) -> tuple[np.ndarray, float]:
        weight_temporal = compute_temporal_kernel(
            self._log_speed_default, self.sin_weights, self.cos_weights, self.temporal_kernel_size
        )
        global_abs_max = float(weight_temporal.detach().abs().max().item())
        temporal_trace_tensor = weight_temporal[out_channel, in_channel, :, 0, 0]
        temporal_trace_np = temporal_trace_tensor.detach().cpu().numpy()
        return temporal_trace_np, global_abs_max

    def get_sin_cos_weights(self, in_channel: int, out_channel: int) -> tuple[np.ndarray, np.ndarray]:
        sin_trace = self.sin_weights[out_channel, in_channel].detach().cpu().numpy()
        cos_trace = self.cos_weights[out_channel, in_channel].detach().cpu().numpy()
        return sin_trace, cos_trace

    def plot_weights(
        self,
        in_channel: int,
        out_channel: int,
        plot_log_speed: bool = False,
        add_titles: bool = True,
        remove_ticks: bool = False,
        spatial_weight_center_positive: bool = True,
    ) -> plt.Figure:
        ncols = 4 if plot_log_speed else 3
        fig_axes_tuple = plt.subplots(ncols=ncols, figsize=(ncols * 6, 6))
        fig: plt.Figure = fig_axes_tuple[0]
        axes: list[plt.Axes] = fig_axes_tuple[1]  # type: ignore
        spatial_weight = self.get_spatial_weight(in_channel, out_channel)
        temporal_weight, temporal_abs_max = self.get_temporal_weight(in_channel, out_channel)
        sin_trace, cos_trace = self.get_sin_cos_weights(in_channel, out_channel)

        center_x, center_y = int(spatial_weight.shape[0] / 2), int(spatial_weight.shape[1] / 2)
        # Optionally make sure the center of the weight matrix is positive
        if (
            spatial_weight_center_positive
            and np.mean(spatial_weight[center_x - 3 : center_x + 2, center_y - 3 : center_y + 2]) < 0
        ):
            spatial_weight *= -1
            temporal_weight *= -1
            sin_trace *= -1
            cos_trace *= -1

        abs_max = float(self.weight_spatial.detach().abs().max().item())
        im = axes[0].imshow(
            spatial_weight, interpolation="none", cmap="RdBu_r", norm=Normalize(vmin=-abs_max, vmax=abs_max)
        )
        color_bar = fig.colorbar(im, orientation="vertical")

        axes[1].set_ylim(-temporal_abs_max, temporal_abs_max)
        axes[1].plot(temporal_weight)
        trace_max = max(self.sin_weights.detach().abs().max().item(), self.cos_weights.detach().abs().max().item())
        trace_max *= 1.1

        axes[2].set_ylim(-trace_max, trace_max)
        axes[2].plot(sin_trace, label="sin")
        axes[2].plot(cos_trace, label="cos")
        axes[2].legend()

        if plot_log_speed:
            log_speeds = {
                n[len("log_speed_") :]: float(getattr(self, n).detach().item())
                for n in dir(self)
                if n.startswith("log_speed_")
            }
            log_speeds_names = sorted(log_speeds.keys())
            axes[3].bar(log_speeds_names, [log_speeds[n] for n in log_speeds_names])
            axes[3].set_xticklabels(log_speeds_names, rotation=90)

        if add_titles:
            fig.suptitle(f"Weights for {in_channel=} {out_channel=}")
            axes[0].set_title("Spatial Weight")
            axes[1].set_title("Temporal Weight")
            axes[2].set_title("Sin/Cos Weights")

        if remove_ticks:
            for ax in axes:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            color_bar.set_ticks([])

        return fig

    def save_weight_visualizations(self, folder_path: str, file_format: str = "jpg") -> None:
        for in_channel in range(self.in_channels):
            for out_channel in range(self.out_channels):
                plot_path = f"{folder_path}/{in_channel}_{out_channel}.{file_format}"
                fig = self.plot_weights(in_channel, out_channel)
                fig.savefig(plot_path, bbox_inches="tight", facecolor="w", dpi=300)
                fig.clf()
                plt.close()

    @staticmethod
    def temporal_weights(
        length: int, num_channels: int, num_feat: int, scale: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates initial weights for the temporal components of the convolution.

        Args:
            length (int): Length of the temporal kernel.
            num_channels (int): Number of input channels.
            num_feat (int): Number of output features.
            scale (float, optional): Scaling factor for weight initialization. Defaults to 0.01.

        Returns:
            tuple: Tuple containing sin and cos weights.
        """
        K = length // 3
        sin_weights = torch.nn.Parameter(data=torch.randn(num_feat, num_channels, K) * scale, requires_grad=True)
        cos_weights = torch.nn.Parameter(data=torch.randn(num_feat, num_channels, K) * scale, requires_grad=True)
        return sin_weights, cos_weights

    @staticmethod
    def temporal_basis(stretches: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the basis for the temporal component of the convolution.

        Args:
            stretches (torch.Tensor): Temporal stretches per ROI.
            T (int): Length of the temporal kernel.

        Returns:
            tuple: Tuple containing sines and cosines tensors.
        """
        K = T // 3
        time = torch.arange(T, dtype=torch.float, device=stretches.device) - T
        stretched = stretches * time
        freq = stretched * 2 * np.pi / T
        mask = STSeparableBatchConv3d.mask_tf(time, stretches, T)
        sines, cosines = [], []
        for k in range(K):
            sines.append(mask * torch.sin(freq * k))
            cosines.append(mask * torch.cos(freq * k))
        sines_stacked = torch.stack(sines, 0)
        cosines_stacked = torch.stack(cosines, 0)
        return sines_stacked, cosines_stacked

    @staticmethod
    def mask_tf(time: torch.Tensor, stretch: torch.Tensor, T: int) -> torch.Tensor:
        """
        Generates a mask for the temporal basis functions.

        Args:
            time (torch.Tensor): Time tensor.
            stretch (torch.Tensor): Stretch tensor.
            T (int): Length of the temporal kernel.

        Returns:
            torch.Tensor: The mask tensor.
        """
        mask = 1 / (1 + torch.exp(-time - int(T * 0.95) / stretch))
        return mask.T if mask.ndim > 1 else mask


def temporal_smoothing(sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    smoother = torch.linspace(0.1, 0.9, sin.shape[2], device=sin.device)[None, None, :]
    F = float(sin.shape[0])
    reg = torch.sum((smoother * sin) ** 2) / F
    reg += torch.sum((smoother * cos) ** 2) / F
    return reg


def get_conv_class(conv_type: str) -> type[nn.Module]:
    if conv_type == "separable":
        return TorchSTSeparableConv3D
    elif conv_type == "custom_separable":
        return STSeparableBatchConv3d
    elif conv_type == "full":
        return TorchFullConv3D
    elif conv_type == "time_independent":
        return TimeIndependentConv3D
    else:
        raise ValueError(f"Un-implemented conv_type {conv_type}")
