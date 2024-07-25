# Implementation according to https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder

import torch
from torch import nn
import lightning


class TorchSTSeparableConv3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temporal_kernel_size: int,
            spatial_kernel_sizes: tuple[int, int],
            stride: int = 1,
            padding_spatial: None | int | tuple[int, int, int] = None,
            padding_temporal: None | int | tuple[int, int, int] = None,
            bias: bool = True,
    ):
        super().__init__()

        if padding_spatial is None:
            padding_spatial = (0, (spatial_kernel_sizes[0]-1) // 2, (spatial_kernel_sizes[1]-1) // 2)
        if padding_temporal is None:
            padding_temporal = ((temporal_kernel_size-1) // 2, 0, 0)

        self.space_conv = nn.Conv3d(
            in_channels,
            out_channels,
            (1, ) + spatial_kernel_sizes,
            stride=stride,
            padding=padding_spatial,
            bias=bias,
        )
        self.time_conv = nn.Conv3d(
            out_channels, out_channels, (temporal_kernel_size, 1, 1), stride=stride,
            padding=padding_temporal, bias=bias,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:

        space_conv = self.space_conv(input_)
        res = self.time_conv(space_conv)

        return res


class NextFramePredictionModel(lightning.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = 0.0005,
            frame_offset: int = 3,
    ):
        if frame_offset < 0:
            raise ValueError(f"Frame offset has to be positive, but was: {frame_offset=}")
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.frame_offset = frame_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_out = self.model.forward(x)
        return model_out

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # make decoder weight unit norm

        session_key, (input_, _) = batch
        reconstructed = self.forward(input_)

        reconstructed_offset = reconstructed[:, :, :-self.frame_offset]
        input_offset = input_[:, :, self.frame_offset:]
        total_loss = torch.nn.functional.mse_loss(reconstructed_offset, input_offset)
        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True, logger=True, on_step=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

