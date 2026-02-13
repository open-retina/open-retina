import torch
from torch import nn


class LightweightVideoDecoder(nn.Module):
    """
    Lightweight decoder mapping core features to frame predictions.

    The decoder intentionally keeps a low parameter count so most modeling capacity
    remains in the core.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 1,
        kernel_size: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"{num_layers=} must be >= 1")
        if kernel_size < 1:
            raise ValueError(f"{kernel_size=} must be >= 1")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv3d(
                    in_channels=current_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ELU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout3d(p=dropout_rate))
            current_channels = hidden_channels

        layers.append(
            nn.Conv3d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def regularizer(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)
