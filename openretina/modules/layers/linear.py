import torch


class Bias3DLayer(torch.nn.Module):
    def __init__(self, channels: int, initial: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.bias = torch.nn.Parameter(torch.empty((1, channels, 1, 1, 1)).fill_(initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class Scale2DLayer(torch.nn.Module):
    def __init__(self, num_channels: int, initial: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.scale = torch.nn.Parameter(torch.empty((1, num_channels, 1, 1)).fill_(initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Scale3DLayer(torch.nn.Module):
    def __init__(self, num_channels: int, initial: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.scale = torch.nn.Parameter(torch.empty((1, num_channels, 1, 1, 1)).fill_(initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
