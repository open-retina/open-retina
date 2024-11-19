import torch
import torch.nn as nn


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


class FiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) is a neural network module that applies
    conditional scaling and shifting to input features.

    This module takes input features and a conditioning tensor, computes scaling (gamma)
    and shifting (beta) parameters from the conditioning tensor, and applies these parameters to
    the input features. The result is a modulated output that can adapt based on the provided conditions.

    Args:
        num_features (int): The number of features in the input tensor.
        cond_dim (int): The dimensionality of the conditioning tensor.

    Returns:
        Tensor: The modulated output tensor after applying the scaling and shifting.
    """

    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim

        self.fc_gamma = nn.Linear(cond_dim, num_features)
        self.fc_beta = nn.Linear(cond_dim, num_features)

        # To avoid perturbations in early epochs, we set these defaults to match the identity function
        nn.init.normal_(self.fc_gamma.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_gamma.bias, 1.0)

        nn.init.normal_(self.fc_beta.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_beta.bias, 0.0)

    def forward(self, x, cond):
        # View the conditioning tensor to match the input tensor shape
        gamma = self.fc_gamma(cond).view(cond.size(0), self.num_features, *[1] * (x.dim() - 2))
        beta = self.fc_beta(cond).view(cond.size(0), self.num_features, *[1] * (x.dim() - 2))

        return gamma * x + beta
