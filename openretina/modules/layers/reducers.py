import torch
import torch.nn as nn


class WeightedChannelSumLayer(nn.Module):
    """A layer that can reduced color inputs to greyscale by taking a weighted sum of the channels.
    If the input is already greyscale, it will return the input unchanged."""

    def __init__(self, init_channel_weights: tuple[float, ...], trainable: bool = False):
        super().__init__()

        # add the channel weights
        self.channel_weights = nn.Parameter(torch.tensor(init_channel_weights), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """If the input is not greyscale, take a weighted sum to turn it greyscale."""

        if x.shape[1] == 1:
            return x

        weighted_input = x * (self.channel_weights.view(1, -1, 1, 1, 1))
        squashed = torch.sum(weighted_input, dim=1, keepdim=True)
        return squashed
