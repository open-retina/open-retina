import torch
import torch.nn as nn


class WeightedChannelSumLayer(nn.Module):
    """A layer that reduces the first channel of its input into one dimension according to the provided weights.
    If the first channel is already one dimensional, will return it unchanged.
    One use case is of this layer is to convert a multi-color channel into a grey-scale channel."""

    def __init__(self, init_channel_weights: tuple[float, ...], trainable: bool = False):
        super().__init__()

        # add the channel weights
        self.channel_weights = nn.Parameter(torch.tensor(init_channel_weights), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """If the input is not multi-dimensional, take a weighted sum to turn it greyscale."""

        if x.shape[1] == 1:
            return x

        weighted_input = x * (self.channel_weights.view(1, -1, 1, 1, 1))
        squashed = torch.sum(weighted_input, dim=1, keepdim=True)
        return squashed
