import torch
import torch.nn as nn


class WeightedChannelSumLayer(nn.Module):
    """A layer that reduces  multi-channel input to single-channel input by computing \
        a weighted sum across the channel dimension using the provided weights.
    If the input only has a single channel, it will return it unchanged.
    One use case is of this layer is to convert a multi-color input into a grey-scale input to the model."""

    def __init__(self, init_channel_weights: tuple[float, ...], trainable: bool = False):
        super().__init__()

        # add the channel weights
        self.channel_weights = nn.Parameter(torch.tensor(init_channel_weights), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """If the input is not already single-channel (i.e. greyscale), take a weighted sum over channels ."""

        if x.shape[1] == 1:
            return x

        weighted_input = x * (self.channel_weights.view(1, -1, 1, 1, 1))
        squashed = torch.sum(weighted_input, dim=1, keepdim=True)
        return squashed
