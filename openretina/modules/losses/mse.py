import torch.nn


class MSE3d(torch.nn.Module):
    """Mean squared error loss for 3D (batch, time, neurons) predictions.
    Handles temporal lag between target and output."""

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :]
        return loss.pow(2).sum(dim=-1).sum()
