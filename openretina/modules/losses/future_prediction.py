import torch
from torch import nn


def _center_crop_along_dim(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    if tensor.size(dim) == size:
        return tensor
    if tensor.size(dim) < size:
        raise ValueError(f"Cannot center-crop dim {dim}: requested {size}, available {tensor.size(dim)}")
    start = (tensor.size(dim) - size) // 2
    return tensor.narrow(dim=dim, start=start, length=size)


class FutureFrameMSELoss(nn.Module):
    """
    MSE loss for frame prediction with automatic shape alignment.

    Predictions and targets are aligned on:
    - time: trailing frames (causal alignment)
    - space: centered crop
    """

    def __init__(self, avg: bool = True):
        super().__init__()
        self.avg = avg

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"Expected 5D tensors (B, C, T, H, W), got {output.shape=} and {target.shape=}")
        if output.size(0) != target.size(0):
            raise ValueError(f"Batch sizes differ: {output.size(0)} != {target.size(0)}")
        if output.size(1) != target.size(1):
            raise ValueError(f"Channel sizes differ: {output.size(1)} != {target.size(1)}")
        if output.size(2) != target.size(2):
            raise ValueError(f"Output and target frames do not match: {output.size(2)=}, {target.size(2)=}")

        common_height = min(output.size(3), target.size(3))
        common_width = min(output.size(4), target.size(4))
        output = _center_crop_along_dim(output, dim=3, size=common_height)
        target = _center_crop_along_dim(target, dim=3, size=common_height)
        output = _center_crop_along_dim(output, dim=4, size=common_width)
        target = _center_crop_along_dim(target, dim=4, size=common_width)

        loss = (output - target).pow(2)
        return loss.mean() if self.avg else loss.sum()

    def __str__(self) -> str:
        return f"FutureFrameMSELoss(avg={self.avg})"
