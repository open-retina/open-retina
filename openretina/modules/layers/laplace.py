import numpy as np
import torch
from torch.functional import F


# Laplace filters
LAPLACE_1D = np.array([-1, 4, -1]).astype(np.float32)[None, None, ...]
LAPLACE_3x3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]
LAPLACE_5x5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ]
).astype(np.float32)[None, None, ...]
LAPLACE_7x7 = np.array(
    [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 3, 3, 3, 1, 0],
        [1, 3, 0, -7, 0, 3, 1],
        [1, 3, -7, -24, -7, 3, 1],
        [1, 3, 0, -7, 0, 3, 1],
        [0, 1, 3, 3, 3, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ]
).astype(np.float32)[None, None, ...]


class Laplace(torch.nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(
            self,
            padding: int | None = None,
            filter_size: int = 3,
    ):
        """Laplace filter for a stack of data"""

        super().__init__()
        if filter_size == 3:
            kernel = LAPLACE_3x3
        elif filter_size == 5:
            kernel = LAPLACE_5x5
        elif filter_size == 7:
            kernel = LAPLACE_7x7
        else:
            raise ValueError(f"Unsupported filter size {filter_size}")

        self.register_buffer("filter", torch.from_numpy(kernel))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class Laplace1d(torch.nn.Module):
    def __init__(self, padding: int | None):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(LAPLACE_1D))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.filter, bias=None, padding=self.padding_size)


class TimeLaplaceL23dnorm(torch.nn.Module):
    """
    Normalized Laplace regularizer for the temporal component of a separable 3D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.laplace = Laplace1d(padding=padding)

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2, k3 = x.size()
        assert (k2, k3) == (1, 1), "space dimensions must be one"
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1)).pow(2)) / agg_fn(x.view(oc * ic, 1, k1).pow(2))


class FlatLaplaceL23dnorm(torch.nn.Module):
    """
    Normalized Laplace regularizer for the spatial component of a separable 3D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding: int | None = None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2, k3 = x.size()
        assert k1 == 1, "time dimension must be one"
        return agg_fn(self.laplace(x.view(oc * ic, 1, k2, k3)).pow(2)) / agg_fn(x.view(oc * ic, 1, k2, k3).pow(2))
