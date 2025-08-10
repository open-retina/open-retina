"""
Adapted from neuralpredictors:
https://github.com/sinzlab/neuralpredictors/blob/v0.3.0.pre/neuralpredictors/regularizers.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Laplace filters
LAPLACE_1D = np.array([-1, 2, -1]).astype(np.float32)[None, None, ...]
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

EPS = 1e-6


def gaussian2d(
    size,
    sigma: float = 5.0,
    gamma: float = 1.0,
    theta: float = 0,
    center: tuple[int, int] = (0, 0),
    normalize: bool = True,
):
    """
    Returns a 2D Gaussian filter.

    Args:
        size (tuple of int, or int): Image height and width.
        sigma (float): std deviation of the Gaussian along x-axis. Default is 5..
        gamma (float): ratio between std devidation along x-axis and y-axis. Default is 1.
        theta (float): Orientation of the Gaussian (in radians). Default is 0.
        center (tuple): The position of the filter. Default is center (0, 0).
        normalize (bool): Whether to normalize the entries. This is computed by
            subtracting the minimum value and then dividing by the max. Default is True.

    Returns:
        2D Numpy array: A 2D Gaussian filter.

    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = (size, size) if isinstance(size, int) else size
    xmax, ymax = (xmax - 1) / 2, (ymax - 1) / 2
    xmin, ymin = -xmax, -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # shift the position
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))

    if normalize:
        gaussian -= gaussian.min()
        gaussian /= gaussian.max()

    return gaussian.astype(np.float32)


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(
        self,
        padding: int | None = None,
        filter_size: int = 3,
        persistent_buffer: bool = True,
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

        self.register_buffer("filter", torch.from_numpy(kernel),
                             persistent=persistent_buffer)
        self.padding_size = kernel.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class Laplace1d(torch.nn.Module):
    def __init__(self, padding: int | None, persistent_buffer: bool = True):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(LAPLACE_1D),
                             persistent=persistent_buffer)
        self.padding_size = LAPLACE_1D.shape[-1] // 2 if padding is None else padding

    def forward(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        agg_fn = torch.mean if avg else torch.sum
        return agg_fn(F.conv1d(x, self.filter, bias=None, padding=self.padding_size)) # type: ignore


class TimeLaplaceL23dnorm(nn.Module):
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
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1)).pow(2)) / (agg_fn(x.view(oc * ic, 1, k1).pow(2)) + EPS)


class FlatLaplaceL23dnorm(nn.Module):
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
        return agg_fn(self.laplace(x.view(oc * ic, 1, k2, k3)).pow(2)) / (
            agg_fn(x.view(oc * ic, 1, k2, k3).pow(2)) + EPS
        )


class GaussianLaplaceL2(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a single 2D convolutional layer.

    """

    def __init__(self, kernel, padding=None):
        """
        Args:
            kernel: Size of the convolutional kernel of the filter that is getting regularized
            padding (int): Controls the amount of zero-padding for the convolution operation.
        """
        super().__init__()

        self.laplace = Laplace(padding=padding)
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        sigma = min(*self.kernel) / 4
        self.gaussian2d = torch.from_numpy(gaussian2d(size=(*self.kernel,), sigma=sigma))

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        out = self.laplace(x.view(oc * ic, 1, k1, k2))
        out = out * (1 - self.gaussian2d.expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / (agg_fn(x.view(oc * ic, 1, k1, k2).pow(2)) + EPS)


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / (
            agg_fn(x.view(oc * ic, 1, k1, k2).pow(2)) + EPS
        )
