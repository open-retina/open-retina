from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim

from openretina.insilico.stimulus_optimization.objective import IncreaseObjective


class MeiAcrossContrasts(nn.Module):
    """
    PyTorch Module wrapper around a stimulus to allow tracking response gradients onto the stimulus.
    """

    def __init__(self, contrast_values: torch.Tensor, stim: torch.Tensor):
        """
        Parameters:
        contrast_values (torch.Tensor): A torch Tensor with 2 scalar values that the MEI green and UV channels will be
        multiplied with.
        stim (torch.Tensor): MEI stimulus as torch Tensor.
        """
        super().__init__()
        self.contrast_values = nn.Parameter(contrast_values, requires_grad=True)
        self.stim = nn.Parameter(stim, requires_grad=False)

    def forward(self):
        """
        Multiplies each channel of the stimulus with the contrast value in the corresponding channel.
        This yields the stimulus at the location in the contrast grid specified by the contrast values.
        """
        return torch.mul(
            torch.stack(
                [
                    torch.ones_like(self.stim[:, 0, ...]) * self.contrast_values[0],
                    torch.ones_like(self.stim[:, 0, ...]) * self.contrast_values[1],
                ],
                dim=1,
            ),
            self.stim.squeeze(),
        )


def trainer_fn(
    mei_contrast_gen: MeiAcrossContrasts,
    model_neuron: IncreaseObjective,
    optimizer_class=optim.Adam,
    lr: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trainer function for getting the gradient on the MEI with different contrasts
    """
    optimizer = optimizer_class(mei_contrast_gen.parameters(), lr=lr)
    loss = model_neuron.forward(mei_contrast_gen())
    loss.backward()
    grad_val: torch.Tensor = deepcopy(mei_contrast_gen.contrast_values.grad)  # type: ignore
    optimizer.zero_grad()
    return grad_val.detach().cpu().numpy().squeeze(), loss.detach().cpu().numpy().squeeze()


def get_gradient_grid(
    stim: torch.Tensor,
    model_neuron: IncreaseObjective,
    n_channels: int = 2,
    start: float = -1,
    stop: float = 1,
    step_size: float = 0.1,
) -> tuple:
    """
    Generate a grid of response gradients for a given stimulus and model neuron.

    Args:
        stim (torchTensor, 1 x n_channels x time x height x width): The MEI stimulus.
        model_neuron (IncreaseObjective): The model neuron objective.
        n_channels (int, optional): The number of channels. Defaults to 2.
        start (float, optional): The starting value for the contrast range. Defaults to -1.
        stop (float, optional): The ending value for the contrast range. Defaults to 1.
        step_size (float, optional): The step size for the contrast range. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the following elements:
            - grid (ndarray): A grid of gradient values with shape (n_channels, len(green_contrast_values),
            len(uv_contrast_values)).
            - resp_grid (ndarray): A grid of loss values with shape (len(green_contrast_values),
            len(uv_contrast_values)).
            - norm_grid (ndarray): A grid of norm values with shape (len(green_contrast_values),
            len(uv_contrast_values)).
            - green_contrast_values (ndarray): An array of green contrast values.
            - uv_contrast_values (ndarray): An array of UV contrast values.
    """
    green_contrast_values = np.arange(start, stop + step_size, step_size)
    uv_contrast_values = np.arange(start, stop + step_size, step_size)
    grid = np.zeros((n_channels, len(green_contrast_values), len(uv_contrast_values)))
    resp_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))
    norm_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))

    for i, contrast_green in enumerate(np.arange(-1, 1 + step_size, step_size)):
        for j, contrast_uv in enumerate(np.arange(-1, 1 + step_size, step_size)):
            mei_contrast_gen = MeiAcrossContrasts(torch.Tensor([contrast_green, contrast_uv]), stim)
            response_gradient, response = trainer_fn(mei_contrast_gen, model_neuron, lr=0.1)
            grid[0, i, j] = response_gradient[0]
            grid[1, i, j] = response_gradient[1]
            resp_grid[i, j] = response
            norm_grid[i, j] = np.linalg.norm(grid[:, i, j])
    return grid, resp_grid, norm_grid, green_contrast_values, uv_contrast_values


def equalize_channels(stim: torch.Tensor, flip_green: bool = False) -> torch.Tensor:
    """
    Scales the (green and UV) channels of an stim such that they have equal norm,
    and the scaled stim has the same norm as the original stim. Optionally flips sign of green channel
    :param stim: torch.Tensor, shape (1, 2, 50, 18, 16)
    :param flip_green: bool
    :return: equalized_stim: torch.Tensor, shape (1, 2, 50, 18, 16)
    """
    green_chan = stim[0, 0]
    uv_chan = stim[0, 1]
    green_norm = torch.norm(green_chan)
    uv_norm = torch.norm(uv_chan)
    total_norm = torch.norm(stim)
    green_factor = (total_norm / 2) / green_norm
    uv_factor = (total_norm / 2) / uv_norm
    equalized_stim = torch.zeros_like(stim, device=stim.device)
    equalized_stim[0, 0] = green_factor * stim[0, 0]
    if flip_green:
        equalized_stim[0, 0] = -1 * equalized_stim[0, 0]
    equalized_stim[0, 1] = uv_factor * stim[0, 1]
    total_factor = total_norm / torch.norm(equalized_stim)
    equalized_stim = total_factor * equalized_stim
    return equalized_stim
