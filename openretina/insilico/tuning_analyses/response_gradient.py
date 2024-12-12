from torch import optim
from torch import nn
from copy import deepcopy
import torch
import numpy as np


class MeiAcrossContrasts(nn.Module):
    """
    Create an MEI stimulus as a PyTorch Module for tracking gradients onto the
    scalars.
    """
    def __init__(self, contrast_values, mei_stim_private):
        """

        :param contrast_values: a torch Tensor with 2 scalar values that the
        MEI green and UV channels will be multiplied with
        :param mei_stim_private: MEI stimulus as torch Tensor
        """
        super().__init__()
        self.contrast_values = nn.Parameter(contrast_values, requires_grad=True)

        self.mei_stim_private = nn.Parameter(mei_stim_private, requires_grad=False)

    def forward(self):
        return torch.mul(
            torch.stack(
                [torch.ones_like(self.mei_stim_private[:, 0, ...]) * self.contrast_values[0],
                 torch.ones_like(self.mei_stim_private[:, 0, ...]) * self.contrast_values[1]
                 ], dim=1),
            self.mei_stim_private.squeeze()
        )


def trainer_fn(mei_contrast_gen, model_neuron, optimizer=optim.Adam, lr=1):
    """
    Trainer function for getting the gradient on the MEI with different contrasts
    """
    optimizer = optimizer(mei_contrast_gen.parameters(), lr=lr)
    loss = model_neuron.forward(mei_contrast_gen())
    loss.backward()
    grad_val = deepcopy(mei_contrast_gen.contrast_values.grad)
    optimizer.zero_grad()
    return grad_val.detach().cpu().numpy().squeeze(), loss.detach().cpu().numpy().squeeze()

def get_gradient_grid(mei_stim: torch.TensorType, 
                      model_neuron, n_channels=2, start=-1, stop=1, step_size=.1):
    """
    Generate a grid of gradient values for a given MEI stimulus and model neuron.

    Args:
        mei_stim (torchTensor, 1 x n_channels x time x height x width): The MEI stimulus.
        model_neuron (SingleNeuronObjective): The model neuron objective.
        n_channels (int, optional): The number of channels. Defaults to 2.
        start (float, optional): The starting value for the contrast range. Defaults to -1.
        stop (float, optional): The ending value for the contrast range. Defaults to 1.
        step_size (float, optional): The step size for the contrast range. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the following elements:
            - grid (ndarray): A grid of gradient values with shape (n_channels, len(green_contrast_values), len(uv_contrast_values)).
            - resp_grid (ndarray): A grid of loss values with shape (len(green_contrast_values), len(uv_contrast_values)).
            - norm_grid (ndarray): A grid of norm values with shape (len(green_contrast_values), len(uv_contrast_values)).
            - green_contrast_values (ndarray): An array of green contrast values.
            - uv_contrast_values (ndarray): An array of UV contrast values.
    """
    green_contrast_values = np.arange(start, stop+step_size, step_size)
    uv_contrast_values = np.arange(start, stop+step_size, step_size)
    grid = np.zeros((n_channels, len(green_contrast_values), len(uv_contrast_values)))
    resp_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))
    norm_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))

    for i, contrast_green in enumerate(np.arange(-1, 1+step_size, step_size)):
        for j, contrast_uv in enumerate(np.arange(-1, 1+step_size, step_size)):
            mei_contrast_gen = MeiAcrossContrasts(torch.Tensor([contrast_green, contrast_uv]), mei_stim)
            response_gradient, response = trainer_fn(mei_contrast_gen, model_neuron, lr=.1)
            grid[0, i, j] = response_gradient[0]
            grid[1, i, j] = response_gradient[1]
            resp_grid[i, j] = response
            norm_grid[i, j] = np.linalg.norm(grid[:, i, j])
    return grid, resp_grid, norm_grid, green_contrast_values, uv_contrast_values


def equalize_channels(mei: torch.Tensor,
                      flip_green: bool = False) -> torch.Tensor:
    '''
    Scales the (green and UV) channels of an MEI such that they have equal norm,
    and the scaled MEI has the same norm as the original MEI. Optionally flips sign of green channel
    :param mei: torch.Tensor, shape (1, 2, 50, 18, 16)
    :param flip_green: bool
    :return: equalized_mei: torch.Tensor, shape (1, 2, 50, 18, 16)
    '''
    green_chan = mei[0, 0]
    uv_chan = mei[0, 1]
    green_norm = torch.norm(green_chan)
    uv_norm = torch.norm(uv_chan)
    total_norm = torch.norm(mei)
    green_factor = (total_norm/2)/green_norm
    uv_factor = (total_norm/2)/uv_norm
    equalized_mei = torch.zeros_like(mei).cuda()
    equalized_mei[0, 0] = green_factor * mei[0, 0]
    if flip_green:
        equalized_mei[0, 0] = -1 * equalized_mei[0, 0]
    equalized_mei[0, 1] = uv_factor * mei[0, 1]
    total_factor = total_norm/torch.norm(equalized_mei)
    equalized_mei = total_factor * equalized_mei
    return equalized_mei
