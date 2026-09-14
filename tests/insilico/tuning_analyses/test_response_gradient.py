"""Tests for ``openretina.insilico.tuning_analyses.response_gradient``."""

import numpy as np
import pytest
import torch
from torch import nn

from openretina.insilico.stimulus_optimization.objective import IncreaseObjective, MeanReducer
from openretina.insilico.tuning_analyses.response_gradient import (
    MeiAcrossContrasts,
    get_gradient_grid,
    trainer_fn,
)

N_NEURONS = 3


class _LinearPoolModel(nn.Module):
    """Cheap stand-in for a trained model: pool over channels and space, scale per neuron.

    Differentiable in the stimulus, which is all `trainer_fn` needs, and exactly reproducible
    so the grid's contents can be checked against a direct evaluation.
    """

    def __init__(self, n_neurons: int = N_NEURONS):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, n_neurons), requires_grad=False)

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        # (batch, channels, time, height, width) -> (batch, time, neurons)
        pooled = stimulus.mean(dim=(1, 3, 4))
        return pooled.unsqueeze(-1) * self.weight


@pytest.fixture
def objective() -> IncreaseObjective:
    return IncreaseObjective(_LinearPoolModel(), neuron_indices=0, data_key=None, response_reducer=MeanReducer(axis=0))


@pytest.fixture
def stimulus() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 2, 4, 3, 3)


@pytest.mark.parametrize(
    ("start", "stop", "step_size"),
    [
        # Narrower than the old hard-coded [-1, 1]: the loop used to run more iterations than the
        # grid had room for, so this raised IndexError outright.
        (0.0, 1.0, 0.5),
        # Wider than [-1, 1]: the loop used to run FEWER iterations than the grid, so the tail was
        # left as zeros and every filled cell held a contrast from the wrong range. Silent.
        (-2.0, 2.0, 1.0),
    ],
)
def test_grid_spans_the_requested_contrast_range(
    objective: IncreaseObjective, stimulus: torch.Tensor, start: float, stop: float, step_size: float
) -> None:
    grid, resp_grid, norm_grid, green_values, uv_values = get_gradient_grid(
        stimulus, objective, start=start, stop=stop, step_size=step_size
    )

    expected_values = np.arange(start, stop + step_size, step_size)
    np.testing.assert_allclose(green_values, expected_values)
    np.testing.assert_allclose(uv_values, expected_values)

    assert grid.shape == (2, len(expected_values), len(expected_values))
    assert resp_grid.shape == norm_grid.shape == (len(expected_values), len(expected_values))


def test_every_cell_holds_the_response_at_its_own_contrast(
    objective: IncreaseObjective, stimulus: torch.Tensor
) -> None:
    """Pins cell (i, j) to the contrast pair the axes claim it was evaluated at.

    This is what the hard-coded range actually broke: for any range other than [-1, 1] the cells
    were filled from a different set of contrasts than the returned axis arrays described.
    """
    grid, resp_grid, _, green_values, uv_values = get_gradient_grid(
        stimulus, objective, start=-2.0, stop=2.0, step_size=1.0
    )

    for i, contrast_green in enumerate(green_values):
        for j, contrast_uv in enumerate(uv_values):
            expected_grad, expected_resp = trainer_fn(
                MeiAcrossContrasts(torch.Tensor([contrast_green, contrast_uv]), stimulus), objective, lr=0.1
            )
            np.testing.assert_allclose(grid[:, i, j], expected_grad, rtol=1e-5)
            np.testing.assert_allclose(resp_grid[i, j], expected_resp, rtol=1e-5)
