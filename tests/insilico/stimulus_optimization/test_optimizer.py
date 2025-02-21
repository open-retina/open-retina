from functools import partial

import pytest

import torch
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.objective import AbstractObjective


class _SimpleIncreaseObjective(AbstractObjective):
    def __init__(self):
        # Pass None to both arguments as we don't use the model nor data key
        super().__init__(None, None)

    def forward(self, stimulus):
        return torch.sum(stimulus)


@pytest.mark.parametrize(
    "stimulus_shape, lr",
    [
        ((1, 2, 3, 5, 7), 0.1),
        ((1, 2, 3, 5, 7), 10.0),
    ],
)
def test_optimize_stimulus(
    stimulus_shape: tuple[int, int, int, int, int],
    lr: float,
):
    stimulus = torch.randn(stimulus_shape, requires_grad=True)
    objective = _SimpleIncreaseObjective()
    initial_score = objective.forward(stimulus)

    optimize_stimulus(
        stimulus,
        optimizer_init_fn=partial(torch.optim.SGD, lr=lr),
        objective_object=objective,
        optimization_stopper=OptimizationStopper(max_iterations=10),
    )
    new_score = objective.forward(stimulus)

    assert stimulus.shape == stimulus_shape, "Stimulus shape should stay the same"
    assert new_score > initial_score, "Score should increase"
