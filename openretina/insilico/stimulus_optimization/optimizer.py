from typing import Callable, TypeVar

import torch
from torch import Tensor

from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.regularizer import (
    StimulusPostprocessor,
    StimulusRegularizationLoss,
)

T = TypeVar('T')


def convert_to_list(x: list[T] | T | None) -> list[T]:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def optimize_stimulus(
    stimulus: Tensor,
    optimizer_init_fn: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
    objective_object,
    optimization_stopper: OptimizationStopper,
    stimulus_regularization_loss: list[StimulusRegularizationLoss] | StimulusRegularizationLoss | None = None,
    stimulus_postprocessor: list[StimulusPostprocessor] | StimulusPostprocessor | None = None,
) -> None:
    """
    Optimize a stimulus to maximize a given objective while minimizing a regularizing function.
    The stimulus is modified in place.
    """
    optimizer = optimizer_init_fn([stimulus])

    for _ in range(optimization_stopper.max_iterations):
        objective = objective_object.forward(stimulus)
        # Maximizing the objective, minimizing the regularization loss
        loss = -objective
        for reg_loss_module in convert_to_list(stimulus_regularization_loss):
            regularization_loss = reg_loss_module.forward(stimulus)
            loss += regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for postprocessor in convert_to_list(stimulus_postprocessor):
            stimulus.data = postprocessor.process(stimulus.data)
        if optimization_stopper.early_stop(float(loss.item())):
            break
    stimulus.detach_()  # Detach the tensor from the computation graph
