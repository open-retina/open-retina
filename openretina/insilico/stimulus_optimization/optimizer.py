from typing import Callable

import torch
from torch import Tensor

from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.regularizer import (
    StimulusPostprocessor,
    StimulusRegularizationLoss,
)


def optimize_stimulus(
    stimulus: Tensor,
    optimizer_init_fn: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
    objective_object,
    optimization_stopper: OptimizationStopper,
    stimulus_regularization_loss: list[StimulusRegularizationLoss] | StimulusRegularizationLoss | None = None,
    stimulus_postprocessor: StimulusPostprocessor | None = None,
) -> None:
    """
    Optimize a stimulus to maximize a given objective while minimizing a regularizing function.
    The stimulus is modified in place.
    """
    optimizer = optimizer_init_fn([stimulus])
    if stimulus_postprocessor is None:
        stimulus_postprocessor_list = []
    elif isinstance(stimulus_postprocessor, StimulusPostprocessor):
        stimulus_postprocessor_list = [stimulus_postprocessor]
    else:
        stimulus_postprocessor_list = stimulus_postprocessor

    for _ in range(optimization_stopper.max_iterations):
        objective = objective_object.forward(stimulus)
        # Maximizing the objective, minimizing the regularization loss
        loss = -objective
        if stimulus_regularization_loss is not None:
            regularization_loss = stimulus_regularization_loss.forward(stimulus)
            loss += regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for postprocessor in stimulus_postprocessor_list:
            stimulus.data = postprocessor.process(stimulus.data)
        if optimization_stopper.early_stop(float(loss.item())):
            break
    stimulus.detach_()  # Detach the tensor from the computation graph
