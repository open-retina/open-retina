from typing import Callable, List, Optional

import torch
from torch import Tensor

from openretina.model_analysis.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.model_analysis.stimulus_optimization.regularizer import StimulusPostprocessor, StimulusRegularizationLoss


def optimize_stimulus(
    stimulus: Tensor,
    optimizer_init_fn: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
    objective_object,
    optimization_stopper: OptimizationStopper,
    stimulus_regularization_loss: Optional[StimulusRegularizationLoss] = None,
    stimulus_postprocessor: Optional[StimulusPostprocessor] = None,
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
        if stimulus_regularization_loss is not None:
            regularization_loss = stimulus_regularization_loss.forward(stimulus)
            loss += regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if stimulus_postprocessor is not None:
            stimulus.data = stimulus_postprocessor.process(stimulus.data)
        if optimization_stopper.early_stop(float(loss.item())):
            break
    stimulus.detach_()  # Detach the tensor from the computation graph
