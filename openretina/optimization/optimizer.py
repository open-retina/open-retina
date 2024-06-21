from typing import Callable, List, Optional

import torch
from torch import Tensor

from openretina.optimization.optimization_stopper import OptimizationStopper


def optimize_stimulus(
        stimulus: Tensor,
        optimizer_init_fn: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
        objective_object,
        optimization_stopper: OptimizationStopper,
        stimulus_regularizing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        postprocess_stimulus_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> None:
    """
    Optimize a stimulus to maximize a given objective while minimizing a regularizing function.
    The stimulus is modified in place.
    """
    optimizer = optimizer_init_fn([stimulus])

    for i in range(optimization_stopper.max_epochs):
        objective = objective_object.forward(stimulus)
        # Maximizing the objective, minimizing the regularization loss
        loss = -objective
        if stimulus_regularizing_fn is not None:
            regularizer_loss = stimulus_regularizing_fn(stimulus)
            loss += regularizer_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if postprocess_stimulus_fn is not None:
            stimulus.data = postprocess_stimulus_fn(stimulus.data)
        if optimization_stopper.early_stop(float(loss.item())):
            break
    stimulus.detach_()  # Detach the tensor from the computation graph
