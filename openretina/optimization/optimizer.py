from typing import Callable, List, Optional
from sys import maxsize

import torch
from torch import Tensor


class OptimizationStopper:
    def __init__(self, max_epochs: Optional[int] = None):
        if max_epochs is None:
            self.max_epochs = maxsize
        else:
            self.max_epochs = max_epochs

    def early_stop(self, loss: float) -> bool:
        return False


class EarlyStopper(OptimizationStopper):
    def __init__(self,
                 max_epochs: Optional[int] = None,
                 patience: int = 1,
                 min_delta: float = 0.0):
        super().__init__(max_epochs)
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_loss = float('inf')

    def early_stop(self, loss: float) -> bool:
        if loss < self._min_loss:
            self._min_loss = loss
            self._counter = 0
        elif loss > (self._min_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True
        return False


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
