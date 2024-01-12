from typing import Callable, List
import torch
from torch import Tensor

def optimize_stimulus(
        stimulus: Tensor,
        optimizer_init_fn: Callable[[List[torch.Tensor]], torch.optim.Optimizer],
        objective,
        stimulus_regularizing_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        max_iterations: int = 10,
) -> None:
    optimizer = optimizer_init_fn(stimulus)

    # when to stop
    for i in range(max_iterations):
        objective = objective.forward(stimulus)
        regularizer_loss = stimulus_regularizing_fn(stimulus)
        loss = regularizer_loss - objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
