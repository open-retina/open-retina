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
    """
    Optimize a stimulus to maximize a given objective while minimizing a regularizing function.
    The stimulus is modified in place.
    """
    optimizer = optimizer_init_fn(stimulus)

    # Could add early stopping interface,
    # e.g. from [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
    for i in range(max_iterations):
        objective = objective.forward(stimulus)
        regularizer_loss = stimulus_regularizing_fn(stimulus)
        # Maximizing the objective, minimizing the regularization loss
        loss = -objective + regularizer_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    stimulus.detach_()  # Detach the tensor from the computation graph
