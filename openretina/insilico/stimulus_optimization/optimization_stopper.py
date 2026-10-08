import sys
from typing import Optional


class OptimizationStopper:
    """Base stopping criterion for stimulus optimization. Stops after max_iterations."""

    def __init__(self, max_iterations: Optional[int]):
        if max_iterations is None:
            self.max_iterations = sys.maxsize
        else:
            self.max_iterations = max_iterations

    def early_stop(self, loss: float) -> bool:
        return False


class EarlyStopper(OptimizationStopper):
    """Stops optimization when loss stops improving for `patience` consecutive steps."""

    def __init__(self, max_iterations: Optional[int] = None, patience: int = 1, min_delta: float = 0.0):
        super().__init__(max_iterations)
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_loss = float("inf")

    def early_stop(self, loss: float) -> bool:
        if loss < self._min_loss:
            self._min_loss = loss
            self._counter = 0
        elif loss > (self._min_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True
        return False
