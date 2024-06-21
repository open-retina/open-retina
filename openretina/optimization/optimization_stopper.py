from typing import Optional
import sys


class OptimizationStopper:
    def __init__(self, max_epochs: Optional[int]):
        if max_epochs is None:
            self.max_epochs = sys.maxsize
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
