"""
Adapted from sinzlab/neuralpredictors/training/cyclers.py
"""

import random

import torch.utils.data
from torch.utils.data import DataLoader


def cycle(iterable):
    """
    itertools.cycle without caching.
    See: https://github.com/pytorch/pytorch/issues/23900
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class LongCycler(torch.utils.data.IterableDataset):
    """
    Cycles through a dictionary of data loaders until the loader with the largest size is exhausted.
    In practice, takes one batch from each loader in each iteration.
    Necessary for dataloaders of unequal size.
    """

    def __init__(self, loaders: dict[str, DataLoader], shuffle: bool = True):
        self.loaders = loaders
        self.max_batches = max(len(loader) for loader in self.loaders.values())
        self.shuffle = shuffle

    def __iter__(self):
        keys = list(self.loaders.keys())
        if self.shuffle:
            random.shuffle(keys)

        cycles = [cycle(self.loaders[k]) for k in keys]
        for k, loader, _ in zip(
            cycle(keys),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)
