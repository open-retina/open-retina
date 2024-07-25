"""
Parts copied from sinzlab/neuralpredictors/training/cyclers.py
"""

import random

import torch.utils.data


def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class LongCycler(torch.utils.data.IterableDataset):
    """
    Cycles through trainloaders until the loader with the largest size is exhausted.
    Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders, shuffle: bool = True):
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

    def __len__(self):
        return len(self.loaders) * self.max_batches
