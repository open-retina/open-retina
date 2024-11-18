"""
Adapted from sinzlab/neuralpredictors/training/cyclers.py
"""

import random
from itertools import islice

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
        worker_info = torch.utils.data.get_worker_info()
        keys = list(self.loaders.keys())

        if self.shuffle:
            random.shuffle(keys)

        # Create cycles for each loader
        cycles = [cycle(self.loaders[k]) for k in keys]

        if worker_info is None:  # Single-process data loading
            iter_start = 0
            iter_end = len(self.loaders) * self.max_batches
            total_iterations = iter_end
        else:
            # Partition the iterations among the workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            total_iterations = len(self.loaders) * self.max_batches
            per_worker = (total_iterations + num_workers - 1) // num_workers
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, total_iterations)

        # Yield batches in the assigned range
        for k, loader, _ in islice(zip(cycle(keys), cycle(cycles), range(total_iterations)), iter_start, iter_end):
            yield k, next(loader)


    def __len__(self):
        return len(self.loaders) * self.max_batches
