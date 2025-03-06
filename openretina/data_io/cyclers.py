"""
Adapted from sinzlab/neuralpredictors/training/cyclers.py
"""

import math
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
    Note: iterable dataloaders as this one can lead to duplicate data when using multiprocessing.
    """

    def __init__(self, loaders: dict[str, DataLoader], shuffle: bool = True):
        self.loaders = loaders
        self.max_batches = max(len(loader) for loader in self.loaders.values())
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("LongCycler does not support multiple workers yet.")

        keys = sorted(self.loaders.keys())

        if self.shuffle:
            random.shuffle(keys)

        # Create cycles for each loader
        cycles = [cycle(self.loaders[k]) for k in keys]
        total_iterations = len(self.loaders) * self.max_batches

        # Yield batches in the assigned range
        for k, loader, _ in zip(cycle(keys), cycle(cycles), range(total_iterations)):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler(torch.utils.data.IterableDataset):
    """
    Cycles through the elements of each dataloader without repeating any element.
    """

    def __init__(self, loaders: dict[str, DataLoader]):
        self.loaders = loaders

    def _get_keys(self) -> list[str]:
        sorted_keys = sorted(self.loaders.keys())
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            if worker_info.num_workers > len(sorted_keys):
                raise ValueError(f"Too many workers for {len(sorted_keys)} sessions: {worker_info=}")

            sess_per_worker = math.ceil(len(sorted_keys) / worker_info.num_workers)
            start_idx = sess_per_worker * worker_info.id
            return sorted_keys[start_idx : start_idx + sess_per_worker]
        else:
            return sorted_keys

    def __iter__(self):
        for k in self._get_keys():
            for example in self.loaders[k]:
                yield k, example
