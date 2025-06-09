"""
Improved MAP-style dataset that combines multiple session datasets with proper support for
multiprocessing, distributed training, and different cycling strategies.
"""

import random
from typing import Any, Literal, Optional, cast

import lightning
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from openretina.data_io.base_dataloader import DataPoint, gen_shifts_with_boundaries


def session_collate_fn(batch):
    """
    Custom collate function that ensures all items in a batch come from the same session.
    Each batch element is (session_key, DataPoint).
    Returns (session_key, stacked_data_point).
    """
    if not batch:
        return None

    # Separate session keys and data points
    session_keys = []
    data_points = []

    for session_key, data_point in batch:
        session_keys.append(session_key)
        data_points.append(data_point)

    # Verify all items in the batch come from the same session
    unique_sessions = set(session_keys)
    if len(unique_sessions) > 1:
        raise ValueError(
            f"Batch contains data from multiple sessions: {unique_sessions}. "
            "This should not happen with proper batch alignment."
        )

    # Get the single session for this batch
    session_key = session_keys[0]

    # Stack the data points from the batch
    # Each data_point is a DataPoint with .inputs and .targets
    inputs = torch.stack([dp.inputs for dp in data_points])
    targets = torch.stack([dp.targets for dp in data_points])

    stacked_data_point = DataPoint(inputs, targets)

    return session_key, stacked_data_point


class MovieDataSetWithSampling(Dataset):
    """
    Enhanced MovieDataSet that incorporates sampling logic directly in __getitem__
    to avoid the need for a separate sampler.
    """

    def __init__(
        self,
        movies: torch.Tensor | np.ndarray,
        responses: torch.Tensor | np.ndarray,
        roi_ids: Optional[np.ndarray],
        roi_coords: Optional[np.ndarray],
        group_assignment: Optional[np.ndarray],
        split: str,
        chunk_size: int,
        start_indices: list[int],
        scene_length: int,
        movie_length: int,
        allow_over_boundaries: bool = False,
        epoch_seed: Optional[int] = None,
    ):
        # Store data
        self.samples = (movies, responses)
        self.roi_ids = roi_ids
        self.roi_coords = roi_coords
        self.group_assignment = group_assignment
        self.split = split
        self.chunk_size = chunk_size
        self.start_indices = start_indices
        self.scene_length = scene_length
        self.movie_length = movie_length
        self.allow_over_boundaries = allow_over_boundaries

        # Calculate mean response for bias initialization
        self.mean_response = torch.mean(torch.Tensor(responses), dim=0)

        # For deterministic sampling across epochs
        self.epoch_seed = epoch_seed
        self._current_epoch = 0

        # Pre-compute sampling indices for this epoch
        self._update_sampling_indices()

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic sampling."""
        self._current_epoch = epoch
        self._update_sampling_indices()

    def _update_sampling_indices(self):
        """Update the sampling indices based on the current epoch."""
        # Set random seed for deterministic behavior
        if self.epoch_seed is not None:
            np.random.seed(self.epoch_seed + self._current_epoch)

        if self.split == "train" and (self.scene_length != self.chunk_size):
            if self.allow_over_boundaries:
                shifts = np.random.randint(0, self.chunk_size, len(self.start_indices))
                shifted_indices = np.minimum(np.array(self.start_indices) + shifts, self.movie_length - self.chunk_size)
            else:
                shifted_indices = gen_shifts_with_boundaries(
                    np.arange(0, self.movie_length + 1, self.scene_length),
                    self.start_indices,
                    self.chunk_size,
                )
            # Shuffle the indices
            indices_shuffling = np.random.permutation(len(self.start_indices))
            self.actual_indices = np.array(shifted_indices)[indices_shuffling]
        else:
            self.actual_indices = np.array(self.start_indices)

    def __getitem__(self, idx: int) -> DataPoint:
        """Get item using the pre-computed sampling indices."""
        actual_idx = self.actual_indices[idx]

        return DataPoint(
            self.samples[0][:, actual_idx : actual_idx + self.chunk_size, ...],
            self.samples[1][actual_idx : actual_idx + self.chunk_size, ...],
        )

    def __len__(self) -> int:
        return len(self.start_indices)

    @property
    def movies(self):
        return self.samples[0]

    @property
    def responses(self):
        return self.samples[1]


class SessionCombinedDataset(Dataset):
    """
    Combined dataset that delegates to individual session datasets in a MAP-style manner.
    """

    def __init__(
        self,
        session_datasets: dict[str, MovieDataSetWithSampling],
        cycling_mode: Literal["long", "short"] = "short",
        shuffle: bool = True,
        seed: int = 42,
        batch_size: int = 1,
        original_session_batch_counts: Optional[dict[str, int]] = None,
    ):
        self.session_datasets = session_datasets
        self.session_keys = sorted(session_datasets.keys())
        self.cycling_mode = cycling_mode
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.original_session_batch_counts = original_session_batch_counts

        # Get session lengths
        self.session_lengths = {key: len(dataset) for key, dataset in session_datasets.items()}
        self.max_session_length = max(self.session_lengths.values())
        self.total_session_length = sum(self.session_lengths.values())

        # Build the global index mapping
        self._build_index_mapping()

    def _build_index_mapping(self):
        """
        Build mapping from global index to (session_key, local_index) ensuring
        that consecutive indices (which form batches) stay within sessions.

        The key insight: the original cyclers work at the batch level, not item level.
        LongCycler takes one batch from each session's dataloader in turn.
        """
        self.index_mapping = []

        if self.cycling_mode == "long":
            # Mimic LongCycler: cycle through sessions, taking one batch from each
            session_order = self.session_keys.copy()
            if self.shuffle:
                rng = random.Random(self.seed)
                rng.shuffle(session_order)

            # Use original session batch counts if provided, otherwise calculate
            if self.original_session_batch_counts is not None:
                session_batch_counts = self.original_session_batch_counts
            else:
                # Fallback: calculate based on dataset length and batch size
                session_batch_counts = {}
                for session_key in session_order:
                    dataset_length = len(self.session_datasets[session_key])
                    session_batch_counts[session_key] = (dataset_length + self.batch_size - 1) // self.batch_size

            max_batches = max(session_batch_counts.values())

            # For each "round" across sessions - ALL sessions contribute max_batches
            for batch_round in range(max_batches):
                for session_key in session_order:
                    # ALL sessions contribute in every round (LongCycler behavior)
                    # Shorter sessions cycle through their available batches
                    session_batch_count = session_batch_counts[session_key]
                    actual_batch_idx = batch_round % session_batch_count

                    # Add batch_size consecutive indices for this session's batch
                    for item_in_batch in range(self.batch_size):
                        # Calculate the actual dataset index for this item
                        dataset_idx = actual_batch_idx * self.batch_size + item_in_batch
                        dataset_length = len(self.session_datasets[session_key])
                        # Wrap around if needed (for cycling behavior)
                        if dataset_idx >= dataset_length:
                            dataset_idx = dataset_idx % dataset_length
                        self.index_mapping.append((session_key, dataset_idx))

        elif self.cycling_mode == "short":
            # Mimic ShortCycler: exhaust each session completely before moving to next
            session_order = self.session_keys.copy()
            if self.shuffle:
                rng = random.Random(self.seed)
                rng.shuffle(session_order)

            for session_key in session_order:
                # Get dataset length for this session
                dataset_length = len(self.session_datasets[session_key])

                # Use original session batch count if provided, otherwise calculate
                if self.original_session_batch_counts is not None:
                    num_batches = self.original_session_batch_counts[session_key]
                else:
                    # Fallback: calculate based on dataset length and batch size
                    num_batches = (dataset_length + self.batch_size - 1) // self.batch_size

                # Add all batches from this session before moving to the next
                for batch_idx in range(num_batches):
                    # Add batch_size consecutive indices for this batch
                    for item_in_batch in range(self.batch_size):
                        # Calculate the actual dataset index for this item in the batch
                        dataset_idx = batch_idx * self.batch_size + item_in_batch
                        # If we go beyond the dataset, wrap around (for incomplete last batch)
                        if dataset_idx >= dataset_length:
                            dataset_idx = dataset_idx % dataset_length
                        self.index_mapping.append((session_key, dataset_idx))

    def set_epoch(self, epoch: int):
        """Set epoch for all session datasets for deterministic sampling."""
        for dataset in self.session_datasets.values():
            dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, global_idx: int) -> tuple[str, Any]:
        """Get item by global index using session datasets."""
        if global_idx >= len(self.index_mapping):
            raise IndexError(f"Index {global_idx} out of range for dataset of size {len(self)}")

        session_key, local_idx = self.index_mapping[global_idx]
        data = self.session_datasets[session_key][local_idx]

        # Ensure session_key is a string, not a tuple
        if isinstance(session_key, (list, tuple)):
            session_key = session_key[0] if len(session_key) > 0 else str(session_key)

        return session_key, data

    def get_session_info(self) -> dict[str, Any]:
        """Get information about the sessions in this dataset. Used in debugging."""
        return {
            "session_keys": self.session_keys,
            "session_lengths": self.session_lengths,
            "cycling_mode": self.cycling_mode,
            "total_length": len(self),
            "max_session_length": self.max_session_length,
            "total_session_length": self.total_session_length,
        }


class DistributedSessionSampler(DistributedSampler):
    """
    Distributed sampler that ensures no data duplication across processes
    while maintaining session structure by distributing complete batches.
    """

    def __init__(self, dataset: SessionCombinedDataset, **kwargs):
        # Don't call super().__init__ because we need custom logic
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = kwargs.get("drop_last", False)
        self.shuffle = kwargs.get("shuffle", True)

        num_replicas = kwargs.get("num_replicas", None)
        rank = kwargs.get("rank", None)

        if num_replicas is None:
            import torch.distributed as dist

            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            import torch.distributed as dist

            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        # Set base class attributes directly
        self.num_replicas = num_replicas
        self.rank = rank

        # Calculate batch boundaries
        self.batch_size = dataset.batch_size
        total_samples = len(dataset)

        # Calculate number of complete batches
        self.total_batches = total_samples // self.batch_size
        if total_samples % self.batch_size != 0 and not self.drop_last:
            self.total_batches += 1

        # Distribute batches across processes - ensure each rank gets different batches
        self.batches_per_replica = self.total_batches // self.num_replicas
        remainder_batches = self.total_batches % self.num_replicas

        # Each rank gets consecutive batches, not interleaved
        if self.rank < remainder_batches:
            # Ranks 0 to remainder_batches-1 get one extra batch
            self.batches_per_replica += 1
            start_batch = self.rank * self.batches_per_replica
        else:
            # Remaining ranks get standard number of batches
            start_batch = (
                remainder_batches * (self.batches_per_replica + 1)
                + (self.rank - remainder_batches) * self.batches_per_replica
            )

        end_batch = start_batch + self.batches_per_replica

        # Calculate sample indices for this rank
        self.start_idx = start_batch * self.batch_size
        self.end_idx = min(end_batch * self.batch_size, total_samples)
        self.num_samples = self.end_idx - self.start_idx

    def __iter__(self):
        # Set epoch for session datasets to ensure deterministic sampling
        self.dataset.set_epoch(self.epoch)

        # Return indices for this rank's portion of the data
        indices = list(range(self.start_idx, self.end_idx))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def create_session_combined_dataloader(
    session_dataloaders: dict[str, torch.utils.data.DataLoader],
    cycling_mode: Literal["long", "short"] = "short",
    shuffle: bool = True,
    batch_size: int = 1,
    num_workers: int = 0,
    distributed: bool = False,
    seed: int = 42,
    **dataloader_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create an improved combined dataloader from session dataloaders.

    This function combines multiple session dataloaders into a single MAP-style
    dataset that supports multiprocessing and distributed training.

    It is functionally equivalent to the LongCycler and ShortCycler classes, but is more flexible and easier to use.

    Args:
        session_dataloaders: Dict of session dataloaders
        cycling_mode: How to cycle through sessions ("long" or "short")
        shuffle: Whether to shuffle session order
        batch_size: Batch size for the combined dataloader
        num_workers: Number of worker processes
        distributed: Whether to use distributed sampling. Should be set to False when using with PyTorch Lightning.
        seed: Random seed for reproducibility
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader with the combined dataset
    """
    # Extract datasets from dataloaders and convert to MovieDataSetWithSampling
    session_datasets = {}
    original_session_batch_counts = {}

    for session_key, dataloader in session_dataloaders.items():
        # Store the original batch count from the dataloader
        original_session_batch_counts[session_key] = len(dataloader)
        dataset = dataloader.dataset
        dataset_sampler = dataloader.sampler

        # Extract the sampling parameters from the original sampler
        if hasattr(dataset_sampler, "indices"):
            start_indices = dataset_sampler.indices  # type: ignore
        else:
            # Fallback: generate indices based on dataset length and chunk size
            if hasattr(dataset, "chunk_size"):
                start_indices = list(range(0, len(dataset) * dataset.chunk_size, dataset.chunk_size))  # type: ignore
            else:
                start_indices = list(range(0, len(dataset)))  # type: ignore

        # Create new dataset with sampling logic
        enhanced_dataset = MovieDataSetWithSampling(
            movies=dataset.movies if hasattr(dataset, "movies") else cast(torch.Tensor, dataset[0][0]),  # type: ignore
            responses=dataset.responses if hasattr(dataset, "responses") else cast(torch.Tensor, dataset[0][1]),  # type: ignore
            roi_ids=dataset.roi_ids if hasattr(dataset, "roi_ids") else None,  # type: ignore
            roi_coords=dataset.roi_coords if hasattr(dataset, "roi_coords") else None,  # type: ignore
            group_assignment=dataset.group_assignment if hasattr(dataset, "group_assignment") else None,  # type: ignore
            split=getattr(dataset_sampler, "split", "train"),
            chunk_size=dataset.chunk_size if hasattr(dataset, "chunk_size") else 50,  # type: ignore
            start_indices=start_indices,
            scene_length=getattr(
                dataset_sampler, "scene_length", dataset.chunk_size if hasattr(dataset, "chunk_size") else 50
            ),  # type: ignore
            movie_length=getattr(
                dataset_sampler, "movie_length", dataset.movies.shape[1] if hasattr(dataset, "movies") else 1000
            ),  # type: ignore
            allow_over_boundaries=getattr(dataset_sampler, "allow_over_boundaries", False),
            epoch_seed=seed,
        )

        session_datasets[session_key] = enhanced_dataset

    # Create combined dataset
    combined_dataset = SessionCombinedDataset(
        session_datasets=session_datasets,
        cycling_mode=cycling_mode,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
        original_session_batch_counts=original_session_batch_counts,
    )

    # Choose appropriate sampler
    session_sampler: Any = None
    shuffle_dataloader: bool = False
    if distributed:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                session_sampler = DistributedSessionSampler(
                    combined_dataset, shuffle=False
                )  # Shuffling handled in dataset
                shuffle_dataloader = False
            else:
                print(
                    "Warning: Distributed training requested but torch.distributed not initialized. "
                    "Using standard sampling."
                )
                session_sampler = None
                shuffle_dataloader = False
        except Exception as e:
            print(f"Warning: Failed to create distributed sampler: {e}. Using standard sampling.")
            session_sampler = None
            shuffle_dataloader = False
    else:
        session_sampler = None
        shuffle_dataloader = False  # Shuffling handled in dataset

    # Create combined dataloader
    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_dataloader,
        sampler=session_sampler,
        collate_fn=session_collate_fn,
        **dataloader_kwargs,
    )


class SessionAwareDataModule(lightning.LightningDataModule):
    """
    Custom DataModule that handles session-aware distributed training.
    """

    def __init__(self, dataloaders_dict, batch_size, seed=42):
        super().__init__()
        self.dataloaders_dict = dataloaders_dict
        self.batch_size = batch_size
        self.seed = seed
        self.train_loader = None
        self.val_loader = None
        self.test_dataloaders_dict = None

    def setup(self, stage=None):
        # Check if we're in a distributed environment
        is_distributed = dist.is_available() and dist.is_initialized()

        if stage == "fit" or stage is None:
            self.train_loader = create_session_combined_dataloader(
                session_dataloaders=self.dataloaders_dict["train"],
                cycling_mode="long",
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                seed=self.seed,
                distributed=is_distributed,
            )

            self.val_loader = create_session_combined_dataloader(
                session_dataloaders=self.dataloaders_dict["validation"],
                cycling_mode="short",
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                seed=self.seed,
                distributed=is_distributed,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        """Return test dataloaders for all splits (train, validation, test)."""
        if self.test_dataloaders_dict is None:
            return []

        # Check if we're in a distributed environment
        is_distributed = dist.is_available() and dist.is_initialized()

        test_loaders = []
        for split_name, session_dataloaders in self.test_dataloaders_dict.items():
            test_loader = create_session_combined_dataloader(
                session_dataloaders=session_dataloaders,
                cycling_mode="short",  # Use short cycling for testing (exhaust all data)
                shuffle=False,  # No shuffling for testing
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                seed=self.seed,
                distributed=is_distributed,
            )
            test_loaders.append(test_loader)

        return test_loaders
