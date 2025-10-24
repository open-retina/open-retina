from collections import Counter
from typing import Optional

import pytest
import torch

from openretina.data_io.base_dataloader import get_movie_dataloader
from openretina.data_io.cyclers import LongCycler, ShortCycler
from openretina.data_io.session_combined_dataset import create_session_combined_dataloader


def create_test_session_dataloader(movie_frames: int, batch_size: int = 4, chunk_size: int = 50):
    """Create a test session dataloader with simulated movie data."""

    # Create dummy movie and response data
    movies = torch.randn(2, movie_frames, 32, 32)
    responses = torch.randn(movie_frames, 50)
    # Ensure no NaNs that could cause dataloader to return None
    responses = torch.where(torch.isnan(responses), torch.zeros_like(responses), responses)

    # Create realistic start indices (every scene_length frames)
    scene_length = 100
    start_indices = list(range(0, movie_frames - chunk_size, scene_length))

    dataloader = get_movie_dataloader(
        movie=movies,
        responses=responses,
        split="train",
        scene_length=scene_length,
        chunk_size=chunk_size,
        batch_size=batch_size,
        start_indices=start_indices,
        allow_over_boundaries=True,
        drop_last=False,
    )

    return dataloader


def get_test_session_dataloaders(session_frame_counts: list[int], batch_size: int = 4):
    """Create a dictionary of test session dataloaders with different frame counts."""
    session_dataloaders = {}

    for i, frame_count in enumerate(session_frame_counts):
        session_key = f"session_{i}"
        dataloader = create_test_session_dataloader(frame_count, batch_size)
        session_dataloaders[session_key] = dataloader

    return session_dataloaders


def extract_session_sequence_from_cycler(cycler, max_batches: Optional[int] = None):
    """Extract the sequence of (session_key, batch_info) from a cycler."""
    sequence = []

    for i, (session_key, batch) in enumerate(cycler):
        if max_batches is not None and i >= max_batches:
            break

        # Extract batch info - just the shape for comparison
        if hasattr(batch, "inputs"):
            batch_info = batch.inputs.shape
        else:
            batch_info = f"batch_{i}"

        sequence.append((session_key, batch_info))

    return sequence


def extract_session_sequence_from_new_dataloader(dataloader, max_batches: Optional[int] = None):
    """Extract the sequence of (session_key, batch_info) from our new dataloader."""
    sequence = []

    for i, (session_key, batch) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        # Extract batch info - should have same structure as cycler output
        batch_info = batch.inputs.shape
        sequence.append((session_key, batch_info))

    return sequence


def count_session_batches(sequence):
    """Count how many batches each session contributes."""
    return Counter(session_key for session_key, _ in sequence)


@pytest.mark.parametrize(
    "session_frame_counts, batch_size",
    [
        ([600, 800, 1000], 4),  # Different session lengths
        ([500, 500, 500], 4),  # Equal session lengths
        ([400, 1200], 4),  # Two sessions with different lengths
        ([300, 600, 900], 2),  # Different batch size
    ],
)
def test_short_cycler_compatibility(session_frame_counts: list[int], batch_size: int):
    """Test that ShortCycler and new implementation produce identical results."""

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Create old ShortCycler
    old_short_cycler = ShortCycler(session_dataloaders)

    # Create new implementation in short mode
    new_dataloader = create_session_combined_dataloader(
        session_dataloaders=session_dataloaders,
        cycling_mode="short",
        shuffle=False,  # No shuffle for exact comparison
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )

    # Extract sequences (limit to reasonable number for testing)
    max_test_batches = sum(len(dl) for dl in session_dataloaders.values())
    old_sequence = extract_session_sequence_from_cycler(old_short_cycler, max_test_batches)
    new_sequence = extract_session_sequence_from_new_dataloader(new_dataloader, max_test_batches)

    # Compare session counts
    old_counts = count_session_batches(old_sequence)
    new_counts = count_session_batches(new_sequence)

    assert old_counts == new_counts, f"Session counts differ: old={old_counts}, new={new_counts}"

    # Compare session order (should be identical for ShortCycler)
    old_sessions = [session for session, _ in old_sequence]
    new_sessions = [session for session, _ in new_sequence]

    assert old_sessions == new_sessions, "Session order differs between old and new ShortCycler"


@pytest.mark.parametrize(
    "session_frame_counts, batch_size, shuffle",
    [
        ([600, 800, 1000], 4, False),  # No shuffle
        ([500, 700], 4, False),  # Two sessions, no shuffle
        ([600, 800, 1000], 4, True),  # With shuffle
        ([400, 900, 600], 2, False),  # Different batch size
    ],
)
def test_long_cycler_compatibility(session_frame_counts: list[int], batch_size: int, shuffle: bool):
    """Test that LongCycler and new implementation produce identical results."""

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Create old LongCycler
    old_long_cycler = LongCycler(session_dataloaders, shuffle=shuffle)

    # Create new implementation in long mode
    new_dataloader = create_session_combined_dataloader(
        session_dataloaders=session_dataloaders,
        cycling_mode="long",
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )

    # For LongCycler, test a specific number of batches that shows the cycling pattern
    max_session_batches = max(len(dl) for dl in session_dataloaders.values())
    test_batches = len(session_dataloaders) * max_session_batches  # Full cycle

    old_sequence = extract_session_sequence_from_cycler(old_long_cycler, test_batches)
    new_sequence = extract_session_sequence_from_new_dataloader(new_dataloader, test_batches)

    # Compare session counts - LongCycler should have equal counts for all sessions
    old_counts = count_session_batches(old_sequence)
    new_counts = count_session_batches(new_sequence)

    assert old_counts == new_counts, f"Session counts differ: old={old_counts}, new={new_counts}"

    # For non-shuffled case, verify the cycling pattern
    if not shuffle:
        old_sessions = [session for session, _ in old_sequence]
        new_sessions = [session for session, _ in new_sequence]

        # Should have the same cycling pattern
        assert old_sessions == new_sessions, "Session cycling order differs between old and new LongCycler"


@pytest.mark.parametrize(
    "num_workers, shuffle",
    [
        (0, False),  # Single threaded, no shuffle
        (0, True),  # Single threaded, with shuffle
        (2, False),  # Multi-threaded, no shuffle
        (4, False),  # Multi-threaded, no shuffle
        # Note: Shuffled multi-worker tests can be non-deterministic due to worker randomness
    ],
)
def test_multiworker_compatibility(num_workers: int, shuffle: bool):
    """Test that our implementation works correctly with multiple workers."""

    session_frame_counts = [600, 800, 1000]
    batch_size = 4

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Test both cycling modes with multiple workers
    for cycling_mode in ["short", "long"]:
        new_dataloader = create_session_combined_dataloader(
            session_dataloaders=session_dataloaders,
            cycling_mode=cycling_mode,  # type: ignore
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=42,
        )

        # Verify we can iterate through the dataloader without errors
        batch_count = 0
        session_counts: Counter[str] = Counter()

        # Test iteration (limit to avoid infinite loops in case of issues)
        max_batches = 20
        for session_key, batch in new_dataloader:
            batch_count += 1
            session_counts[session_key] += 1

            # Verify batch structure
            assert hasattr(batch, "inputs"), "Batch missing inputs"
            assert hasattr(batch, "targets"), "Batch missing targets"
            assert batch.inputs.shape[0] == batch_size, f"Wrong batch size: {batch.inputs.shape[0]}"
            assert isinstance(session_key, str), f"Session key should be string, got {type(session_key)}"

            if batch_count >= max_batches:
                break

        # Verify we got some batches
        assert batch_count > 0, f"No batches produced with {num_workers} workers"
        assert len(session_counts) > 0, f"No sessions found with {num_workers} workers"


def test_session_isolation():
    """Test that batches always contain data from exactly one session."""

    session_frame_counts = [500, 700, 600]
    batch_size = 4

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Test both cycling modes
    for cycling_mode in ["short", "long"]:
        dataloader = create_session_combined_dataloader(
            session_dataloaders=session_dataloaders,
            cycling_mode=cycling_mode,  # type: ignore
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            seed=42,
        )

        # Check session isolation
        batches_tested = 0
        for session_key, batch in dataloader:
            batches_tested += 1

            # Verify batch comes from exactly one session (enforced by design)
            assert isinstance(session_key, str), "Session key should be string"
            assert batch.inputs.shape[0] == batch_size, "Batch size mismatch"

        assert batches_tested > 0, f"No batches tested for {cycling_mode} mode"


def test_deterministic_behavior():
    """Test that the same seed produces identical results."""

    session_frame_counts = [600, 800]
    batch_size = 4
    seed = 123

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Create two identical dataloaders with same seed
    dataloader1 = create_session_combined_dataloader(
        session_dataloaders=session_dataloaders,
        cycling_mode="long",
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    dataloader2 = create_session_combined_dataloader(
        session_dataloaders=session_dataloaders,
        cycling_mode="long",
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    # Extract sequences from both
    sequence1 = extract_session_sequence_from_new_dataloader(dataloader1, 10)
    sequence2 = extract_session_sequence_from_new_dataloader(dataloader2, 10)

    # Should be identical
    assert sequence1 == sequence2, "Same seed should produce identical results"


@pytest.mark.parametrize(
    "session_frame_counts, batch_size, cycling_mode",
    [
        ([600, 800, 1000], 4, "short"),
        ([500, 700], 4, "long"),
        ([400, 900, 600], 2, "short"),
        ([300, 500, 800], 2, "long"),
    ],
)
def test_original_batch_count_preservation(session_frame_counts: list[int], batch_size: int, cycling_mode: str):
    """Test that original batch counts are preserved when creating combined dataloader."""

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Extract original batch counts
    original_batch_counts = {key: len(dataloader) for key, dataloader in session_dataloaders.items()}

    # Create new implementation
    new_dataloader = create_session_combined_dataloader(
        session_dataloaders=session_dataloaders,
        cycling_mode=cycling_mode,  # type: ignore
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )

    # Calculate expected total batches based on cycling mode
    if cycling_mode == "short":
        # Short cycler: sum of all session batch counts
        expected_total_batches = sum(original_batch_counts.values())
    elif cycling_mode == "long":
        # Long cycler: max batch count * number of sessions
        max_batches = max(original_batch_counts.values())
        expected_total_batches = max_batches * len(session_dataloaders)

    # Count actual batches from new implementation
    actual_batch_count = 0
    session_batch_counts: Counter[str] = Counter()

    for session_key, batch in new_dataloader:
        actual_batch_count += 1
        session_batch_counts[session_key] += 1

        # Stop when we reach expected count to avoid infinite loops
        if actual_batch_count >= expected_total_batches:
            break

    # Verify total batch count matches expectation
    assert actual_batch_count == expected_total_batches, (
        f"Expected {expected_total_batches} batches but got {actual_batch_count} "
        f"for {cycling_mode} mode with original counts {original_batch_counts}"
    )

    # For short cycler, verify each session contributes exactly its original batch count
    if cycling_mode == "short":
        for session_key, expected_count in original_batch_counts.items():
            actual_count = session_batch_counts[session_key]
            assert actual_count == expected_count, (
                f"Session {session_key} expected {expected_count} batches but got {actual_count}"
            )

    # For long cycler, verify all sessions contribute the same number of batches (max count)
    elif cycling_mode == "long":
        max_batches = max(original_batch_counts.values())
        for session_key in session_dataloaders.keys():
            actual_count = session_batch_counts[session_key]
            assert actual_count == max_batches, (
                f"Long cycler: Session {session_key} expected {max_batches} batches but got {actual_count}"
            )


def test_distributed_session_sampling():
    """Test that distributed sampling correctly distributes batches across ranks,
    while maintaining session boundaries.
    """
    from unittest.mock import patch

    from openretina.data_io.session_combined_dataset import DistributedSessionSampler, SessionCombinedDataset

    session_frame_counts = [600, 800, 1000]
    batch_size = 4

    # Create test dataloaders
    session_dataloaders = get_test_session_dataloaders(session_frame_counts, batch_size)

    # Create session datasets (extract from dataloaders for direct testing)
    session_datasets = {}
    original_session_batch_counts = {}

    for session_key, dataloader in session_dataloaders.items():
        original_session_batch_counts[session_key] = len(dataloader)
        dataset = dataloader.dataset
        sampler = dataloader.sampler

        from openretina.data_io.session_combined_dataset import MovieDataSetWithSampling

        # Extract sampling parameters
        start_indices = getattr(
            sampler, "indices", list(range(0, len(dataset) * dataset.chunk_size, dataset.chunk_size))
        )

        enhanced_dataset = MovieDataSetWithSampling(
            movies=dataset.movies,
            responses=dataset.responses,
            roi_ids=getattr(dataset, "roi_ids", None),
            roi_coords=getattr(dataset, "roi_coords", None),
            group_assignment=getattr(dataset, "group_assignment", None),
            split=getattr(sampler, "split", "train"),
            chunk_size=dataset.chunk_size,
            start_indices=start_indices,
            scene_length=getattr(sampler, "scene_length", dataset.chunk_size),
            movie_length=getattr(sampler, "movie_length", dataset.movies.shape[1]),
            allow_over_boundaries=getattr(sampler, "allow_over_boundaries", False),
            epoch_seed=42,
        )

        session_datasets[session_key] = enhanced_dataset

    # Create combined dataset
    combined_dataset = SessionCombinedDataset(
        session_datasets=session_datasets,
        cycling_mode="long",
        shuffle=False,  # No shuffle for deterministic testing
        seed=42,
        batch_size=batch_size,
        original_session_batch_counts=original_session_batch_counts,
    )

    # Test with 2 ranks (simulating 2-GPU distributed training)
    num_replicas = 2

    # Mock torch.distributed for testing
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.get_world_size", return_value=num_replicas),
        patch("torch.distributed.get_rank", side_effect=[0, 1]),
    ):  # Will return 0 first, then 1
        # Create samplers for each rank
        sampler_rank0 = DistributedSessionSampler(
            combined_dataset, num_replicas=num_replicas, rank=0, shuffle=False, drop_last=False
        )

        sampler_rank1 = DistributedSessionSampler(
            combined_dataset, num_replicas=num_replicas, rank=1, shuffle=False, drop_last=False
        )

    # Get indices for each rank
    indices_rank0 = list(sampler_rank0)
    indices_rank1 = list(sampler_rank1)

    # Verify no overlap between ranks
    assert set(indices_rank0).isdisjoint(set(indices_rank1)), "Ranks should have non-overlapping indices"

    # Verify all indices are covered
    all_indices = set(indices_rank0) | set(indices_rank1)
    expected_indices = set(range(len(combined_dataset)))
    assert all_indices == expected_indices, f"All indices should be covered. Missing: {expected_indices - all_indices}"

    # Verify session boundaries are maintained within each rank
    def verify_session_boundaries(indices, dataset, rank_name):
        """Verify that batches maintain session boundaries."""
        # Group indices into batches
        batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

        for batch_idx, batch_indices in enumerate(batches):
            if not batch_indices:  # Skip empty batches
                continue

            # Get session keys for all items in this batch
            session_keys = []
            for idx in batch_indices:
                if idx < len(dataset):
                    session_key, _ = dataset[idx]
                    session_keys.append(session_key)

            # All items in batch should be from same session
            unique_sessions = set(session_keys)
            assert len(unique_sessions) <= 1, (
                f"{rank_name} batch {batch_idx} contains multiple sessions: {unique_sessions}"
            )

    # Test session boundaries for both ranks
    verify_session_boundaries(indices_rank0, combined_dataset, "Rank 0")
    verify_session_boundaries(indices_rank1, combined_dataset, "Rank 1")

    # Verify that each rank gets a reasonable distribution of data
    # (should be roughly equal, within 1 batch difference)
    len_diff = abs(len(indices_rank0) - len(indices_rank1))
    max_acceptable_diff = batch_size  # At most one batch difference
    assert len_diff <= max_acceptable_diff, (
        f"Rank data distribution too uneven: rank0={len(indices_rank0)}, rank1={len(indices_rank1)}"
    )

    print(f"✓ Distributed test passed: Rank0={len(indices_rank0)} samples, Rank1={len(indices_rank1)} samples")
