"""Utilities for computing unique frame and transition statistics from dataloaders.

Frame fingerprints are lightweight summaries of individual video frames used to count
unique frames and consecutive-frame transitions without storing full frame data.
"""

import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from openretina.data_io.base import DatasetStatistics

log = logging.getLogger(__name__)

FrameFingerprint = tuple[float, float, float, float, float]
TransitionFingerprint = tuple[FrameFingerprint, FrameFingerprint]


def _compute_frame_fingerprints(inputs: torch.Tensor) -> list[FrameFingerprint]:
    """Compute lightweight fingerprints for all frames in a batch.

    Each fingerprint is a tuple of (mean, std, first_pixel, last_pixel, mid_pixel).
    The mid_pixel is sampled at an off-center position (channel 0, row H//2, col W//3)
    to break horizontal-flip symmetry — mean, std, and corner pairs can coincide
    between a frame and its horizontal flip, but an off-center pixel will not.

    Args:
        inputs: Batch of video clips with shape (B, C, T, H, W).

    Returns:
        List of fingerprint tuples, one per frame (length B*T).
    """
    B, C, T, H, W = inputs.shape
    frames = inputs.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    flat = frames.reshape(B * T, -1)
    means = flat.mean(dim=1)
    stds = flat.std(dim=1)
    first_pixels = flat[:, 0]
    last_pixels = flat[:, -1]
    # Off-center pixel to break horizontal-flip symmetry
    mid_pixels = frames[:, 0, H // 2, W // 3]

    return [
        (float(means[i]), float(stds[i]), float(first_pixels[i]), float(last_pixels[i]), float(mid_pixels[i]))
        for i in range(B * T)
    ]


def _extract_transitions(
    fingerprints: list[FrameFingerprint], batch_size: int, time_steps: int
) -> list[TransitionFingerprint]:
    """Extract consecutive-frame transition fingerprints from a batch.

    Args:
        fingerprints: Flat list of frame fingerprints (length B*T).
        batch_size: Number of clips in the batch (B).
        time_steps: Number of time steps per clip (T).

    Returns:
        List of transition fingerprint tuples.
    """
    transitions: list[TransitionFingerprint] = []
    for b in range(batch_size):
        offset = b * time_steps
        for t in range(time_steps - 1):
            transitions.append((fingerprints[offset + t], fingerprints[offset + t + 1]))
    return transitions


def _collect_from_dataloaders(
    split_dataloaders: dict[str, DataLoader],
    n_iterations: int = 1,
) -> tuple[set[FrameFingerprint], set[TransitionFingerprint]]:
    """Iterate over all session dataloaders for a split and collect unique frames and transitions.

    Args:
        split_dataloaders: Dict mapping session name to DataLoader.
        n_iterations: Number of times to iterate over each dataloader.

    Returns:
        Tuple of (unique_frames set, unique_transitions set).
    """
    unique_frames: set[FrameFingerprint] = set()
    unique_transitions: set[TransitionFingerprint] = set()

    for _iteration in tqdm(
        range(n_iterations), desc="Computing unique frames and transitions", total=n_iterations, leave=False
    ):
        for dl in split_dataloaders.values():
            for batch in dl:
                inputs = batch.inputs  # (B, C, T, H, W)
                B, C, T, H, W = inputs.shape
                fps = _compute_frame_fingerprints(inputs)
                unique_frames.update(fps)
                unique_transitions.update(_extract_transitions(fps, B, T))

    return unique_frames, unique_transitions


def compute_dataloader_statistics(
    dataloaders: dict[str, dict[str, DataLoader]],
    n_augmentation_iterations: int = 10,
) -> DatasetStatistics:
    """Compute unique frame and transition counts by iterating over dataloaders.

    This replaces the old config-parameter-based estimation by directly counting
    what the model sees. Training dataloaders are iterated multiple times to
    capture the effective diversity introduced by data augmentation (random shifts).
    Validation and test dataloaders are deterministic, so a single pass suffices.

    Args:
        dataloaders: Nested dict with structure {split_name: {session_name: DataLoader}}.
            Expected keys include "train", "validation" (or "val"), and test split names.
        n_augmentation_iterations: Number of times to iterate over training dataloaders.

    Returns:
        DatasetStatistics with unique frame and transition counts per split.
    """
    # Identify which splits are train, val, and test
    train_key = "train" if "train" in dataloaders else None
    val_key = next((k for k in dataloaders if k in ("validation", "val")), None)
    test_keys = [k for k in dataloaders if k not in ("train", "validation", "val")]

    # Count sessions across all splits (use train if available, else union of all)
    all_sessions: set[str] = set()
    for split_dls in dataloaders.values():
        all_sessions.update(split_dls.keys())
    n_sessions = len(all_sessions)

    # Collect train frames and transitions
    train_frames: set[FrameFingerprint] = set()
    train_transitions: set[TransitionFingerprint] = set()
    if train_key is not None:
        log.info(f"Computing training statistics ({n_augmentation_iterations} iterations)...")
        train_frames, train_transitions = _collect_from_dataloaders(
            dataloaders[train_key], n_iterations=n_augmentation_iterations
        )

    # Collect val frames and transitions (single pass)
    val_frames: set[FrameFingerprint] = set()
    val_transitions: set[TransitionFingerprint] = set()
    if val_key is not None:
        log.info("Computing validation statistics...")
        val_frames, val_transitions = _collect_from_dataloaders(dataloaders[val_key], n_iterations=1)

    # Collect test frames and transitions (single pass per test split)
    unique_test_frames: dict[str, int] = {}
    unique_test_transitions: dict[str, int] = {}
    for test_key in test_keys:
        log.info(f"Computing test statistics for '{test_key}'...")
        test_f, test_t = _collect_from_dataloaders(dataloaders[test_key], n_iterations=1)
        unique_test_frames[test_key] = len(test_f)
        unique_test_transitions[test_key] = len(test_t)

    # Compute deduplicated train+val frame count
    train_val_frames = train_frames | val_frames

    return DatasetStatistics(
        unique_train_frames=len(train_frames),
        unique_val_frames=len(val_frames),
        unique_train_val_frames=len(train_val_frames),
        unique_test_frames=unique_test_frames,
        unique_train_transitions=len(train_transitions),
        unique_val_transitions=len(val_transitions),
        unique_test_transitions=unique_test_transitions,
        n_sessions=n_sessions,
    )
