"""Configurable omitted-stimulus-response screening for core-readout models.

The functions in this module deliberately keep stimulus construction, model
alignment, response metrics, and classification explicit.  The command-line
entry point is ``scripts/analyze_osr.py``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import spearmanr

import openretina
from openretina.data_io.hoefling_2024.constants import (
    RGC_GROUP_GROUP_ID_TO_CLASS_NAME,
    RGC_GROUP_NAMES_DICT,
    STIMULUS_RANGE_CONSTRAINTS,
    MEAN_STD_DICT_74x64,
    pre_normalisation_values_18x16,
)
from openretina.models.core_readout import load_core_readout_model

LOGGER = logging.getLogger("openretina.osr")
SCALE_FLOOR = 1e-6


@dataclass(frozen=True)
class StimulusCondition:
    """Compact description of one spatially uniform stimulus condition."""

    condition_id: str
    condition_type: str
    period_frames: int
    n_flashes: int
    flash_frames: int
    amplitude: float
    polarity: str
    channel_vector: tuple[float, ...]
    baseline_vector: tuple[float, ...]
    flash_vector: tuple[float, ...]
    variant: int
    control_seed: int | None
    flash_onsets: tuple[int, ...]
    last_real_flash_frame: int | None
    expected_flash_frame: int
    expected_flash_present: bool
    flashes_resume: bool
    time_steps: int

    def to_record(self, fps: float, cut_frames: int) -> dict[str, Any]:
        """Return a CSV-friendly condition record."""

        intervals = np.diff(self.flash_onsets).astype(int).tolist() if len(self.flash_onsets) > 1 else []
        return {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type,
            "period_frames": self.period_frames,
            "period_hz": fps / self.period_frames,
            "n_flashes": self.n_flashes,
            "flash_frames": self.flash_frames,
            "amplitude": self.amplitude,
            "polarity": self.polarity,
            "channel_vector": json.dumps(self.channel_vector),
            "baseline_vector": json.dumps(self.baseline_vector),
            "flash_vector": json.dumps(self.flash_vector),
            "variant": self.variant,
            "control_seed": self.control_seed,
            "flash_onsets": json.dumps(self.flash_onsets),
            "inter_flash_intervals": json.dumps(intervals),
            "last_real_flash_frame": self.last_real_flash_frame,
            "expected_flash_frame": self.expected_flash_frame,
            "expected_output_frame": self.expected_flash_frame - cut_frames,
            "expected_flash_present": self.expected_flash_present,
            "flashes_resume": self.flashes_resume,
            "time_steps": self.time_steps,
            "inferred_cut_frames": cut_frames,
        }


@dataclass(frozen=True)
class GeneratedCondition:
    """A condition and its compact channels-by-time model-space trace."""

    metadata: StimulusCondition
    trace: np.ndarray


@dataclass(frozen=True)
class ValueSpaceResolution:
    """Resolved normalization strategy and model-space stimulus vectors."""

    requested_value_space: str
    strategy: str
    raw_stim_mean: Any
    raw_stim_std: Any
    normalization_mean: tuple[float, ...] | None
    normalization_std: tuple[float, ...] | None
    baseline_model: tuple[float, ...]
    documented_min: tuple[float, ...] | None
    documented_max: tuple[float, ...] | None


@dataclass(frozen=True)
class TimingResolution:
    """Input/output timing values resolved from rates, conditions, and model cut."""

    fps: float
    cut_frames: int
    expected_frame: int
    time_steps: int
    output_time_steps: int
    osr_start_offset: int
    osr_stop_offset: int
    osr_start_ms_resolved: float
    osr_stop_ms_resolved: float
    pre_event_offsets: tuple[int, ...]
    minimum_time_steps: int


@dataclass(frozen=True)
class AnalysisThresholds:
    """Engineering defaults used for auditable OSR classification."""

    min_baseline_gain: float = 0.10
    min_periodicity_gain: float = 0.10
    min_history_gain: float = 0.10
    min_spearman_rho: float = 0.50
    timing_slope_min: float = 0.50
    timing_slope_max: float = 1.50
    min_timing_r2: float = 0.50
    max_expected_latency_std: float = 1.0


@dataclass
class ResolvedRunConfig:
    """Fully resolved command-line configuration."""

    model: str
    device: str
    output_dir: Path
    batch_size: int
    num_threads: int
    seed: int
    overwrite: bool
    quick: bool
    fps: float
    time_steps: int | None
    auto_time_steps: bool
    period_frames: tuple[int, ...]
    n_flashes: tuple[int, ...]
    flash_frames: int
    post_omission_frames: int | None
    embedded_post_flashes: int
    value_space: str
    baseline_values: tuple[float, ...] | None
    flash_values: tuple[float, ...] | None
    flash_amplitudes: tuple[float, ...]
    polarity: tuple[str, ...]
    channel_mode: str
    channel_vector: tuple[float, ...] | None
    n_jitter_controls: int
    include_embedded_omission: bool
    include_sustained_controls: bool
    osr_window_ms: tuple[float, float]
    target_n_flashes: int
    thresholds: AnalysisThresholds
    top_k_plots: int
    save_raw_responses: bool


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list with duplicate removal."""

    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers, got {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return values


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated finite float list."""

    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers, got {value!r}") from exc
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("At least one finite number is required")
    return values


def parse_float_pair(value: str) -> tuple[float, float]:
    """Parse exactly two comma-separated finite floats."""

    values = parse_csv_floats(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"Expected exactly two comma-separated numbers, got {value!r}")
    return values[0], values[1]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the OSR command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Screen an OpenRetina core-readout model for terminal, periodicity-specific, "
            "and predictively timed omitted-stimulus responses."
        )
    )
    runtime = parser.add_argument_group("model and runtime")
    runtime.add_argument("--model", default="hoefling_2024_low_res", help="Registered model name, path, or URL.")
    runtime.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--batch-size", type=int, default=8)
    runtime.add_argument("--num-threads", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=12345)
    runtime.add_argument("--overwrite", action="store_true")
    runtime.add_argument("--quick", action="store_true", help="Use the small CPU-oriented default condition battery.")

    timing = parser.add_argument_group("timing")
    timing.add_argument("--fps", type=float, default=None, help="Override the model stimulus/response rate.")
    timing.add_argument("--time-steps", type=int, default=None)
    timing.add_argument("--auto-time-steps", action="store_true")
    timing.add_argument("--period-frames", type=parse_csv_ints, default=None)
    timing.add_argument("--n-flashes", type=parse_csv_ints, default=None)
    timing.add_argument("--flash-frames", type=int, default=1)
    timing.add_argument(
        "--post-omission-frames",
        type=int,
        default=None,
        help="Minimum frames retained after the expected event; otherwise derived from the OSR window.",
    )
    timing.add_argument("--embedded-post-flashes", type=int, default=3)

    stimulus = parser.add_argument_group("stimulus values")
    stimulus.add_argument("--value-space", choices=("model", "raw", "auto"), default="auto")
    stimulus.add_argument("--baseline-values", type=parse_csv_floats, default=None)
    stimulus.add_argument(
        "--flash-values",
        type=parse_csv_floats,
        default=None,
        help="Explicit flash vector in the requested value space; disables amplitude/polarity expansion.",
    )
    stimulus.add_argument("--flash-amplitudes", type=parse_csv_floats, default=None)
    stimulus.add_argument("--polarity", choices=("dark", "bright", "both"), default="dark")
    stimulus.add_argument(
        "--channel-mode",
        choices=("all", "channel0", "channel1", "custom"),
        default="all",
    )
    stimulus.add_argument("--channel-vector", type=parse_csv_floats, default=None)

    controls = parser.add_argument_group("controls")
    controls.add_argument("--n-jitter-controls", type=int, default=None)
    controls.add_argument(
        "--include-embedded-omission",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    controls.add_argument(
        "--include-sustained-controls",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    analysis = parser.add_argument_group("analysis and output")
    analysis.add_argument("--osr-window-ms", type=parse_float_pair, default=(33.0, 200.0))
    analysis.add_argument("--target-n-flashes", type=int, default=7)
    analysis.add_argument("--min-baseline-gain", type=float, default=0.10)
    analysis.add_argument("--min-periodicity-gain", type=float, default=0.10)
    analysis.add_argument("--min-history-gain", type=float, default=0.10)
    analysis.add_argument("--min-spearman-rho", type=float, default=0.50)
    analysis.add_argument("--timing-slope-range", type=parse_float_pair, default=(0.50, 1.50))
    analysis.add_argument("--min-timing-r2", type=float, default=0.50)
    analysis.add_argument("--top-k-plots", type=int, default=12)
    analysis.add_argument("--save-raw-responses", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    return requested


def resolve_cli_config(args: argparse.Namespace, model_data_info: dict[str, Any] | None = None) -> ResolvedRunConfig:
    """Apply quick/full defaults and validate command-line values."""

    quick = bool(args.quick)
    periods = args.period_frames or ((3,) if quick else (2, 3, 4, 5))
    n_flashes = args.n_flashes or ((0, 1, 4, 7) if quick else (0, 1, 2, 3, 4, 5, 7, 10))
    amplitudes = args.flash_amplitudes or ((0.5,) if quick else (0.25, 0.5, 0.75))
    jitter_controls = args.n_jitter_controls if args.n_jitter_controls is not None else (2 if quick else 8)
    include_embedded = args.include_embedded_omission
    if include_embedded is None:
        include_embedded = not quick
    include_sustained = args.include_sustained_controls
    if include_sustained is None:
        include_sustained = not quick
    save_raw = args.save_raw_responses
    if save_raw is None:
        save_raw = quick

    if args.fps is not None:
        fps = float(args.fps)
    elif model_data_info is not None:
        fps = float(model_data_info.get("stimulus_rate_hz") or model_data_info.get("response_rate_hz") or 30.0)
    else:
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Frame rate must be finite and positive, got {fps}")
    if any(period <= 0 for period in periods):
        raise ValueError("All periods must be positive integers")
    if args.flash_frames <= 0 or args.flash_frames >= min(periods):
        raise ValueError(f"flash_frames={args.flash_frames} must be positive and shorter than every period {periods}")
    if any(n < 0 for n in n_flashes):
        raise ValueError("n_flashes values must be nonnegative")
    if any(amplitude <= 0 or not math.isfinite(amplitude) for amplitude in amplitudes):
        raise ValueError("Flash amplitudes must be finite and positive")
    if jitter_controls < 0:
        raise ValueError("n_jitter_controls must be nonnegative")
    if args.batch_size <= 0 or args.num_threads <= 0:
        raise ValueError("batch-size and num-threads must be positive")
    if args.embedded_post_flashes < 1:
        raise ValueError("embedded-post-flashes must be at least one")
    if args.top_k_plots < 0:
        raise ValueError("top-k-plots must be nonnegative")
    if args.osr_window_ms[0] < 0 or args.osr_window_ms[1] < args.osr_window_ms[0]:
        raise ValueError("osr-window-ms must be nonnegative and ordered START,STOP")
    slope_min, slope_max = args.timing_slope_range
    if slope_max < slope_min:
        raise ValueError("timing-slope-range must be ordered MIN,MAX")
    if args.time_steps is not None and args.time_steps <= 0:
        raise ValueError("time-steps must be positive")

    polarity = ("dark", "bright") if args.polarity == "both" else (args.polarity,)
    thresholds = AnalysisThresholds(
        min_baseline_gain=float(args.min_baseline_gain),
        min_periodicity_gain=float(args.min_periodicity_gain),
        min_history_gain=float(args.min_history_gain),
        min_spearman_rho=float(args.min_spearman_rho),
        timing_slope_min=float(slope_min),
        timing_slope_max=float(slope_max),
        min_timing_r2=float(args.min_timing_r2),
    )
    return ResolvedRunConfig(
        model=args.model,
        device=_resolve_device(args.device),
        output_dir=args.output_dir.resolve(),
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        overwrite=args.overwrite,
        quick=quick,
        fps=fps,
        time_steps=args.time_steps,
        auto_time_steps=args.auto_time_steps or args.time_steps is None,
        period_frames=tuple(sorted(dict.fromkeys(periods))),
        n_flashes=tuple(sorted(dict.fromkeys(n_flashes))),
        flash_frames=args.flash_frames,
        post_omission_frames=args.post_omission_frames,
        embedded_post_flashes=args.embedded_post_flashes,
        value_space=args.value_space,
        baseline_values=args.baseline_values,
        flash_values=args.flash_values,
        flash_amplitudes=tuple(amplitudes),
        polarity=polarity,
        channel_mode=args.channel_mode,
        channel_vector=args.channel_vector,
        n_jitter_controls=jitter_controls,
        include_embedded_omission=bool(include_embedded),
        include_sustained_controls=bool(include_sustained),
        osr_window_ms=args.osr_window_ms,
        target_n_flashes=args.target_n_flashes,
        thresholds=thresholds,
        top_k_plots=args.top_k_plots,
        save_raw_responses=bool(save_raw),
    )


def _as_channel_vector(value: Any, channels: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array.item(), channels)
    elif array.ndim == 1 and array.size == 1:
        array = np.repeat(array.item(), channels)
    elif array.ndim != 1 or array.size != channels:
        raise ValueError(f"{name} must be scalar or have exactly {channels} entries; got shape {array.shape}")
    return array.astype(float, copy=False)


def normalize_raw_values(
    values: Sequence[float] | float,
    mean: Sequence[float] | float,
    std: Sequence[float] | float,
    channels: int,
) -> np.ndarray:
    """Convert raw values to model coordinates with strict finite checks."""

    values_array = _as_channel_vector(values, channels, "raw values")
    mean_array = _as_channel_vector(mean, channels, "stim_mean")
    std_array = _as_channel_vector(std, channels, "stim_std")
    if not np.isfinite(values_array).all() or not np.isfinite(mean_array).all():
        raise ValueError("Raw values and normalization means must be finite")
    if not np.isfinite(std_array).all() or np.any(std_array <= 0):
        raise ValueError("Normalization standard deviations must be finite and strictly positive")
    normalized = (values_array - mean_array) / std_array
    if not np.isfinite(normalized).all():
        raise ValueError("Raw-to-model normalization produced nonfinite values")
    return normalized


def _metadata_normalization(data_info: dict[str, Any], channels: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        mean = _as_channel_vector(data_info["stim_mean"], channels, "stim_mean")
        std = _as_channel_vector(data_info["stim_std"], channels, "stim_std")
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        return None
    return mean, std


def _hoefling_normalization_fallback(
    model_name: str,
    input_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return source-verified Höfling constants for known spatial shapes."""

    channels, height, width = input_shape
    if channels != 2 or "hoefling" not in model_name.lower():
        return None
    if (height, width) == (18, 16):
        mean = np.array(
            [
                pre_normalisation_values_18x16["channel_0_mean"],
                pre_normalisation_values_18x16["channel_1_mean"],
            ],
            dtype=float,
        )
        std = np.array(
            [
                pre_normalisation_values_18x16["channel_0_std"],
                pre_normalisation_values_18x16["channel_1_std"],
            ],
            dtype=float,
        )
        return mean, std, "hoefling_18x16_per_channel_constants"
    if (height, width) in {(72, 64), (74, 64)}:
        mean = np.repeat(float(MEAN_STD_DICT_74x64["joint_mean"]), 2)
        std = np.repeat(float(MEAN_STD_DICT_74x64["joint_std"]), 2)
        return mean, std, "hoefling_high_res_joint_constants"
    return None


def _documented_model_range(model_name: str, channels: int) -> tuple[np.ndarray, np.ndarray] | None:
    if channels != 2 or "hoefling" not in model_name.lower():
        return None
    minimum = np.array(
        [
            STIMULUS_RANGE_CONSTRAINTS["x_min_green"],
            STIMULUS_RANGE_CONSTRAINTS["x_min_uv"],
        ],
        dtype=float,
    )
    maximum = np.array(
        [
            STIMULUS_RANGE_CONSTRAINTS["x_max_green"],
            STIMULUS_RANGE_CONSTRAINTS["x_max_uv"],
        ],
        dtype=float,
    )
    return minimum, maximum


def resolve_value_space(
    *,
    requested: str,
    model_name: str,
    data_info: dict[str, Any],
    input_shape: tuple[int, int, int],
    baseline_values: Sequence[float] | None,
) -> ValueSpaceResolution:
    """Resolve model/raw/auto value-space behavior without propagating NaNs."""

    channels = input_shape[0]
    raw_mean = data_info.get("stim_mean")
    raw_std = data_info.get("stim_std")
    metadata_norm = _metadata_normalization(data_info, channels)
    documented = _documented_model_range(model_name, channels)
    documented_min = tuple(documented[0]) if documented is not None else None
    documented_max = tuple(documented[1]) if documented is not None else None

    if requested in {"model", "auto"}:
        baseline = (
            np.zeros(channels)
            if baseline_values is None
            else _as_channel_vector(baseline_values, channels, "baseline-values")
        )
        if not np.isfinite(baseline).all():
            raise ValueError("Model-space baseline contains nonfinite values")
        if requested == "auto" and metadata_norm is None:
            strategy = "auto_model_space_nonfinite_or_missing_metadata"
        elif requested == "auto":
            strategy = "auto_model_space_artificial_stimulus"
        else:
            strategy = "explicit_model_space"
        return ValueSpaceResolution(
            requested_value_space=requested,
            strategy=strategy,
            raw_stim_mean=raw_mean,
            raw_stim_std=raw_std,
            normalization_mean=None,
            normalization_std=None,
            baseline_model=tuple(float(v) for v in baseline),
            documented_min=documented_min,
            documented_max=documented_max,
        )

    normalization = metadata_norm
    source = "finite_model_metadata"
    if normalization is None:
        fallback = _hoefling_normalization_fallback(model_name, input_shape)
        if fallback is None:
            raise ValueError(
                "Raw value-space was requested, but finite stim_mean/stim_std and an unambiguous "
                "model-specific fallback are unavailable"
            )
        normalization = fallback[0], fallback[1]
        source = fallback[2]
    mean, std = normalization
    raw_baseline = mean if baseline_values is None else _as_channel_vector(baseline_values, channels, "baseline-values")
    baseline_model = normalize_raw_values(raw_baseline, mean, std, channels)
    return ValueSpaceResolution(
        requested_value_space=requested,
        strategy=f"raw_to_model_{source}",
        raw_stim_mean=raw_mean,
        raw_stim_std=raw_std,
        normalization_mean=tuple(float(v) for v in mean),
        normalization_std=tuple(float(v) for v in std),
        baseline_model=tuple(float(v) for v in baseline_model),
        documented_min=documented_min,
        documented_max=documented_max,
    )


def resolve_channel_vector(mode: str, custom: Sequence[float] | None, channels: int) -> np.ndarray:
    """Resolve the channel mask/vector used to apply scalar flash amplitudes."""

    if mode == "all":
        vector = np.ones(channels)
    elif mode == "channel0":
        vector = np.zeros(channels)
        vector[0] = 1.0
    elif mode == "channel1":
        if channels < 2:
            raise ValueError("channel1 mode requires a model with at least two channels")
        vector = np.zeros(channels)
        vector[1] = 1.0
    elif mode == "custom":
        if custom is None:
            raise ValueError("--channel-mode custom requires --channel-vector")
        vector = _as_channel_vector(custom, channels, "channel-vector")
    else:
        raise ValueError(f"Unknown channel mode {mode!r}")
    if not np.isfinite(vector).all() or np.allclose(vector, 0):
        raise ValueError("Channel vector must be finite and not identically zero")
    return vector


def resolve_flash_vectors(
    config: ResolvedRunConfig,
    value_resolution: ValueSpaceResolution,
    channels: int,
) -> list[tuple[float, str, np.ndarray]]:
    """Expand amplitudes/polarities or normalize an explicit flash vector."""

    baseline_model = np.asarray(value_resolution.baseline_model, dtype=float)
    if config.flash_values is not None:
        if value_resolution.requested_value_space == "raw":
            assert value_resolution.normalization_mean is not None
            assert value_resolution.normalization_std is not None
            flash_model = normalize_raw_values(
                config.flash_values,
                value_resolution.normalization_mean,
                value_resolution.normalization_std,
                channels,
            )
        else:
            flash_model = _as_channel_vector(config.flash_values, channels, "flash-values")
        amplitude = float(np.linalg.norm(flash_model - baseline_model))
        return [(amplitude, "explicit", flash_model)]

    channel_vector = resolve_channel_vector(config.channel_mode, config.channel_vector, channels)
    resolved: list[tuple[float, str, np.ndarray]] = []
    for amplitude in config.flash_amplitudes:
        for polarity in config.polarity:
            signed = -amplitude if polarity == "dark" else amplitude
            if value_resolution.requested_value_space == "raw":
                assert value_resolution.normalization_std is not None
                raw_baseline = np.asarray(value_resolution.normalization_mean, dtype=float)
                if config.baseline_values is not None:
                    raw_baseline = _as_channel_vector(config.baseline_values, channels, "baseline-values")
                raw_flash = raw_baseline + signed * channel_vector
                flash_model = normalize_raw_values(
                    raw_flash,
                    value_resolution.normalization_mean,
                    value_resolution.normalization_std,
                    channels,
                )
            else:
                flash_model = baseline_model + signed * channel_vector
            resolved.append((float(amplitude), polarity, flash_model))
    return resolved


def resolve_timing(
    *,
    fps: float,
    cut_frames: int,
    output_time_for_requested_input: Callable[[int], int],
    periods: Sequence[int],
    n_flashes: Sequence[int],
    flash_frames: int,
    osr_window_ms: tuple[float, float],
    requested_time_steps: int | None,
    embedded_post_flashes: int,
    include_embedded: bool,
    post_omission_frames: int | None = None,
) -> TimingResolution:
    """Resolve a valid expected event and duration, with explicit alignment checks."""

    osr_start = int(math.ceil(osr_window_ms[0] * fps / 1000.0 - 1e-12))
    osr_stop = int(math.floor(osr_window_ms[1] * fps / 1000.0 + 1e-12))
    if osr_stop < osr_start:
        raise ValueError(
            f"OSR window {osr_window_ms} ms contains no frames at {fps:g} Hz (resolved offsets {osr_start}..{osr_stop})"
        )
    max_n = max(n_flashes, default=0)
    max_period = max(periods)
    pre_context = max(5, flash_frames + 1)
    expected = max(cut_frames + 1, max_n * max_period + pre_context)
    required_after = osr_stop
    if post_omission_frames is not None:
        required_after = max(required_after, post_omission_frames)
    if include_embedded:
        required_after = max(required_after, embedded_post_flashes * max_period + flash_frames - 1)
    minimum_time_steps = expected + required_after + 1
    time_steps = requested_time_steps if requested_time_steps is not None else minimum_time_steps
    if time_steps < minimum_time_steps:
        raise ValueError(
            f"time_steps={time_steps} is too short; at least {minimum_time_steps} frames are required "
            f"for expected_frame={expected}, cut={cut_frames}, and post-event extent={required_after}"
        )
    output_time = output_time_for_requested_input(time_steps)
    if output_time <= 0:
        raise ValueError(f"Model produced no output frames for time_steps={time_steps}")
    inferred_cut = time_steps - output_time
    if inferred_cut != cut_frames:
        raise ValueError(
            f"Temporal cut changed with duration: probe cut={cut_frames}, "
            f"time_steps-output_time={inferred_cut} for time_steps={time_steps}"
        )
    expected_output = expected - cut_frames
    if expected_output + osr_start < 0 or expected_output + osr_stop >= output_time:
        raise ValueError(
            "Expected omission/search window does not map inside model output: "
            f"expected_input={expected}, expected_output={expected_output}, "
            f"offsets={osr_start}..{osr_stop}, output_time={output_time}"
        )
    pre_offsets = tuple(offset for offset in range(-2, 0) if 0 <= expected_output + offset < output_time)
    return TimingResolution(
        fps=fps,
        cut_frames=cut_frames,
        expected_frame=expected,
        time_steps=time_steps,
        output_time_steps=output_time,
        osr_start_offset=osr_start,
        osr_stop_offset=osr_stop,
        osr_start_ms_resolved=1000.0 * osr_start / fps,
        osr_stop_ms_resolved=1000.0 * osr_stop / fps,
        pre_event_offsets=pre_offsets,
        minimum_time_steps=minimum_time_steps,
    )


def periodic_flash_onsets(expected_frame: int, n_flashes: int, period_frames: int) -> tuple[int, ...]:
    """Return onsets ending one period before the expected event."""

    return tuple(expected_frame - k * period_frames for k in range(n_flashes, 0, -1))


def generate_aperiodic_onsets(
    *,
    expected_frame: int,
    n_flashes: int,
    period_frames: int,
    flash_frames: int,
    seed: int,
) -> tuple[int, ...] | None:
    """Generate duration-matched irregular onsets, or return ``None`` if impossible."""

    if n_flashes < 3 or period_frames - 1 < flash_frames + 1:
        return None
    periodic = periodic_flash_onsets(expected_frame, n_flashes, period_frames)
    n_intervals = n_flashes - 1
    rng = np.random.default_rng(seed)
    jitter = np.zeros(n_intervals, dtype=int)
    plus, minus = rng.choice(n_intervals, size=2, replace=False)
    jitter[plus] = 1
    jitter[minus] = -1
    intervals = period_frames + jitter
    if np.any(intervals < flash_frames + 1):
        return None
    onsets = [periodic[0]]
    for interval in intervals:
        onsets.append(onsets[-1] + int(interval))
    result = tuple(onsets)
    if result[0] != periodic[0] or result[-1] != periodic[-1]:
        raise AssertionError("Aperiodic control did not preserve first/last onset")
    if len(set(np.diff(result).tolist())) == 1:
        raise AssertionError("Aperiodic jitter accidentally remained periodic")
    return result


def render_full_field_trace(
    *,
    channels: int,
    time_steps: int,
    baseline: Sequence[float],
    flash: Sequence[float],
    onsets: Sequence[int],
    flash_frames: int,
) -> np.ndarray:
    """Render a channels-by-time spatially uniform trace."""

    baseline_array = _as_channel_vector(baseline, channels, "baseline")
    flash_array = _as_channel_vector(flash, channels, "flash")
    trace = np.repeat(baseline_array[:, None], time_steps, axis=1)
    for onset in onsets:
        if onset < 0 or onset + flash_frames > time_steps:
            raise ValueError(f"Flash onset {onset} with duration {flash_frames} is outside 0..{time_steps - 1}")
        trace[:, onset : onset + flash_frames] = flash_array[:, None]
    if not np.isfinite(trace).all():
        raise ValueError("Generated stimulus contains NaN or infinity")
    return trace


def _make_condition(
    *,
    prefix: str,
    condition_type: str,
    period: int,
    n_flashes: int,
    flash_frames: int,
    amplitude: float,
    polarity: str,
    channel_vector: np.ndarray,
    baseline: np.ndarray,
    flash: np.ndarray,
    variant: int,
    control_seed: int | None,
    onsets: tuple[int, ...],
    expected: int,
    expected_present: bool,
    resumes: bool,
    time_steps: int,
) -> GeneratedCondition:
    last_real_candidates = [onset for onset in onsets if onset < expected]
    last_real = max(last_real_candidates) if last_real_candidates else None
    condition_id = f"{prefix}_T{period}_n{n_flashes}_a{amplitude:g}_{polarity}_v{variant}".replace(".", "p").replace(
        "-", "m"
    )
    metadata = StimulusCondition(
        condition_id=condition_id,
        condition_type=condition_type,
        period_frames=period,
        n_flashes=n_flashes,
        flash_frames=flash_frames,
        amplitude=amplitude,
        polarity=polarity,
        channel_vector=tuple(float(v) for v in channel_vector),
        baseline_vector=tuple(float(v) for v in baseline),
        flash_vector=tuple(float(v) for v in flash),
        variant=variant,
        control_seed=control_seed,
        flash_onsets=onsets,
        last_real_flash_frame=last_real,
        expected_flash_frame=expected,
        expected_flash_present=expected_present,
        flashes_resume=resumes,
        time_steps=time_steps,
    )
    trace = render_full_field_trace(
        channels=baseline.size,
        time_steps=time_steps,
        baseline=baseline,
        flash=flash,
        onsets=onsets,
        flash_frames=flash_frames,
    )
    return GeneratedCondition(metadata=metadata, trace=trace)


def generate_conditions(
    config: ResolvedRunConfig,
    timing: TimingResolution,
    value_resolution: ValueSpaceResolution,
    input_shape: tuple[int, int, int],
) -> list[GeneratedCondition]:
    """Generate the required periodic, matched, embedded, and level controls."""

    channels = input_shape[0]
    baseline = np.asarray(value_resolution.baseline_model, dtype=float)
    channel_vector = resolve_channel_vector(config.channel_mode, config.channel_vector, channels)
    flash_settings = resolve_flash_vectors(config, value_resolution, channels)
    conditions: list[GeneratedCondition] = []

    for amplitude, polarity, flash in flash_settings:
        for period in config.period_frames:
            conditions.append(
                _make_condition(
                    prefix="steady_baseline",
                    condition_type="steady_baseline",
                    period=period,
                    n_flashes=0,
                    flash_frames=config.flash_frames,
                    amplitude=amplitude,
                    polarity=polarity,
                    channel_vector=channel_vector,
                    baseline=baseline,
                    flash=flash,
                    variant=0,
                    control_seed=None,
                    onsets=(),
                    expected=timing.expected_frame,
                    expected_present=False,
                    resumes=False,
                    time_steps=timing.time_steps,
                )
            )
            positive_ns = [n for n in config.n_flashes if n > 0]
            for n_flashes in positive_ns:
                periodic = periodic_flash_onsets(timing.expected_frame, n_flashes, period)
                conditions.append(
                    _make_condition(
                        prefix="periodic_omission_terminal",
                        condition_type="periodic_omission_terminal",
                        period=period,
                        n_flashes=n_flashes,
                        flash_frames=config.flash_frames,
                        amplitude=amplitude,
                        polarity=polarity,
                        channel_vector=channel_vector,
                        baseline=baseline,
                        flash=flash,
                        variant=0,
                        control_seed=None,
                        onsets=periodic,
                        expected=timing.expected_frame,
                        expected_present=False,
                        resumes=False,
                        time_steps=timing.time_steps,
                    )
                )
                conditions.append(
                    _make_condition(
                        prefix="periodic_continuation",
                        condition_type="periodic_continuation",
                        period=period,
                        n_flashes=n_flashes,
                        flash_frames=config.flash_frames,
                        amplitude=amplitude,
                        polarity=polarity,
                        channel_vector=channel_vector,
                        baseline=baseline,
                        flash=flash,
                        variant=0,
                        control_seed=None,
                        onsets=periodic + (timing.expected_frame,),
                        expected=timing.expected_frame,
                        expected_present=True,
                        resumes=False,
                        time_steps=timing.time_steps,
                    )
                )
                for variant in range(config.n_jitter_controls):
                    control_seed = config.seed + period * 100_000 + n_flashes * 1_000 + variant
                    aperiodic = generate_aperiodic_onsets(
                        expected_frame=timing.expected_frame,
                        n_flashes=n_flashes,
                        period_frames=period,
                        flash_frames=config.flash_frames,
                        seed=control_seed,
                    )
                    if aperiodic is None:
                        break
                    conditions.append(
                        _make_condition(
                            prefix="aperiodic_stop",
                            condition_type="aperiodic_stop",
                            period=period,
                            n_flashes=n_flashes,
                            flash_frames=config.flash_frames,
                            amplitude=amplitude,
                            polarity=polarity,
                            channel_vector=channel_vector,
                            baseline=baseline,
                            flash=flash,
                            variant=variant,
                            control_seed=control_seed,
                            onsets=aperiodic,
                            expected=timing.expected_frame,
                            expected_present=False,
                            resumes=False,
                            time_steps=timing.time_steps,
                        )
                    )
                if config.include_embedded_omission:
                    resumed = tuple(
                        timing.expected_frame + k * period for k in range(1, config.embedded_post_flashes + 1)
                    )
                    conditions.append(
                        _make_condition(
                            prefix="embedded_omission",
                            condition_type="embedded_omission",
                            period=period,
                            n_flashes=n_flashes,
                            flash_frames=config.flash_frames,
                            amplitude=amplitude,
                            polarity=polarity,
                            channel_vector=channel_vector,
                            baseline=baseline,
                            flash=flash,
                            variant=0,
                            control_seed=None,
                            onsets=periodic + resumed,
                            expected=timing.expected_frame,
                            expected_present=False,
                            resumes=True,
                            time_steps=timing.time_steps,
                        )
                    )
                    conditions.append(
                        _make_condition(
                            prefix="embedded_continuation",
                            condition_type="embedded_continuation",
                            period=period,
                            n_flashes=n_flashes,
                            flash_frames=config.flash_frames,
                            amplitude=amplitude,
                            polarity=polarity,
                            channel_vector=channel_vector,
                            baseline=baseline,
                            flash=flash,
                            variant=0,
                            control_seed=None,
                            onsets=periodic + (timing.expected_frame,) + resumed,
                            expected=timing.expected_frame,
                            expected_present=True,
                            resumes=True,
                            time_steps=timing.time_steps,
                        )
                    )

            if config.include_sustained_controls:
                diagnostic_n = _resolved_target_n(config.n_flashes, config.target_n_flashes)
                last_flash = timing.expected_frame - period if diagnostic_n > 0 else timing.expected_frame
                conditions.append(
                    _make_condition(
                        prefix="baseline_only",
                        condition_type="baseline_only",
                        period=period,
                        n_flashes=0,
                        flash_frames=config.flash_frames,
                        amplitude=amplitude,
                        polarity=polarity,
                        channel_vector=channel_vector,
                        baseline=baseline,
                        flash=flash,
                        variant=0,
                        control_seed=None,
                        onsets=(),
                        expected=timing.expected_frame,
                        expected_present=False,
                        resumes=False,
                        time_steps=timing.time_steps,
                    )
                )
                flash_level_trace = np.repeat(flash[:, None], timing.time_steps, axis=1)
                metadata = StimulusCondition(
                    condition_id=(
                        f"flash_level_only_T{period}_n0_a{amplitude:g}_{polarity}_v0".replace(".", "p").replace(
                            "-", "m"
                        )
                    ),
                    condition_type="flash_level_only",
                    period_frames=period,
                    n_flashes=0,
                    flash_frames=config.flash_frames,
                    amplitude=amplitude,
                    polarity=polarity,
                    channel_vector=tuple(float(v) for v in channel_vector),
                    baseline_vector=tuple(float(v) for v in baseline),
                    flash_vector=tuple(float(v) for v in flash),
                    variant=0,
                    control_seed=None,
                    flash_onsets=(0,),
                    last_real_flash_frame=0,
                    expected_flash_frame=timing.expected_frame,
                    expected_flash_present=False,
                    flashes_resume=False,
                    time_steps=timing.time_steps,
                )
                conditions.append(GeneratedCondition(metadata, flash_level_trace))
                hold_trace = np.repeat(baseline[:, None], timing.time_steps, axis=1)
                hold_trace[:, last_flash:] = flash[:, None]
                hold_metadata = StimulusCondition(
                    condition_id=(
                        f"last_flash_then_hold_flash_T{period}_n{diagnostic_n}_a{amplitude:g}_{polarity}_v0".replace(
                            ".", "p"
                        ).replace("-", "m")
                    ),
                    condition_type="last_flash_then_hold_flash",
                    period_frames=period,
                    n_flashes=diagnostic_n,
                    flash_frames=config.flash_frames,
                    amplitude=amplitude,
                    polarity=polarity,
                    channel_vector=tuple(float(v) for v in channel_vector),
                    baseline_vector=tuple(float(v) for v in baseline),
                    flash_vector=tuple(float(v) for v in flash),
                    variant=0,
                    control_seed=None,
                    flash_onsets=(last_flash,),
                    last_real_flash_frame=last_flash,
                    expected_flash_frame=timing.expected_frame,
                    expected_flash_present=False,
                    flashes_resume=False,
                    time_steps=timing.time_steps,
                )
                conditions.append(GeneratedCondition(hold_metadata, hold_trace))

    ids = [condition.metadata.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        duplicates = sorted({condition_id for condition_id in ids if ids.count(condition_id) > 1})
        raise AssertionError(f"Duplicate condition IDs generated: {duplicates}")
    validate_conditions(conditions, timing)
    return conditions


def validate_conditions(conditions: Sequence[GeneratedCondition], timing: TimingResolution) -> None:
    """Validate finite values and input/output search-window bounds."""

    expected_output = timing.expected_frame - timing.cut_frames
    if expected_output + timing.osr_start_offset < 0:
        raise ValueError("OSR window starts before model output")
    if expected_output + timing.osr_stop_offset >= timing.output_time_steps:
        raise ValueError("OSR window ends after model output")
    for generated in conditions:
        condition = generated.metadata
        if generated.trace.shape[1] != timing.time_steps:
            raise ValueError(f"{condition.condition_id} has an inconsistent trace duration")
        if not np.isfinite(generated.trace).all():
            raise ValueError(f"{condition.condition_id} contains nonfinite stimulus values")
        if condition.condition_type in {"periodic_omission_terminal", "embedded_omission"}:
            if condition.expected_flash_frame in condition.flash_onsets:
                raise ValueError(f"{condition.condition_id} contains a flash at its omitted event")
        if condition.condition_type in {"periodic_continuation", "embedded_continuation"}:
            if condition.expected_flash_frame not in condition.flash_onsets:
                raise ValueError(f"{condition.condition_id} lacks its expected continuation flash")


def stimulus_channel_statistics(conditions: Sequence[GeneratedCondition]) -> list[dict[str, float]]:
    """Summarize model-space stimulus values per channel."""

    traces = np.stack([condition.trace for condition in conditions])
    records = []
    for channel in range(traces.shape[1]):
        values = traces[:, channel, :]
        records.append(
            {
                "channel": channel,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )
    return records


def check_documented_range(
    stats: Sequence[dict[str, float]],
    value_resolution: ValueSpaceResolution,
) -> list[str]:
    """Return warnings for generated values outside a documented model range."""

    if value_resolution.documented_min is None or value_resolution.documented_max is None:
        return []
    messages = []
    for record, minimum, maximum in zip(
        stats, value_resolution.documented_min, value_resolution.documented_max, strict=True
    ):
        if record["min"] < minimum or record["max"] > maximum:
            messages.append(
                f"Channel {record['channel']} generated range [{record['min']:.4g}, {record['max']:.4g}] "
                f"exceeds documented model range [{minimum:.4g}, {maximum:.4g}]"
            )
    return messages


def build_neuron_mapping(model: torch.nn.Module) -> pd.DataFrame:
    """Build an explicit sorted session/local/global neuron mapping."""

    data_info = getattr(model, "data_info", {})
    n_neurons_dict = data_info.get("n_neurons_dict", {})
    if hasattr(model, "readout") and hasattr(model.readout, "readout_keys"):
        keys = list(model.readout.readout_keys())
    elif n_neurons_dict:
        keys = sorted(n_neurons_dict)
    else:
        raise ValueError("Model exposes neither readout keys nor data_info['n_neurons_dict']")
    session_kwargs = data_info.get("sessions_kwargs", {})
    records: list[dict[str, Any]] = []
    global_index = 0
    for key in keys:
        if key in n_neurons_dict:
            count = int(n_neurons_dict[key])
        elif hasattr(model, "readout"):
            count = int(model.readout[key].number_of_neurons())
        else:
            raise ValueError(f"Cannot determine neuron count for session {key}")
        if hasattr(model, "readout"):
            readout_count = int(model.readout[key].number_of_neurons())
            if readout_count != count:
                raise ValueError(f"Neuron count mismatch for {key}: metadata={count}, readout={readout_count}")
        metadata = session_kwargs.get(key, {})
        roi_ids = np.asarray(metadata.get("roi_ids", []))
        groups = np.asarray(metadata.get("group_assignment", []))
        eye = metadata.get("eye")
        for local_index in range(count):
            roi_id = roi_ids[local_index].item() if roi_ids.size == count else None
            group_id = groups[local_index].item() if groups.size == count else None
            if group_id is not None:
                group_name = RGC_GROUP_NAMES_DICT.get(int(group_id))
                group_class = RGC_GROUP_GROUP_ID_TO_CLASS_NAME.get(int(group_id))
            else:
                group_name = None
                group_class = None
            records.append(
                {
                    "global_neuron_index": global_index,
                    "session_key": key,
                    "local_neuron_index": local_index,
                    "roi_id": roi_id,
                    "group_id": group_id,
                    "group_name": group_name,
                    "group_class": group_class,
                    "eye": eye,
                }
            )
            global_index += 1
    return pd.DataFrame.from_records(records)


def forward_all_readouts(model: torch.nn.Module, stimuli: torch.Tensor) -> torch.Tensor:
    """Run the core once and concatenate explicitly sorted session readouts."""

    if hasattr(model, "core") and hasattr(model, "readout") and hasattr(model.readout, "readout_keys"):
        core_output = model.core(stimuli)
        keys = list(model.readout.readout_keys())
        # Session wrappers can perform required temporal reshaping and apply a
        # readout-level nonlinearity, so invoke the wrapper with each explicit
        # sorted key rather than bypassing it via ``model.readout[key]``.
        outputs = [model.readout(core_output, data_key=key) for key in keys]
        return torch.cat(outputs, dim=-1)
    output = model(stimuli)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Model forward returned {type(output).__name__}, expected torch.Tensor")
    return output


def probe_model(
    model: torch.nn.Module,
    *,
    device: str,
    time_steps: int = 50,
) -> tuple[tuple[int, int, int, int, int], tuple[int, ...], int]:
    """Run a finite random probe and infer the temporal cut."""

    shape = tuple(int(value) for value in model.stimulus_shape(time_steps=time_steps))
    if len(shape) != 5:
        raise ValueError(
            f"model.stimulus_shape(time_steps={time_steps}) returned {shape}; expected "
            "(batch, channels, time, height, width)"
        )
    with torch.inference_mode():
        output = forward_all_readouts(model, torch.rand(shape, device=device))
    if output.ndim != 3:
        raise ValueError(f"Model output must be batch,time,neuron; got shape {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Random probe produced nonfinite model output")
    cut = time_steps - int(output.shape[1])
    metadata_cut = getattr(model, "data_info", {}).get("model_cut_frames")
    if metadata_cut is not None:
        try:
            metadata_cut_float = float(metadata_cut)
        except (TypeError, ValueError):
            metadata_cut_float = math.nan
        if math.isfinite(metadata_cut_float) and metadata_cut_float.is_integer():
            if int(metadata_cut_float) != cut:
                raise ValueError(
                    f"Inferred model cut {cut} disagrees with finite metadata model_cut_frames={metadata_cut}"
                )
    return shape, tuple(int(value) for value in output.shape), cut


def validate_baseline_and_random_probes(
    model: torch.nn.Module,
    *,
    device: str,
    input_shape: tuple[int, int, int],
    baseline: Sequence[float],
    time_steps: int,
) -> dict[str, tuple[int, ...]]:
    """Run required finite baseline-only and random-normal probes."""

    channels, height, width = input_shape
    baseline_vector = _as_channel_vector(baseline, channels, "baseline")
    baseline_tensor = torch.as_tensor(baseline_vector, dtype=torch.float32, device=device)
    baseline_tensor = baseline_tensor.view(1, channels, 1, 1, 1).expand(1, channels, time_steps, height, width)
    random_tensor = torch.randn((1, channels, time_steps, height, width), device=device)
    shapes: dict[str, tuple[int, ...]] = {}
    with torch.inference_mode():
        for name, tensor in (("baseline", baseline_tensor), ("random_normal", random_tensor)):
            output = forward_all_readouts(model, tensor)
            if not torch.isfinite(output).all():
                raise ValueError(f"{name} probe produced nonfinite model outputs")
            shapes[name] = tuple(int(value) for value in output.shape)
    return shapes


def infer_conditions(
    model: torch.nn.Module,
    conditions: Sequence[GeneratedCondition],
    *,
    input_shape: tuple[int, int, int],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run condition traces in bounded batches and return condition,time,neuron."""

    channels, height, width = input_shape
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(conditions), batch_size):
            batch_conditions = conditions[start : start + batch_size]
            compact = np.stack([condition.trace for condition in batch_conditions]).astype(np.float32)
            tensor = torch.from_numpy(compact).to(device)
            tensor = tensor[:, :, :, None, None].expand(-1, channels, -1, height, width).contiguous()
            output = forward_all_readouts(model, tensor)
            if output.ndim != 3:
                raise ValueError(f"Model output must have three dimensions, got {tuple(output.shape)}")
            if not torch.isfinite(output).all():
                bad = [condition.metadata.condition_id for condition in batch_conditions]
                raise ValueError(f"Nonfinite model responses for conditions {bad}")
            outputs.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
            LOGGER.info("Inference conditions %d-%d/%d", start + 1, start + len(batch_conditions), len(conditions))
    responses = np.concatenate(outputs, axis=0)
    if responses.shape[0] != len(conditions):
        raise AssertionError("Condition response count changed during inference")
    return responses


def _condition_match_key(condition: StimulusCondition) -> tuple[int, float, str, tuple[float, ...]]:
    return (
        condition.period_frames,
        condition.amplitude,
        condition.polarity,
        condition.flash_vector,
    )


def calculate_condition_metrics(
    *,
    conditions: Sequence[GeneratedCondition],
    responses: np.ndarray,
    timing: TimingResolution,
    neuron_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calculate raw and globally scaled per-neuron/per-condition metrics."""

    if responses.ndim != 3:
        raise ValueError("responses must have shape condition,output_time,neuron")
    if responses.shape[0] != len(conditions):
        raise ValueError("responses condition dimension does not match condition metadata")
    if responses.shape[2] != len(neuron_mapping):
        raise ValueError("responses neuron dimension does not match explicit neuron mapping")
    finite_responses = np.where(np.isfinite(responses), responses, np.nan)
    p99 = np.nanpercentile(finite_responses, 99, axis=(0, 1))
    p01 = np.nanpercentile(finite_responses, 1, axis=(0, 1))
    raw_scale = p99 - p01
    scale = np.where(np.isfinite(raw_scale), np.maximum(raw_scale, SCALE_FLOOR), np.nan)

    expected_output = timing.expected_frame - timing.cut_frames
    search_indices = np.arange(
        expected_output + timing.osr_start_offset,
        expected_output + timing.osr_stop_offset + 1,
    )
    early_indices = np.asarray(
        [expected_output + offset for offset in timing.pre_event_offsets],
        dtype=int,
    )
    baseline_indices: dict[tuple[int, float, str, tuple[float, ...]], int] = {}
    for index, generated in enumerate(conditions):
        if generated.metadata.condition_type == "steady_baseline":
            baseline_indices[_condition_match_key(generated.metadata)] = index

    mapping_records = neuron_mapping.to_dict(orient="records")
    records: list[dict[str, Any]] = []
    for condition_index, generated in enumerate(conditions):
        condition = generated.metadata
        key = _condition_match_key(condition)
        if key not in baseline_indices:
            raise ValueError(f"No matching steady baseline for {condition.condition_id}")
        response = responses[condition_index]
        baseline_response = responses[baseline_indices[key]]
        window = response[search_indices, :]
        baseline_window = baseline_response[search_indices, :]
        finite = np.isfinite(window).all(axis=0) & np.isfinite(baseline_window).all(axis=0)
        safe_window = np.where(np.isfinite(window), window, -np.inf)
        peak_arg = np.argmax(safe_window, axis=0)
        peak = safe_window[peak_arg, np.arange(response.shape[1])]
        peak[~finite] = np.nan
        peak_output_frame = search_indices[peak_arg].astype(float)
        peak_output_frame[~finite] = np.nan
        peak_input_frame = peak_output_frame + timing.cut_frames
        baseline_peak = np.max(baseline_window, axis=0)
        mean_response = np.mean(window, axis=0)
        auc = np.sum(window - baseline_window, axis=0)
        early_mean = (
            np.mean(response[early_indices, :], axis=0) if early_indices.size else np.full(response.shape[1], np.nan)
        )
        last_real = condition.last_real_flash_frame
        latency_last = peak_input_frame - last_real if last_real is not None else np.full(response.shape[1], np.nan)
        for neuron_index, mapping in enumerate(mapping_records):
            peak_above = peak[neuron_index] - baseline_peak[neuron_index]
            records.append(
                {
                    **mapping,
                    "condition_index": condition_index,
                    "condition_id": condition.condition_id,
                    "condition_type": condition.condition_type,
                    "period_frames": condition.period_frames,
                    "period_hz": timing.fps / condition.period_frames,
                    "n_flashes": condition.n_flashes,
                    "amplitude": condition.amplitude,
                    "polarity": condition.polarity,
                    "variant": condition.variant,
                    "expected_flash_present": condition.expected_flash_present,
                    "flashes_resume": condition.flashes_resume,
                    "mean_response": mean_response[neuron_index],
                    "peak_response": peak[neuron_index],
                    "steady_baseline_peak": baseline_peak[neuron_index],
                    "peak_above_steady_baseline": peak_above,
                    "auc_above_steady_baseline": auc[neuron_index],
                    "peak_output_frame": peak_output_frame[neuron_index],
                    "peak_input_frame": peak_input_frame[neuron_index],
                    "peak_latency_expected_frames": (peak_input_frame[neuron_index] - condition.expected_flash_frame),
                    "peak_latency_expected_ms": (
                        1000.0 * (peak_input_frame[neuron_index] - condition.expected_flash_frame) / timing.fps
                    ),
                    "peak_latency_last_flash_frames": latency_last[neuron_index],
                    "peak_latency_last_flash_ms": 1000.0 * latency_last[neuron_index] / timing.fps,
                    "early_control_mean": early_mean[neuron_index],
                    "response_scale_raw": raw_scale[neuron_index],
                    "response_scale": scale[neuron_index],
                    "peak_response_normalized": peak[neuron_index] / scale[neuron_index],
                    "peak_above_steady_baseline_normalized": peak_above / scale[neuron_index],
                    "auc_above_steady_baseline_normalized": auc[neuron_index] / scale[neuron_index],
                    "finite_data": bool(finite[neuron_index]),
                }
            )
    return pd.DataFrame.from_records(records), scale


def _resolved_target_n(n_values: Sequence[int], requested: int) -> int:
    positive = sorted(n for n in n_values if n > 0)
    if not positive:
        raise ValueError("At least one positive n_flashes value is required")
    return requested if requested in positive else max(value for value in positive if value <= max(positive))


def _safe_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def _linear_timing_fit(periods: np.ndarray, latencies: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(periods) & np.isfinite(latencies)
    x = periods[finite]
    y = latencies[finite]
    if x.size < 2 or np.unique(x).size < 2:
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum((y - prediction) ** 2) / denominator if denominator > 0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def _monotonic_fraction(values: Sequence[float], tolerance: float = 0.02) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return math.nan
    return float(np.mean(np.diff(finite) >= -tolerance))


def summarize_neurons(
    *,
    condition_metrics: pd.DataFrame,
    neuron_mapping: pd.DataFrame,
    config: ResolvedRunConfig,
) -> pd.DataFrame:
    """Calculate core gains, history/timing dependence, labels, and ranking."""

    target_n = _resolved_target_n(config.n_flashes, config.target_n_flashes)
    records: list[dict[str, Any]] = []
    for neuron_index, sub in condition_metrics.groupby("global_neuron_index", sort=True):
        mapping = neuron_mapping.loc[neuron_mapping["global_neuron_index"] == neuron_index].iloc[0].to_dict()
        target_periodic = sub[(sub["condition_type"] == "periodic_omission_terminal") & (sub["n_flashes"] == target_n)]
        combo_effects: list[dict[str, float]] = []
        for _, periodic_row in target_periodic.iterrows():
            match = (
                (sub["period_frames"] == periodic_row["period_frames"])
                & (sub["amplitude"] == periodic_row["amplitude"])
                & (sub["polarity"] == periodic_row["polarity"])
            )
            aperiodic = sub[match & (sub["condition_type"] == "aperiodic_stop") & (sub["n_flashes"] == target_n)]
            continuation = sub[
                match & (sub["condition_type"] == "periodic_continuation") & (sub["n_flashes"] == target_n)
            ]
            history_n1 = sub[match & (sub["condition_type"] == "periodic_omission_terminal") & (sub["n_flashes"] == 1)]
            embedded_omission = sub[
                match & (sub["condition_type"] == "embedded_omission") & (sub["n_flashes"] == target_n)
            ]
            embedded_continuation = sub[
                match & (sub["condition_type"] == "embedded_continuation") & (sub["n_flashes"] == target_n)
            ]
            baseline_gain = float(periodic_row["peak_above_steady_baseline"])
            baseline_gain_norm = float(periodic_row["peak_above_steady_baseline_normalized"])
            periodicity_gain = (
                float(periodic_row["peak_response"] - aperiodic["peak_response"].median())
                if not aperiodic.empty
                else math.nan
            )
            scale = float(periodic_row["response_scale"])
            history_n0 = baseline_gain
            history_n1_gain = (
                float(periodic_row["peak_response"] - history_n1["peak_response"].median())
                if not history_n1.empty
                else math.nan
            )
            continuation_gain = (
                float(periodic_row["peak_response"] - continuation["peak_response"].median())
                if not continuation.empty
                else math.nan
            )
            embedded_peak_gain = (
                float(embedded_omission["peak_response"].median() - embedded_continuation["peak_response"].median())
                if not embedded_omission.empty and not embedded_continuation.empty
                else math.nan
            )
            embedded_auc_gain = (
                float(
                    embedded_omission["auc_above_steady_baseline"].median()
                    - embedded_continuation["auc_above_steady_baseline"].median()
                )
                if not embedded_omission.empty and not embedded_continuation.empty
                else math.nan
            )
            combo_effects.append(
                {
                    "baseline": baseline_gain,
                    "baseline_norm": baseline_gain_norm,
                    "periodicity": periodicity_gain,
                    "periodicity_norm": periodicity_gain / scale if np.isfinite(periodicity_gain) else math.nan,
                    "history_n0": history_n0,
                    "history_n0_norm": history_n0 / scale,
                    "history_n1": history_n1_gain,
                    "history_n1_norm": history_n1_gain / scale if np.isfinite(history_n1_gain) else math.nan,
                    "continuation": continuation_gain,
                    "continuation_norm": continuation_gain / scale if np.isfinite(continuation_gain) else math.nan,
                    "embedded_peak": embedded_peak_gain,
                    "embedded_peak_norm": (embedded_peak_gain / scale if np.isfinite(embedded_peak_gain) else math.nan),
                    "embedded_auc": embedded_auc_gain,
                    "embedded_auc_norm": (embedded_auc_gain / scale if np.isfinite(embedded_auc_gain) else math.nan),
                }
            )

        periodic_all = sub[sub["condition_type"] == "periodic_omission_terminal"]
        rho_values: list[float] = []
        monotonic_values: list[float] = []
        for (_, amplitude, polarity), series in periodic_all.groupby(["period_frames", "amplitude", "polarity"]):
            series = series.sort_values("n_flashes")
            n_values = [0, *series["n_flashes"].astype(int).tolist()]
            peak_values = [0.0, *series["peak_above_steady_baseline_normalized"].tolist()]
            if len(set(n_values)) >= 3 and np.isfinite(peak_values).all() and np.std(peak_values) > 0:
                rho = spearmanr(n_values, peak_values).statistic
                rho_values.append(float(rho))
            monotonic_values.append(_monotonic_fraction(peak_values))

        timing_periods: list[float] = []
        timing_last: list[float] = []
        timing_expected: list[float] = []
        for period, rows in target_periodic.groupby("period_frames"):
            timing_periods.append(float(period))
            timing_last.append(_safe_median(rows["peak_latency_last_flash_frames"]))
            timing_expected.append(_safe_median(rows["peak_latency_expected_frames"]))
        period_array = np.asarray(timing_periods, dtype=float)
        last_array = np.asarray(timing_last, dtype=float)
        expected_array = np.asarray(timing_expected, dtype=float)
        slope, intercept, timing_r2 = _linear_timing_fit(period_array, last_array)
        valid_expected = expected_array[np.isfinite(expected_array)]

        effects = pd.DataFrame(combo_effects)
        median_effect = {
            column: _safe_median(effects[column]) if column in effects else math.nan
            for column in (
                "baseline",
                "baseline_norm",
                "periodicity",
                "periodicity_norm",
                "history_n0",
                "history_n0_norm",
                "history_n1",
                "history_n1_norm",
                "continuation",
                "continuation_norm",
                "embedded_peak",
                "embedded_peak_norm",
                "embedded_auc",
                "embedded_auc_norm",
            )
        }
        history_norm = (
            median_effect["history_n1_norm"]
            if np.isfinite(median_effect["history_n1_norm"])
            else median_effect["history_n0_norm"]
        )
        history_raw = (
            median_effect["history_n1"] if np.isfinite(median_effect["history_n1"]) else median_effect["history_n0"]
        )
        sustained = sub[sub["condition_type"].isin(("flash_level_only", "last_flash_then_hold_flash"))]
        tonic_gain_norm = (
            float(sustained["peak_above_steady_baseline_normalized"].max()) if not sustained.empty else 0.0
        )
        continuation_rows = sub[(sub["condition_type"] == "periodic_continuation") & (sub["n_flashes"] == target_n)]
        continuation_peak_norm = (
            float(continuation_rows["peak_above_steady_baseline_normalized"].median())
            if not continuation_rows.empty
            else math.nan
        )
        positive_effect_count = int(
            np.sum(periodic_all["peak_above_steady_baseline_normalized"].to_numpy(dtype=float) > 0)
        )
        combo_periodicity = effects.get("periodicity_norm", pd.Series(dtype=float)).to_numpy(dtype=float)
        amplitude_consistency = (
            float(np.mean(combo_periodicity[np.isfinite(combo_periodicity)] > 0))
            if np.isfinite(combo_periodicity).any()
            else 0.0
        )
        finite_all = bool(sub["finite_data"].all())
        record = {
            **mapping,
            "target_n_flashes_resolved": target_n,
            "response_scale_raw": _safe_median(sub["response_scale_raw"]),
            "response_scale": _safe_median(sub["response_scale"]),
            "baseline_gain": median_effect["baseline"],
            "baseline_gain_normalized": median_effect["baseline_norm"],
            "periodicity_gain": median_effect["periodicity"],
            "periodicity_gain_normalized": median_effect["periodicity_norm"],
            "history_gain": history_raw,
            "history_gain_normalized": history_norm,
            "history_gain_vs_n0": median_effect["history_n0"],
            "history_gain_vs_n0_normalized": median_effect["history_n0_norm"],
            "history_gain_vs_n1": median_effect["history_n1"],
            "history_gain_vs_n1_normalized": median_effect["history_n1_norm"],
            "continuation_contrast": median_effect["continuation"],
            "continuation_contrast_normalized": median_effect["continuation_norm"],
            "embedded_peak_contrast": median_effect["embedded_peak"],
            "embedded_peak_contrast_normalized": median_effect["embedded_peak_norm"],
            "embedded_auc_contrast": median_effect["embedded_auc"],
            "embedded_auc_contrast_normalized": median_effect["embedded_auc_norm"],
            "spearman_rho_peak_vs_n": _safe_median(rho_values),
            "monotonic_fraction": _safe_median(monotonic_values),
            "timing_slope_from_last": slope,
            "timing_intercept_from_last": intercept,
            "timing_r_squared": timing_r2,
            "expected_latency_mean_frames": (float(np.mean(valid_expected)) if valid_expected.size else math.nan),
            "expected_latency_std_frames": (float(np.std(valid_expected)) if valid_expected.size else math.nan),
            "valid_timing_periods": int(np.isfinite(last_array).sum()),
            "positive_effect_count": positive_effect_count,
            "amplitude_period_consistency": amplitude_consistency,
            "tonic_control_gain_normalized": tonic_gain_norm,
            "continuation_peak_normalized": continuation_peak_norm,
            "finite_all_conditions": finite_all,
        }
        records.append(record)

    metrics = pd.DataFrame.from_records(records)
    return classify_and_rank(metrics, config.thresholds)


def classify_and_rank(metrics: pd.DataFrame, thresholds: AnalysisThresholds) -> pd.DataFrame:
    """Apply transparent candidate rules and calculate a continuous score."""

    result = metrics.copy()
    finite = result["finite_all_conditions"].fillna(False)
    end_candidate = finite & (result["baseline_gain_normalized"] >= thresholds.min_baseline_gain)
    periodic_candidate = (
        end_candidate
        & (result["periodicity_gain_normalized"] >= thresholds.min_periodicity_gain)
        & (result["history_gain_normalized"] >= thresholds.min_history_gain)
        & (result["spearman_rho_peak_vs_n"] >= thresholds.min_spearman_rho)
        & (result["positive_effect_count"] > 1)
    )
    timing_consistent = (result["timing_r_squared"] >= thresholds.min_timing_r2) | (
        result["expected_latency_std_frames"] <= thresholds.max_expected_latency_std
    )
    predictive_candidate = (
        periodic_candidate
        & (result["valid_timing_periods"] >= 3)
        & result["timing_slope_from_last"].between(
            thresholds.timing_slope_min, thresholds.timing_slope_max, inclusive="both"
        )
        & timing_consistent.fillna(False)
    )
    result["end_response_candidate"] = end_candidate
    result["periodic_osr_candidate"] = periodic_candidate
    result["predictive_osr_candidate"] = predictive_candidate
    result["candidate_class"] = np.select(
        [predictive_candidate, periodic_candidate, end_candidate],
        ["predictive_osr_candidate", "periodic_osr_candidate", "end_response_candidate"],
        default="rejected",
    )

    baseline_component = np.clip(result["baseline_gain_normalized"].fillna(-1), -1, 2)
    periodicity_component = np.clip(result["periodicity_gain_normalized"].fillna(-1), -1, 2)
    history_component = np.clip(result["history_gain_normalized"].fillna(-1), -1, 2)
    rho_component = np.clip(result["spearman_rho_peak_vs_n"].fillna(-1), 0, 1)
    slope_closeness = np.clip(1.0 - np.abs(result["timing_slope_from_last"].fillna(3) - 1.0), 0, 1)
    r2_component = np.clip(result["timing_r_squared"].fillna(0), 0, 1)
    timing_component = slope_closeness * r2_component
    embedded_component = np.clip(result["embedded_peak_contrast_normalized"].fillna(0), 0, 2)
    consistency_component = np.clip(result["amplitude_period_consistency"].fillna(0), 0, 1)
    tonic_excess = np.clip(
        result["tonic_control_gain_normalized"].fillna(0) - result["periodicity_gain_normalized"].fillna(0),
        0,
        2,
    )
    result["composite_score"] = (
        0.20 * baseline_component
        + 0.25 * periodicity_component
        + 0.20 * history_component
        + 0.10 * rho_component
        + 0.15 * timing_component
        + 0.07 * embedded_component
        + 0.03 * consistency_component
        - 0.15 * tonic_excess
    )
    result.loc[~finite, "composite_score"] = -np.inf

    rejection_reasons = []
    for _, row in result.iterrows():
        reasons = []
        if not bool(row["finite_all_conditions"]):
            reasons.append("nonfinite_response")
        if not row["baseline_gain_normalized"] >= thresholds.min_baseline_gain:
            reasons.append("baseline_gain_below_threshold")
        if not row["periodicity_gain_normalized"] >= thresholds.min_periodicity_gain:
            reasons.append("periodicity_gain_below_threshold_or_unavailable")
        if not row["history_gain_normalized"] >= thresholds.min_history_gain:
            reasons.append("history_gain_below_threshold")
        if not row["spearman_rho_peak_vs_n"] >= thresholds.min_spearman_rho:
            reasons.append("n_dependence_below_threshold")
        if row["positive_effect_count"] <= 1:
            reasons.append("insufficient_positive_sequence_lengths")
        if row["valid_timing_periods"] < 3:
            reasons.append("fewer_than_three_timing_periods")
        elif not thresholds.timing_slope_min <= row["timing_slope_from_last"] <= thresholds.timing_slope_max:
            reasons.append("timing_slope_outside_range")
        elif not bool(timing_consistent.loc[row.name]):
            reasons.append("timing_inconsistent")
        rejection_reasons.append(";".join(reasons))
    result["rejection_reasons"] = rejection_reasons
    return result.sort_values("composite_score", ascending=False, kind="stable").reset_index(drop=True)


def threshold_sensitivity(
    metrics: pd.DataFrame,
    thresholds: AnalysisThresholds,
) -> pd.DataFrame:
    """Count candidates at 0.5x, 1x, and 1.5x magnitude/rho thresholds."""

    records = []
    for multiplier in (0.5, 1.0, 1.5):
        varied = AnalysisThresholds(
            min_baseline_gain=thresholds.min_baseline_gain * multiplier,
            min_periodicity_gain=thresholds.min_periodicity_gain * multiplier,
            min_history_gain=thresholds.min_history_gain * multiplier,
            min_spearman_rho=min(1.0, thresholds.min_spearman_rho * multiplier),
            timing_slope_min=thresholds.timing_slope_min,
            timing_slope_max=thresholds.timing_slope_max,
            min_timing_r2=min(1.0, thresholds.min_timing_r2 * multiplier),
            max_expected_latency_std=thresholds.max_expected_latency_std,
        )
        classified = classify_and_rank(metrics, varied)
        records.append(
            {
                "threshold_multiplier": multiplier,
                "end_response_candidates": int(classified["end_response_candidate"].sum()),
                "periodic_osr_candidates": int(classified["periodic_osr_candidate"].sum()),
                "predictive_osr_candidates": int(classified["predictive_osr_candidate"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _select_neurons_for_plots(neuron_metrics: pd.DataFrame, top_k: int) -> list[int]:
    if top_k <= 0 or neuron_metrics.empty:
        return []
    selected: list[int] = []

    def add_rows(rows: pd.DataFrame, count: int = 1) -> None:
        for value in rows["global_neuron_index"].head(count):
            index = int(value)
            if index not in selected and len(selected) < top_k:
                selected.append(index)

    add_rows(neuron_metrics[neuron_metrics["predictive_osr_candidate"]], 3)
    add_rows(neuron_metrics[neuron_metrics["periodic_osr_candidate"]], 2)
    add_rows(neuron_metrics.sort_values("periodicity_gain_normalized", ascending=False), 1)
    add_rows(neuron_metrics.sort_values("history_gain_normalized", ascending=False), 1)
    add_rows(
        neuron_metrics[
            neuron_metrics["end_response_candidate"] & ~neuron_metrics["periodic_osr_candidate"]
        ].sort_values("baseline_gain_normalized", ascending=False),
        1,
    )
    add_rows(
        neuron_metrics[~neuron_metrics["predictive_osr_candidate"]].sort_values(
            "continuation_peak_normalized", ascending=False
        ),
        1,
    )
    add_rows(neuron_metrics.sort_values("tonic_control_gain_normalized", ascending=False), 1)
    add_rows(neuron_metrics, top_k)
    return selected[:top_k]


def plot_neuron_diagnostic(
    *,
    neuron_index: int,
    neuron_metrics: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    conditions: Sequence[GeneratedCondition],
    responses: np.ndarray,
    timing: TimingResolution,
    output_path: Path,
) -> None:
    """Create a four-panel stimulus/response, history, timing, and control figure."""

    summary = neuron_metrics[neuron_metrics["global_neuron_index"] == neuron_index].iloc[0]
    per_condition = condition_metrics[condition_metrics["global_neuron_index"] == neuron_index]
    target_n = int(summary["target_n_flashes_resolved"])
    target_rows = per_condition[
        (per_condition["condition_type"] == "periodic_omission_terminal") & (per_condition["n_flashes"] == target_n)
    ].sort_values(["period_frames", "amplitude", "polarity"])
    if target_rows.empty:
        return
    primary = target_rows.iloc[0]
    period = int(primary["period_frames"])
    amplitude = float(primary["amplitude"])
    polarity = str(primary["polarity"])
    match = (
        (per_condition["period_frames"] == period)
        & (per_condition["amplitude"] == amplitude)
        & (per_condition["polarity"] == polarity)
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_trace, ax_history, ax_timing, ax_controls = axes.flat
    response_time = (np.arange(responses.shape[1]) + timing.cut_frames - timing.expected_frame) / timing.fps
    stimulus_time = (np.arange(timing.time_steps) - timing.expected_frame) / timing.fps
    trace_styles = [
        ("periodic_omission_terminal", "periodic omission", "C0"),
        ("periodic_continuation", "continuation", "C1"),
        ("steady_baseline", "steady baseline", "0.45"),
        ("embedded_omission", "embedded omission", "C2"),
        ("embedded_continuation", "embedded continuation", "C3"),
    ]
    for condition_type, label, color in trace_styles:
        rows = per_condition[
            match
            & (per_condition["condition_type"] == condition_type)
            & ((per_condition["n_flashes"] == target_n) | (per_condition["condition_type"] == "steady_baseline"))
        ]
        if rows.empty:
            continue
        condition_indices = rows["condition_index"].astype(int).to_numpy()
        response = np.median(responses[condition_indices, :, neuron_index], axis=0)
        ax_trace.plot(response_time, response, label=label, color=color, linewidth=1.5)
    aperiodic = per_condition[
        match & (per_condition["condition_type"] == "aperiodic_stop") & (per_condition["n_flashes"] == target_n)
    ]
    if not aperiodic.empty:
        condition_indices = aperiodic["condition_index"].astype(int).to_numpy()
        response = np.median(responses[condition_indices, :, neuron_index], axis=0)
        ax_trace.plot(response_time, response, label="aperiodic-stop median", color="C4", linewidth=1.5)
    primary_condition = conditions[int(primary["condition_index"])]
    stimulus_axis = ax_trace.twinx()
    for channel, channel_trace in enumerate(primary_condition.trace):
        stimulus_axis.step(
            stimulus_time,
            channel_trace,
            where="post",
            linewidth=0.8,
            alpha=0.35,
            label=f"stim ch{channel}",
        )
    stimulus_axis.set_ylabel("stimulus (model units)", color="0.35")
    ax_trace.axvline(0, color="k", linestyle="--", linewidth=1)
    last_real = primary_condition.metadata.last_real_flash_frame
    if last_real is not None:
        ax_trace.axvline((last_real - timing.expected_frame) / timing.fps, color="0.3", linestyle=":")
    ax_trace.axvspan(
        timing.osr_start_offset / timing.fps,
        timing.osr_stop_offset / timing.fps,
        color="C0",
        alpha=0.08,
    )
    ax_trace.set_xlabel("time from expected flash (s)")
    ax_trace.set_ylabel("model response")
    ax_trace.set_title("A. Aligned stimulus and response")
    ax_trace.legend(fontsize=7, ncols=2)

    history_rows = per_condition[
        match
        & per_condition["condition_type"].isin(
            ("periodic_omission_terminal", "periodic_continuation", "aperiodic_stop")
        )
    ]
    for condition_type, label, color in (
        ("periodic_omission_terminal", "periodic omission", "C0"),
        ("aperiodic_stop", "aperiodic stop", "C4"),
        ("periodic_continuation", "continuation", "C1"),
    ):
        rows = history_rows[history_rows["condition_type"] == condition_type]
        if rows.empty:
            continue
        grouped = rows.groupby("n_flashes")["peak_above_steady_baseline_normalized"].median()
        ax_history.plot(grouped.index, grouped.values, marker="o", label=label, color=color)
    ax_history.axhline(0, color="0.5", linewidth=0.8)
    ax_history.set_xlabel("preceding flashes")
    ax_history.set_ylabel("normalized OSR-window peak")
    ax_history.set_title("B. History dependence")
    ax_history.legend(fontsize=8)

    timing_rows = (
        per_condition[
            (per_condition["condition_type"] == "periodic_omission_terminal") & (per_condition["n_flashes"] == target_n)
        ]
        .groupby("period_frames", as_index=False)["peak_latency_last_flash_frames"]
        .median()
    )
    ax_timing.scatter(
        timing_rows["period_frames"],
        timing_rows["peak_latency_last_flash_frames"],
        color="C0",
        label="observed",
    )
    if len(timing_rows) >= 2 and np.isfinite(summary["timing_slope_from_last"]):
        x = np.linspace(timing_rows["period_frames"].min(), timing_rows["period_frames"].max(), 100)
        y = summary["timing_intercept_from_last"] + summary["timing_slope_from_last"] * x
        ax_timing.plot(x, y, color="C0", label=f"fit slope={summary['timing_slope_from_last']:.2f}")
        reference_intercept = float(
            np.nanmean(timing_rows["peak_latency_last_flash_frames"] - timing_rows["period_frames"])
        )
        ax_timing.plot(x, reference_intercept + x, "--", color="0.4", label="slope 1")
    ax_timing.set_xlabel("period (frames)")
    ax_timing.set_ylabel("peak latency from last flash (frames)")
    ax_timing.set_title("C. Predictive timing")
    ax_timing.legend(fontsize=8)

    gain_labels = ["baseline", "periodicity", "history", "embedded", "tonic"]
    gains = [
        summary["baseline_gain_normalized"],
        summary["periodicity_gain_normalized"],
        summary["history_gain_normalized"],
        summary["embedded_peak_contrast_normalized"],
        summary["tonic_control_gain_normalized"],
    ]
    colors = ["C0", "C4", "C2", "C3", "0.5"]
    ax_controls.bar(gain_labels, np.nan_to_num(gains, nan=0.0), color=colors)
    ax_controls.axhline(0, color="k", linewidth=0.8)
    ax_controls.tick_params(axis="x", rotation=25)
    ax_controls.set_ylabel("normalized effect")
    ax_controls.set_title(
        f"D. Controls and classification\n{summary['candidate_class']}; score={summary['composite_score']:.3f}"
    )

    group_label = summary["group_name"] if pd.notna(summary["group_name"]) else "unknown group"
    fig.suptitle(
        f"Neuron {neuron_index} | {summary['session_key']} local {summary['local_neuron_index']} | "
        f"{group_label}\n{timing.fps:g} Hz, T={period}, amplitude={amplitude:g} ({polarity})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_population_summary(
    *,
    neuron_metrics: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create the required multi-panel population summary and candidate-count figure."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax_scatter, ax_rank, ax_heatmap, ax_slopes, ax_sessions, ax_sensitivity = axes.flat
    class_colors = {
        "rejected": "0.75",
        "end_response_candidate": "C1",
        "periodic_osr_candidate": "C2",
        "predictive_osr_candidate": "C3",
    }
    for candidate_class, rows in neuron_metrics.groupby("candidate_class"):
        ax_scatter.scatter(
            rows["periodicity_gain_normalized"],
            rows["history_gain_normalized"],
            s=12,
            alpha=0.65,
            label=candidate_class,
            color=class_colors.get(candidate_class, "C0"),
        )
    ax_scatter.axhline(0, color="0.5", linewidth=0.8)
    ax_scatter.axvline(0, color="0.5", linewidth=0.8)
    ax_scatter.set_xlabel("normalized periodicity gain")
    ax_scatter.set_ylabel("normalized history gain")
    ax_scatter.set_title("Periodicity versus history")
    ax_scatter.legend(fontsize=7)

    ranked = neuron_metrics["composite_score"].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    ax_rank.plot(np.arange(len(ranked)), ranked.to_numpy(), color="C0")
    ax_rank.set_xlabel("neuron rank")
    ax_rank.set_ylabel("composite OSR score")
    ax_rank.set_title("Composite ranking")

    top_neurons = neuron_metrics.head(min(20, len(neuron_metrics)))["global_neuron_index"].astype(int)
    heat_rows = condition_metrics[
        condition_metrics["global_neuron_index"].isin(top_neurons)
        & (condition_metrics["condition_type"] == "periodic_omission_terminal")
    ].copy()
    heat_rows["condition_label"] = (
        "T" + heat_rows["period_frames"].astype(str) + "/n" + heat_rows["n_flashes"].astype(str)
    )
    if not heat_rows.empty:
        pivot = heat_rows.pivot_table(
            index="global_neuron_index",
            columns="condition_label",
            values="peak_above_steady_baseline_normalized",
            aggfunc="median",
        ).reindex(top_neurons)
        image = ax_heatmap.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
        ax_heatmap.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=90, fontsize=6)
        ax_heatmap.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=6)
        fig.colorbar(image, ax=ax_heatmap, fraction=0.046)
    ax_heatmap.set_title("Top-neuron periodic omission heatmap")
    ax_heatmap.set_xlabel("period / preceding flashes")
    ax_heatmap.set_ylabel("global neuron")

    slopes = neuron_metrics["timing_slope_from_last"].replace([np.inf, -np.inf], np.nan).dropna()
    ax_slopes.hist(slopes, bins=30, color="C0", alpha=0.8)
    ax_slopes.axvline(1, color="k", linestyle="--", linewidth=1)
    ax_slopes.set_xlabel("latency-from-last slope")
    ax_slopes.set_ylabel("neurons")
    ax_slopes.set_title("Predictive timing slopes")

    candidate_rows = neuron_metrics[neuron_metrics["candidate_class"] != "rejected"]
    counts = candidate_rows.groupby("session_key").size().sort_values(ascending=False).head(15)
    ax_sessions.barh(counts.index[::-1], counts.values[::-1], color="C2")
    ax_sessions.tick_params(axis="y", labelsize=6)
    ax_sessions.set_xlabel("candidate neurons")
    ax_sessions.set_title("Candidates by session (top 15)")

    x = sensitivity["threshold_multiplier"]
    ax_sensitivity.plot(x, sensitivity["end_response_candidates"], marker="o", label="end")
    ax_sensitivity.plot(x, sensitivity["periodic_osr_candidates"], marker="o", label="periodic")
    ax_sensitivity.plot(x, sensitivity["predictive_osr_candidates"], marker="o", label="predictive")
    ax_sensitivity.set_xlabel("threshold multiplier")
    ax_sensitivity.set_ylabel("candidate count")
    ax_sensitivity.set_title("Threshold sensitivity")
    ax_sensitivity.legend(fontsize=8)

    fig.suptitle("OpenRetina omitted-stimulus-response screen", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "summary.pdf", bbox_inches="tight")
    plt.close(fig)

    if "group_name" in neuron_metrics and neuron_metrics["group_name"].notna().any():
        fig, (ax_session, ax_group) = plt.subplots(1, 2, figsize=(14, 6))
        session_counts = candidate_rows.groupby("session_key").size().sort_values(ascending=False).head(20)
        group_counts = candidate_rows.groupby("group_name").size().sort_values(ascending=False).head(20)
        ax_session.barh(session_counts.index[::-1], session_counts.values[::-1], color="C2")
        ax_group.barh(group_counts.index[::-1], group_counts.values[::-1], color="C3")
        ax_session.tick_params(axis="y", labelsize=6)
        ax_group.tick_params(axis="y", labelsize=7)
        ax_session.set_title("Candidates by session (top 20)")
        ax_group.set_title("Candidates by cell-type group (top 20)")
        ax_session.set_xlabel("count")
        ax_group.set_xlabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / "candidate_counts.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert metadata to compact JSON without serializing large tensors/masks."""

    if depth > 4:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), depth=depth + 1)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return _array_summary(array)
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1) for key, item in value.items() if str(key) not in {"roi_mask"}
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 30:
            return {"length": len(value), "sample": [_json_safe(item, depth=depth + 1) for item in value[:5]]}
        return [_json_safe(item, depth=depth + 1) for item in value]
    return repr(value)


def _array_summary(array: np.ndarray) -> Any:
    if array.size <= 30:
        return _json_safe(array.tolist(), depth=1)
    numeric = np.issubdtype(array.dtype, np.number)
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if numeric and array.size:
        finite = np.asarray(array, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            result.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()))
    return result


def _package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        versions["openretina"] = importlib.metadata.version("openretina")
        versions["openretina_distribution_candidates"] = [
            {
                "version": distribution.version,
                "metadata_path": str(getattr(distribution, "_path", distribution.locate_file(""))),
            }
            for distribution in importlib.metadata.distributions(name="openretina")
        ]
    except importlib.metadata.PackageNotFoundError:
        versions["openretina"] = "not-installed"
        versions["openretina_distribution_candidates"] = []
    versions["openretina_module_path"] = str(Path(openretina.__file__).resolve())
    pyproject_path = Path(openretina.__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject_path.exists():
        version_match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject_path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if version_match:
            versions["openretina_source_pyproject_version"] = version_match.group(1)
    return versions


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory {path} is not empty; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots" / "neurons").mkdir(parents=True, exist_ok=True)


def _configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _config_to_json(config: ResolvedRunConfig, timing: TimingResolution | None = None) -> dict[str, Any]:
    record = asdict(config)
    record["output_dir"] = str(config.output_dir)
    if timing is not None:
        record["resolved_timing"] = asdict(timing)
    record["package_versions"] = _package_versions()
    return _json_safe(record)


def _output_time_probe(model: torch.nn.Module, device: str, input_shape: tuple[int, int, int]) -> Callable[[int], int]:
    def probe(time_steps: int) -> int:
        channels, height, width = input_shape
        with torch.inference_mode():
            output = forward_all_readouts(
                model,
                torch.zeros((1, channels, time_steps, height, width), device=device),
            )
        return int(output.shape[1])

    return probe


def run_analysis(
    args: argparse.Namespace,
    *,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Run the complete OSR screen and return a compact result summary."""

    preliminary_config = resolve_cli_config(args)
    _prepare_output_directory(preliminary_config.output_dir, preliminary_config.overwrite)
    _configure_logging(preliminary_config.output_dir)
    config = preliminary_config
    LOGGER.info("Loading model %s on %s", config.model, config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.num_threads)
    if model is None:
        model = load_core_readout_model(config.model, config.device)
    model = model.to(config.device)
    model.eval()
    config = resolve_cli_config(args, getattr(model, "data_info", {}))

    probe_input_shape, probe_output_shape, inferred_cut = probe_model(model, device=config.device, time_steps=50)
    input_shape = tuple(int(value) for value in probe_input_shape[1:2] + probe_input_shape[3:5])
    channels, height, width = input_shape
    LOGGER.info("50-frame input shape: %s", probe_input_shape)
    LOGGER.info("50-frame output shape: %s", probe_output_shape)
    LOGGER.info("Inferred temporal cut: %d frames", inferred_cut)
    LOGGER.info("Raw stim_mean=%r stim_std=%r", model.data_info.get("stim_mean"), model.data_info.get("stim_std"))

    neuron_mapping = build_neuron_mapping(model)
    if probe_output_shape[-1] != len(neuron_mapping):
        raise ValueError(
            f"Probe returned {probe_output_shape[-1]} neurons but explicit mapping contains {len(neuron_mapping)}"
        )
    value_resolution = resolve_value_space(
        requested=config.value_space,
        model_name=config.model,
        data_info=model.data_info,
        input_shape=input_shape,
        baseline_values=config.baseline_values,
    )
    timing = resolve_timing(
        fps=config.fps,
        cut_frames=inferred_cut,
        output_time_for_requested_input=_output_time_probe(model, config.device, input_shape),
        periods=config.period_frames,
        n_flashes=config.n_flashes,
        flash_frames=config.flash_frames,
        osr_window_ms=config.osr_window_ms,
        requested_time_steps=config.time_steps,
        embedded_post_flashes=config.embedded_post_flashes,
        include_embedded=config.include_embedded_omission,
        post_omission_frames=config.post_omission_frames,
    )
    LOGGER.info("Value-space strategy: %s", value_resolution.strategy)
    LOGGER.info(
        "Expected event frame=%d, output frame=%d, time_steps=%d, OSR offsets=%d..%d",
        timing.expected_frame,
        timing.expected_frame - timing.cut_frames,
        timing.time_steps,
        timing.osr_start_offset,
        timing.osr_stop_offset,
    )
    conditions = generate_conditions(config, timing, value_resolution, input_shape)
    stats = stimulus_channel_statistics(conditions)
    for record in stats:
        LOGGER.info(
            "Stimulus channel %d: min=%.5g max=%.5g mean=%.5g std=%.5g",
            record["channel"],
            record["min"],
            record["max"],
            record["mean"],
            record["std"],
        )
    for message in check_documented_range(stats, value_resolution):
        LOGGER.warning(message)
    probe_shapes = validate_baseline_and_random_probes(
        model,
        device=config.device,
        input_shape=input_shape,
        baseline=value_resolution.baseline_model,
        time_steps=timing.time_steps,
    )
    LOGGER.info("Finite baseline/random probe output shapes: %s", probe_shapes)

    condition_records = [condition.metadata.to_record(config.fps, inferred_cut) for condition in conditions]
    pd.DataFrame.from_records(condition_records).to_csv(config.output_dir / "stimulus_conditions.csv", index=False)
    np.savez_compressed(
        config.output_dir / "stimulus_traces.npz",
        condition_ids=np.asarray([condition.metadata.condition_id for condition in conditions]),
        traces=np.stack([condition.trace for condition in conditions]).astype(np.float32),
    )
    responses = infer_conditions(
        model,
        conditions,
        input_shape=input_shape,
        device=config.device,
        batch_size=config.batch_size,
    )
    if responses.shape[1] != timing.output_time_steps:
        raise ValueError(
            f"Condition output time {responses.shape[1]} differs from timing probe {timing.output_time_steps}"
        )
    if config.save_raw_responses:
        np.savez_compressed(
            config.output_dir / "raw_responses.npz",
            condition_ids=np.asarray([condition.metadata.condition_id for condition in conditions]),
            responses=responses,
        )

    condition_metrics, _ = calculate_condition_metrics(
        conditions=conditions,
        responses=responses,
        timing=timing,
        neuron_mapping=neuron_mapping,
    )
    condition_metrics.to_csv(config.output_dir / "condition_metrics.csv", index=False)
    neuron_metrics = summarize_neurons(
        condition_metrics=condition_metrics,
        neuron_mapping=neuron_mapping,
        config=config,
    )
    neuron_metrics.to_csv(config.output_dir / "neuron_metrics.csv", index=False)
    candidates = neuron_metrics[
        neuron_metrics["end_response_candidate"]
        | neuron_metrics["periodic_osr_candidate"]
        | neuron_metrics["predictive_osr_candidate"]
    ]
    candidates.to_csv(config.output_dir / "osr_candidates.csv", index=False)
    sensitivity = threshold_sensitivity(neuron_metrics, config.thresholds)
    sensitivity.to_csv(config.output_dir / "threshold_sensitivity.csv", index=False)

    plot_population_summary(
        neuron_metrics=neuron_metrics,
        condition_metrics=condition_metrics,
        sensitivity=sensitivity,
        output_dir=config.output_dir / "plots",
    )
    selected_neurons = _select_neurons_for_plots(neuron_metrics, config.top_k_plots)
    for neuron_index in selected_neurons:
        plot_neuron_diagnostic(
            neuron_index=neuron_index,
            neuron_metrics=neuron_metrics,
            condition_metrics=condition_metrics,
            conditions=conditions,
            responses=responses,
            timing=timing,
            output_path=config.output_dir / "plots" / "neurons" / f"neuron_{neuron_index:04d}.png",
        )

    session_summary: dict[str, dict[str, Any]] = {}
    offset = 0
    for session_key, rows in neuron_mapping.groupby("session_key", sort=False):
        count = len(rows)
        session_summary[session_key] = {
            "global_start": offset,
            "global_stop_exclusive": offset + count,
            "n_neurons": count,
            "eye": rows["eye"].iloc[0],
        }
        offset += count
    model_info = {
        "model": config.model,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "package_versions": _package_versions(),
        "probe_input_shape": probe_input_shape,
        "probe_output_shape": probe_output_shape,
        "input_shape_channels_height_width": input_shape,
        "metadata_model_cut_frames": model.data_info.get("model_cut_frames"),
        "inferred_cut_frames": inferred_cut,
        "raw_stim_mean": model.data_info.get("stim_mean"),
        "raw_stim_std": model.data_info.get("stim_std"),
        "value_space_resolution": asdict(value_resolution),
        "stimulus_channel_statistics": stats,
        "stimulus_rate_hz": model.data_info.get("stimulus_rate_hz"),
        "response_rate_hz": model.data_info.get("response_rate_hz"),
        "resolved_fps": config.fps,
        "sessions": session_summary,
        "n_neurons_total": len(neuron_mapping),
        "compact_data_info": _json_safe(model.data_info),
        "finite_probe_shapes": probe_shapes,
    }
    with (config.output_dir / "model_info.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(model_info), handle, indent=2, sort_keys=True)
    with (config.output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_to_json(config, timing), handle, indent=2, sort_keys=True)

    summary = {
        "output_dir": str(config.output_dir),
        "n_conditions": len(conditions),
        "n_neurons": len(neuron_mapping),
        "end_response_candidates": int(neuron_metrics["end_response_candidate"].sum()),
        "periodic_osr_candidates": int(neuron_metrics["periodic_osr_candidate"].sum()),
        "predictive_osr_candidates": int(neuron_metrics["predictive_osr_candidate"].sum()),
        "plotted_neurons": selected_neurons,
        "probe_input_shape": probe_input_shape,
        "probe_output_shape": probe_output_shape,
        "value_space_strategy": value_resolution.strategy,
    }
    LOGGER.info("Completed OSR screen: %s", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_analysis(args)
    except Exception:
        LOGGER.exception("OSR analysis failed")
        return 1
    return 0


__all__ = [
    "AnalysisThresholds",
    "GeneratedCondition",
    "ResolvedRunConfig",
    "StimulusCondition",
    "TimingResolution",
    "ValueSpaceResolution",
    "build_arg_parser",
    "build_neuron_mapping",
    "calculate_condition_metrics",
    "classify_and_rank",
    "forward_all_readouts",
    "generate_aperiodic_onsets",
    "generate_conditions",
    "main",
    "normalize_raw_values",
    "periodic_flash_onsets",
    "probe_model",
    "render_full_field_trace",
    "resolve_cli_config",
    "resolve_timing",
    "resolve_value_space",
    "run_analysis",
    "summarize_neurons",
]
