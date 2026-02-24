"""Utilities for model evaluation and result aggregation."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openretina.data_io.base import DatasetStatistics


@dataclass
class EvaluationSummary:
    """Summary statistics from model evaluation, suitable for cross-model comparison.

    This dataclass holds all aggregate metrics computed during evaluation and provides
    methods for serialization (to_dict, to_json) and formatted console output (print_report).
    """

    # Model identification
    model_path: str
    model_tag: str
    exp_name: str
    species: str | None
    data_split: str

    # Model characteristics
    temporal_lag: int

    # Trial/repeat information
    n_test_repeats_min: int
    n_test_repeats_max: int
    n_test_repeats_avg: float

    # Neuron counts
    n_neurons_total: int
    n_neurons_filtered: int
    var_ratio_cutoff: float
    filtering_applied: bool

    # Aggregate metrics (on filtered neurons)
    corr_to_avg_mean: float
    corr_by_trial_mean: float
    corr_by_trial_std: float
    mse_to_avg_mean: float
    mse_by_trial_mean: float
    mse_by_trial_std: float
    feve_mean: float
    fev_mean: float
    poisson_loss_mean: float
    jackknife_mean: float

    # Dataset statistics (all default to 0/empty when compute_dataset_statistics is disabled)
    n_sessions: int = 0
    unique_train_frames: int = 0
    unique_val_frames: int = 0
    unique_train_val_frames: int = 0
    unique_test_frames: dict[str, int] = field(default_factory=dict)
    unique_train_transitions: int = 0
    unique_val_transitions: int = 0
    unique_test_transitions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df_filtered: pd.DataFrame,
        *,
        model_path: str,
        model_tag: str,
        exp_name: str,
        species: str | None,
        data_split: str,
        temporal_lag: int,
        n_trials_per_session: list[int],
        n_neurons_total: int,
        var_ratio_cutoff: float,
        filtering_applied: bool,
        dataset_stats: DatasetStatistics,
    ) -> "EvaluationSummary":
        """Build an EvaluationSummary from a filtered DataFrame and metadata.

        Args:
            df_filtered: DataFrame containing per-neuron results (after var_ratio filtering).
            model_path: Path or identifier for the model.
            model_tag: Human-readable tag for the model.
            exp_name: Name of the dataset/experiment.
            species: Species name (e.g., "mouse", "marmoset") or None.
            data_split: Data split used for evaluation (e.g., "test").
            temporal_lag: Temporal lag between responses and model predictions.
            n_trials_per_session: List of trial counts per session.
            n_neurons_total: Total number of neurons before filtering.
            var_ratio_cutoff: Variance ratio threshold used for filtering.
            filtering_applied: Whether var_ratio filtering was actually applied.
            dataset_stats: DatasetStatistics with frame counts.

        Returns:
            EvaluationSummary instance with all computed metrics.
        """
        # Compute per-trial aggregate metrics
        corr_by_trial_avgs: list[float] = []
        mse_by_trial_avgs: list[float] = []
        i = 0
        while f"corr_{i}" in df_filtered.columns:
            corr_by_trial_avgs.append(float(np.nanmean(df_filtered[f"corr_{i}"])))
            mse_by_trial_avgs.append(float(np.nanmean(df_filtered[f"mse_{i}"])))
            i += 1

        return cls(
            model_path=model_path,
            model_tag=model_tag,
            exp_name=exp_name,
            species=species,
            data_split=data_split,
            temporal_lag=temporal_lag,
            n_test_repeats_min=min(n_trials_per_session, default=0),
            n_test_repeats_max=max(n_trials_per_session, default=0),
            n_test_repeats_avg=float(np.mean(n_trials_per_session)) if n_trials_per_session else 0.0,
            n_neurons_total=n_neurons_total,
            n_neurons_filtered=len(df_filtered),
            var_ratio_cutoff=var_ratio_cutoff,
            filtering_applied=filtering_applied,
            corr_to_avg_mean=float(np.nanmean(df_filtered["corr_to_average"])),
            corr_by_trial_mean=float(np.nanmean(corr_by_trial_avgs)) if corr_by_trial_avgs else float("nan"),
            corr_by_trial_std=float(np.nanstd(corr_by_trial_avgs)) if corr_by_trial_avgs else float("nan"),
            mse_to_avg_mean=float(np.nanmean(df_filtered["mse_to_average"])),
            mse_by_trial_mean=float(np.nanmean(mse_by_trial_avgs)) if mse_by_trial_avgs else float("nan"),
            mse_by_trial_std=float(np.nanstd(mse_by_trial_avgs)) if mse_by_trial_avgs else float("nan"),
            feve_mean=float(np.nanmean(df_filtered["feve"])),
            fev_mean=float(np.nanmean(df_filtered["var_ratio"])),
            poisson_loss_mean=float(np.nanmean(df_filtered["poisson_loss_to_average"])),
            jackknife_mean=float(np.nanmean(df_filtered["jackknife"])),
            n_sessions=dataset_stats.n_sessions,
            unique_train_frames=dataset_stats.unique_train_frames,
            unique_val_frames=dataset_stats.unique_val_frames,
            unique_train_val_frames=dataset_stats.unique_train_val_frames,
            unique_test_frames=dataset_stats.unique_test_frames,
            unique_train_transitions=dataset_stats.unique_train_transitions,
            unique_val_transitions=dataset_stats.unique_val_transitions,
            unique_test_transitions=dataset_stats.unique_test_transitions,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary suitable for CSV row (flattens nested dicts)."""
        d = asdict(self)
        # Flatten unique_test_frames dict
        test_frames = d.pop("unique_test_frames")
        for test_name, frames in test_frames.items():
            d[f"unique_test_frames_{test_name}"] = frames
        # Flatten unique_test_transitions dict
        test_transitions = d.pop("unique_test_transitions")
        for test_name, transitions in test_transitions.items():
            d[f"unique_test_transitions_{test_name}"] = transitions
        return d

    def save_json(self, path: str | Path) -> None:
        """Save summary to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_report(self) -> None:
        """Print formatted evaluation report to stdout.

        This produces the same output format as the original eval.py printing logic.
        """
        can_filter = self.filtering_applied or self.n_neurons_filtered < self.n_neurons_total

        # Header
        print("=" * 80)
        print("Model Evaluation Results")
        print("=" * 80)
        print(f"Model path: {self.model_path}")
        print(f"Data split: {self.data_split}")
        print(f"Temporal lag: {self.temporal_lag}")

        # Dataset statistics (only shown when compute_dataset_statistics was enabled)
        has_dataset_stats = self.n_sessions > 0 or self.unique_train_val_frames > 0 or bool(self.unique_test_frames)
        if has_dataset_stats:
            print("-" * 80)
            print("Dataset Statistics (unique frames/frame transitions across sessions, from dataloaders):")
            print(f"  Sessions: {self.n_sessions}")
            if self.unique_train_frames > 0 and self.unique_val_frames > 0:
                print(f"  Training frames: {self.unique_train_frames:,}")
                print(f"  Training pairwise frame transitions: {self.unique_train_transitions:,}")
                print(f"  Validation frames: {self.unique_val_frames:,}")
                print(f"  Validation pairwise frame transitions: {self.unique_val_transitions:,}")
            else:
                print(f"  Training + validation frames: {self.unique_train_val_frames:,}")
            for test_name, test_frames in sorted(self.unique_test_frames.items()):
                test_trans = self.unique_test_transitions.get(test_name, 0)
                print(f"  Test frames ({test_name}): {test_frames:,} ({test_trans:,} transitions)")

        # Neuron counts
        print("-" * 80)
        print(f"Total neurons: {self.n_neurons_total}")
        n_neurons_excluded = self.n_neurons_total - self.n_neurons_filtered
        if can_filter:
            print(f"Variance ratio cutoff: {self.var_ratio_cutoff}")
            if self.filtering_applied:
                print(f"Neurons above var_ratio threshold (>={self.var_ratio_cutoff}): {self.n_neurons_filtered}")
                excluded_pct = n_neurons_excluded / self.n_neurons_total * 100
                print(f"Neurons excluded: {n_neurons_excluded} ({excluded_pct:.1f}%)")
            else:
                print("Note: var_ratio filtering not applied (no valid var_ratio values)")
        else:
            print(f"Variance ratio cutoff: {self.var_ratio_cutoff} (NOT APPLIED - no trial/repeats data available)")
            print("Note: var_ratio could not be computed (no trial/repeats data). All neurons included.")
        print("-" * 80)

        # Trial-averaged metrics header
        if self.filtering_applied and can_filter:
            print("\nTrial/repeats-averaged metrics (computed on neurons above var_ratio threshold):")
        else:
            print("\nTrial/repeats-averaged metrics (computed on all neurons - var_ratio filtering not applied):")

        # Number of trials/repeats
        if self.n_test_repeats_max <= 1:
            print("\n(Number of trials/repeats: N/A)")
        elif self.n_test_repeats_min == self.n_test_repeats_max:
            print(f"(Number of trials/repeats: {self.n_test_repeats_min} (constant across all sessions))")
        else:
            print(
                f"(Number of trials/repeats: min={self.n_test_repeats_min}, "
                f"max={self.n_test_repeats_max}, avg={self.n_test_repeats_avg:.1f})"
            )

        # Trial-averaged metrics
        print("-" * 80)
        metric_values = [
            ("Correlation", self.corr_to_avg_mean),
            ("MSE", self.mse_to_avg_mean),
            ("FEVe", self.feve_mean),
            ("FEV (expl. var ratio)", self.fev_mean),
            ("Poisson loss", self.poisson_loss_mean),
        ]
        for name, value in metric_values:
            print(f"  {name:30s}: {value:.4f}")

        # Per-trial metrics
        if self.filtering_applied and can_filter:
            print("\nPer-trial metrics (computed on neurons above var_ratio threshold):")
        else:
            print("\nPer-trial metrics (computed on all neurons - var_ratio filtering not applied):")
        print("-" * 80)

        if not np.isnan(self.corr_by_trial_mean):
            print(f"  {'Correlation':30s}: {self.corr_by_trial_mean:.4f} (+/-{self.corr_by_trial_std:.4f})")
        if not np.isnan(self.mse_by_trial_mean):
            print(f"  {'MSE':30s}: {self.mse_by_trial_mean:.4f} (+/-{self.mse_by_trial_std:.4f})")

        print("=" * 80)
