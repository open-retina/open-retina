from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openretina.insilico.osr_analysis import (
    AnalysisThresholds,
    TimingResolution,
    build_arg_parser,
    calculate_condition_metrics,
    classify_and_rank,
    generate_aperiodic_onsets,
    generate_conditions,
    normalize_raw_values,
    periodic_flash_onsets,
    render_full_field_trace,
    resolve_cli_config,
    resolve_timing,
    resolve_value_space,
    run_analysis,
)


class DummyOSRModel(torch.nn.Module):
    """Small deterministic model with a known two-frame temporal cut."""

    def __init__(self) -> None:
        super().__init__()
        self.data_info = {
            "input_shape": (2, 4, 3),
            "n_neurons_dict": {"session_a": 5},
            "stim_mean": float("nan"),
            "stim_std": float("nan"),
            "model_cut_frames": 2,
            "stimulus_rate_hz": 30,
            "response_rate_hz": 30,
            "sessions_kwargs": {
                "session_a": {
                    "eye": "left",
                    "roi_ids": np.arange(10, 15),
                    "group_assignment": np.arange(1, 6),
                }
            },
        }

    def stimulus_shape(self, time_steps: int, num_batches: int = 1) -> tuple[int, ...]:
        return num_batches, 2, time_steps, 4, 3

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        trace = stimulus.mean(dim=(1, 3, 4))[:, 2:]
        return torch.stack(
            [
                trace,
                -trace,
                trace.roll(1, dims=1),
                torch.ones_like(trace),
                torch.zeros_like(trace),
            ],
            dim=-1,
        )


def _quick_config(tmp_path: Path, *extra: str):
    args = build_arg_parser().parse_args(
        [
            "--model",
            "dummy",
            "--device",
            "cpu",
            "--quick",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )
    return args, resolve_cli_config(args, DummyOSRModel().data_info)


def test_periodic_and_embedded_stimulus_timing(tmp_path: Path) -> None:
    _, config = _quick_config(
        tmp_path,
        "--include-embedded-omission",
        "--include-sustained-controls",
        "--n-flashes",
        "0,1,4",
    )
    value_resolution = resolve_value_space(
        requested="auto",
        model_name="dummy",
        data_info=DummyOSRModel().data_info,
        input_shape=(2, 4, 3),
        baseline_values=None,
    )
    timing = resolve_timing(
        fps=30,
        cut_frames=2,
        output_time_for_requested_input=lambda time: time - 2,
        periods=config.period_frames,
        n_flashes=config.n_flashes,
        flash_frames=1,
        osr_window_ms=(33, 200),
        requested_time_steps=None,
        embedded_post_flashes=3,
        include_embedded=True,
    )
    conditions = generate_conditions(config, timing, value_resolution, (2, 4, 3))
    by_type = {}
    for generated in conditions:
        by_type.setdefault(generated.metadata.condition_type, []).append(generated)

    omission = next(item for item in by_type["periodic_omission_terminal"] if item.metadata.n_flashes == 4)
    continuation = next(item for item in by_type["periodic_continuation"] if item.metadata.n_flashes == 4)
    embedded = next(item for item in by_type["embedded_omission"] if item.metadata.n_flashes == 4)
    assert np.all(np.diff(omission.metadata.flash_onsets) == 3)
    assert timing.expected_frame not in omission.metadata.flash_onsets
    assert timing.expected_frame in continuation.metadata.flash_onsets
    assert timing.expected_frame not in embedded.metadata.flash_onsets
    assert timing.expected_frame + 3 in embedded.metadata.flash_onsets
    assert timing.expected_frame + 6 in embedded.metadata.flash_onsets
    assert timing.expected_frame + timing.osr_stop_offset - timing.cut_frames < timing.output_time_steps
    assert np.all(omission.trace[:, omission.metadata.flash_onsets[0]] == -0.5)
    assert np.all(omission.trace[:, omission.metadata.flash_onsets[0] + 1] == 0.0)


def test_render_full_field_trace_has_exact_duration_and_uniform_pixels() -> None:
    trace = render_full_field_trace(
        channels=2,
        time_steps=12,
        baseline=(0.0, 1.0),
        flash=(-0.5, 0.5),
        onsets=(3, 8),
        flash_frames=2,
    )
    full_field = np.broadcast_to(trace[:, :, None, None], (2, 12, 4, 3))
    assert np.all(full_field[0, 3:5] == -0.5)
    assert np.all(full_field[1, 3:5] == 0.5)
    assert np.all(full_field[:, 5:8] == np.array([0.0, 1.0])[:, None, None, None])


def test_aperiodic_control_is_duration_matched_and_deterministic() -> None:
    kwargs = {
        "expected_frame": 40,
        "n_flashes": 7,
        "period_frames": 3,
        "flash_frames": 1,
        "seed": 123,
    }
    aperiodic = generate_aperiodic_onsets(**kwargs)
    repeated = generate_aperiodic_onsets(**kwargs)
    periodic = periodic_flash_onsets(40, 7, 3)
    assert aperiodic is not None
    assert aperiodic == repeated
    assert len(aperiodic) == len(periodic)
    assert aperiodic[0] == periodic[0]
    assert aperiodic[-1] == periodic[-1]
    intervals = np.diff(aperiodic)
    assert intervals.sum() == np.diff(periodic).sum()
    assert not np.all(intervals == 3)
    assert np.all(intervals >= 2)
    assert (
        generate_aperiodic_onsets(
            expected_frame=20,
            n_flashes=2,
            period_frames=3,
            flash_frames=1,
            seed=1,
        )
        is None
    )
    assert (
        generate_aperiodic_onsets(
            expected_frame=20,
            n_flashes=4,
            period_frames=2,
            flash_frames=1,
            seed=1,
        )
        is None
    )

    periodic_trace = render_full_field_trace(
        channels=1,
        time_steps=50,
        baseline=(0.0,),
        flash=(-0.5,),
        onsets=periodic,
        flash_frames=1,
    )
    aperiodic_trace = render_full_field_trace(
        channels=1,
        time_steps=50,
        baseline=(0.0,),
        flash=(-0.5,),
        onsets=aperiodic,
        flash_frames=1,
    )
    assert np.array_equal(
        periodic_trace[:, periodic[-1] :],
        aperiodic_trace[:, aperiodic[-1] :],
    )


def test_scalar_and_per_channel_normalization() -> None:
    scalar = normalize_raw_values((3.0, 5.0), mean=1.0, std=2.0, channels=2)
    vector = normalize_raw_values((3.0, 5.0), mean=(1.0, 2.0), std=(2.0, 1.5), channels=2)
    np.testing.assert_allclose(scalar, (1.0, 2.0))
    np.testing.assert_allclose(vector, (1.0, 2.0))


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_normalization_std_raises(std: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        normalize_raw_values((1.0, 2.0), mean=0.0, std=std, channels=2)


def test_nan_metadata_auto_uses_finite_model_space_and_raw_unknown_raises() -> None:
    data_info = {"stim_mean": float("nan"), "stim_std": float("nan")}
    auto = resolve_value_space(
        requested="auto",
        model_name="unknown_model",
        data_info=data_info,
        input_shape=(3, 4, 4),
        baseline_values=None,
    )
    assert auto.strategy == "auto_model_space_nonfinite_or_missing_metadata"
    assert np.isfinite(auto.baseline_model).all()
    with pytest.raises(ValueError, match="Raw value-space"):
        resolve_value_space(
            requested="raw",
            model_name="unknown_model",
            data_info=data_info,
            input_shape=(3, 4, 4),
            baseline_values=(1.0, 1.0, 1.0),
        )


def test_channel_vectors_require_exact_dimension(tmp_path: Path) -> None:
    args, config = _quick_config(
        tmp_path,
        "--channel-mode",
        "custom",
        "--channel-vector",
        "1,0,1",
    )
    value_resolution = resolve_value_space(
        requested="model",
        model_name="dummy",
        data_info=DummyOSRModel().data_info,
        input_shape=(2, 4, 3),
        baseline_values=None,
    )
    timing = TimingResolution(
        fps=30,
        cut_frames=2,
        expected_frame=26,
        time_steps=40,
        output_time_steps=38,
        osr_start_offset=1,
        osr_stop_offset=6,
        osr_start_ms_resolved=1000 / 30,
        osr_stop_ms_resolved=200,
        pre_event_offsets=(-2, -1),
        minimum_time_steps=33,
    )
    with pytest.raises(ValueError, match="exactly 2 entries"):
        generate_conditions(config, timing, value_resolution, (2, 4, 3))
    assert args.channel_mode == "custom"


def test_temporal_alignment_maps_synthetic_peak_to_input_event(tmp_path: Path) -> None:
    _, config = _quick_config(tmp_path, "--n-flashes", "0,1")
    value_resolution = resolve_value_space(
        requested="model",
        model_name="dummy",
        data_info=DummyOSRModel().data_info,
        input_shape=(2, 4, 3),
        baseline_values=None,
    )
    timing = TimingResolution(
        fps=30,
        cut_frames=3,
        expected_frame=10,
        time_steps=20,
        output_time_steps=17,
        osr_start_offset=1,
        osr_stop_offset=4,
        osr_start_ms_resolved=1000 / 30,
        osr_stop_ms_resolved=4000 / 30,
        pre_event_offsets=(-2, -1),
        minimum_time_steps=15,
    )
    conditions = generate_conditions(config, timing, value_resolution, (2, 4, 3))
    responses = np.zeros((len(conditions), 17, 1), dtype=np.float32)
    terminal_index = next(
        index
        for index, condition in enumerate(conditions)
        if condition.metadata.condition_type == "periodic_omission_terminal"
    )
    peak_output_index = timing.expected_frame - timing.cut_frames + 2
    responses[terminal_index, peak_output_index, 0] = 5.0
    mapping = pd.DataFrame(
        [
            {
                "global_neuron_index": 0,
                "session_key": "session",
                "local_neuron_index": 0,
                "roi_id": None,
                "group_id": None,
                "group_name": None,
                "group_class": None,
                "eye": None,
            }
        ]
    )
    metrics, _ = calculate_condition_metrics(
        conditions=conditions,
        responses=responses,
        timing=timing,
        neuron_mapping=mapping,
    )
    terminal = metrics[metrics["condition_type"] == "periodic_omission_terminal"].iloc[0]
    assert terminal["peak_output_frame"] == peak_output_index
    assert terminal["peak_input_frame"] == timing.expected_frame + 2
    assert terminal["peak_latency_expected_frames"] == 2


def _classification_row(index: int, **overrides) -> dict:
    row = {
        "global_neuron_index": index,
        "session_key": "session",
        "local_neuron_index": index,
        "roi_id": index,
        "group_id": 1,
        "group_name": "OFF local, OS",
        "group_class": "OFF",
        "eye": "left",
        "finite_all_conditions": True,
        "baseline_gain_normalized": 0.2,
        "periodicity_gain_normalized": 0.2,
        "history_gain_normalized": 0.2,
        "spearman_rho_peak_vs_n": 0.8,
        "positive_effect_count": 3,
        "valid_timing_periods": 4,
        "timing_slope_from_last": 1.0,
        "timing_r_squared": 0.9,
        "expected_latency_std_frames": 0.2,
        "embedded_peak_contrast_normalized": 0.15,
        "amplitude_period_consistency": 1.0,
        "tonic_control_gain_normalized": 0.0,
    }
    row.update(overrides)
    return row


def test_classification_rejects_key_confounders() -> None:
    population = pd.DataFrame(
        [
            _classification_row(0),  # true predictive OSR
            _classification_row(1, periodicity_gain_normalized=0.0),  # generic end detector
            _classification_row(2, baseline_gain_normalized=0.02, tonic_control_gain_normalized=1.0),
            _classification_row(3, periodicity_gain_normalized=-0.2),  # real-flash/continuation responder
            _classification_row(4, timing_slope_from_last=0.0),  # history-dependent, nonpredictive
            _classification_row(5, finite_all_conditions=False),  # nonfinite
        ]
    )
    classified = classify_and_rank(population, AnalysisThresholds()).set_index("global_neuron_index")
    assert classified.loc[0, "predictive_osr_candidate"]
    assert classified.loc[1, "end_response_candidate"]
    assert not classified.loc[1, "periodic_osr_candidate"]
    assert classified.loc[4, "periodic_osr_candidate"]
    assert not classified.loc[4, "predictive_osr_candidate"]
    assert not classified.loc[2, "end_response_candidate"]
    assert not classified.loc[3, "periodic_osr_candidate"]
    assert not classified.loc[5, "end_response_candidate"]
    assert "nonfinite_response" in classified.loc[5, "rejection_reasons"]


def test_dummy_cli_pipeline_writes_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "dummy_run"
    args = build_arg_parser().parse_args(
        [
            "--model",
            "dummy",
            "--device",
            "cpu",
            "--quick",
            "--output-dir",
            str(output_dir),
            "--top-k-plots",
            "1",
        ]
    )
    summary = run_analysis(args, model=DummyOSRModel())
    expected = {
        "resolved_config.json",
        "model_info.json",
        "stimulus_conditions.csv",
        "condition_metrics.csv",
        "neuron_metrics.csv",
        "osr_candidates.csv",
        "plots/summary.png",
        "plots/summary.pdf",
        "run.log",
    }
    actual = {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    assert expected <= actual
    assert summary["n_neurons"] == 5
    with (output_dir / "model_info.json").open(encoding="utf-8") as handle:
        model_info = json.load(handle)
    assert model_info["inferred_cut_frames"] == 2
    assert model_info["value_space_resolution"]["strategy"].startswith("auto_model_space")
    assert list(model_info["sessions"]) == ["session_a"]
    assert model_info["sessions"]["session_a"]["global_stop_exclusive"] == 5
