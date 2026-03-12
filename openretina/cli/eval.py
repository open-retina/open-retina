#!/usr/bin/env python3

import logging
import math
import os
from pathlib import Path

import hydra
import lightning.pytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from openretina.data_io.base import DatasetStatistics
from openretina.eval.metrics import MSE_numpy, correlation_numpy, explainable_vs_total_var, feve
from openretina.eval.oracles import oracle_corr_jackknife
from openretina.models.core_readout import load_core_readout_model
from openretina.modules.losses import PoissonLoss3d
from openretina.utils.eval_utils import EvaluationSummary, align_responses_to_model_output
from openretina.utils.frame_fingerprints import compute_dataloader_statistics
from openretina.utils.misc import reorder_like_a

log = logging.getLogger(__name__)
logging.captureWarnings(True)


def _get_session_metadata(
    session: str,
    neuron_data_dict: dict[str, object],
    model_data_info: dict[str, object],
) -> dict[str, object]:
    """Merge session metadata from the checkpoint and the loaded responses."""
    model_sessions_kwargs = model_data_info.get("sessions_kwargs", {})
    model_session_kwargs = {}
    if isinstance(model_sessions_kwargs, dict):
        model_session_kwargs = model_sessions_kwargs.get(session, {})

    response_session_kwargs = getattr(neuron_data_dict[session], "session_kwargs", {})

    merged_session_kwargs = {}
    if isinstance(model_session_kwargs, dict):
        merged_session_kwargs.update(model_session_kwargs)
    if isinstance(response_session_kwargs, dict):
        merged_session_kwargs.update(response_session_kwargs)
    return merged_session_kwargs


def _plot_group_response_traces(
    df_filtered: pandas.DataFrame,
    session_plot_data: dict[str, dict[str, np.ndarray]],
    group_metric_breakdown: list[dict[str, object]],
    output_path: str | os.PathLike[str],
) -> None:
    """Plot average group responses across all used neurons for each eligible cell group."""
    if not group_metric_breakdown:
        log.warning("No cell groups passed the minimum used-neuron threshold. Skipping group response plot.")
        return

    n_groups = len(group_metric_breakdown)
    n_cols = 2 if n_groups > 1 else 1
    n_rows = math.ceil(n_groups / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, group_summary in zip(axes_flat, group_metric_breakdown, strict=False):
        group_id = group_summary["group_assignment"]
        group_name = str(group_summary.get("group_name", group_id))
        group_neuron_count = int(group_summary["n_neurons_filtered"])

        trial_sums: np.ndarray | None = None
        trial_cell_counts: np.ndarray | None = None
        avg_response_sum: np.ndarray | None = None
        predicted_response_sum: np.ndarray | None = None
        total_group_neurons = 0
        time_len: int | None = None

        for session, session_df in df_filtered[df_filtered["group_assignment"] == group_id].groupby("session"):
            session_data = session_plot_data.get(session)
            if session_data is None:
                continue

            neuron_indices = session_df["neuron_relative_idx"].astype(int).to_numpy()
            if neuron_indices.size == 0:
                continue

            session_avg_responses = session_data["avg_responses"][:, neuron_indices]
            session_predicted_responses = session_data["model_responses"][:, neuron_indices]
            session_responses_by_trial = session_data["responses_by_trial"][:, :, neuron_indices]

            if time_len is None:
                time_len = session_avg_responses.shape[0]
                avg_response_sum = np.zeros(time_len, dtype=np.float64)
                predicted_response_sum = np.zeros(time_len, dtype=np.float64)
                trial_sums = np.zeros((time_len, session_responses_by_trial.shape[1]), dtype=np.float64)
                trial_cell_counts = np.zeros(session_responses_by_trial.shape[1], dtype=np.float64)
            elif session_avg_responses.shape[0] != time_len:
                log.warning(
                    "Skipping session %s for group %s because the aligned response length does not match (%s != %s).",
                    session,
                    group_id,
                    session_avg_responses.shape[0],
                    time_len,
                )
                continue

            assert avg_response_sum is not None
            assert predicted_response_sum is not None
            assert trial_sums is not None
            assert trial_cell_counts is not None

            n_trials_session = session_responses_by_trial.shape[1]
            if n_trials_session > trial_sums.shape[1]:
                expanded_trial_sums = np.zeros((time_len, n_trials_session), dtype=np.float64)
                expanded_trial_sums[:, : trial_sums.shape[1]] = trial_sums
                trial_sums = expanded_trial_sums

                expanded_trial_counts = np.zeros(n_trials_session, dtype=np.float64)
                expanded_trial_counts[: trial_cell_counts.shape[0]] = trial_cell_counts
                trial_cell_counts = expanded_trial_counts

            avg_response_sum += session_avg_responses.sum(axis=1)
            predicted_response_sum += session_predicted_responses.sum(axis=1)
            trial_sums[:, :n_trials_session] += session_responses_by_trial.sum(axis=2)
            trial_cell_counts[:n_trials_session] += neuron_indices.size
            total_group_neurons += neuron_indices.size

        if total_group_neurons == 0:
            ax.set_visible(False)
            continue

        assert avg_response_sum is not None
        assert predicted_response_sum is not None
        assert trial_sums is not None
        assert trial_cell_counts is not None

        avg_response = avg_response_sum / total_group_neurons
        predicted_response = predicted_response_sum / total_group_neurons
        valid_trial_mask = trial_cell_counts > 0
        trial_responses = trial_sums[:, valid_trial_mask] / trial_cell_counts[valid_trial_mask]

        for trial_idx in range(trial_responses.shape[1]):
            label = "Individual trials" if trial_idx == 0 else None
            ax.plot(trial_responses[:, trial_idx], color="0.75", linewidth=1.0, alpha=0.9, label=label)
        ax.plot(avg_response, color="black", linewidth=2.0, label="Average over trials")
        ax.plot(predicted_response, color="tab:red", linewidth=2.0, label="Predicted response")
        ax.set_title(f"{group_name} (group {group_id}, n={group_neuron_count})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Response")
        ax.grid(alpha=0.2, linewidth=0.5)

    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Group response plot saved to %s", output_path)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="karamanlis_2024_eval",
)
def main(cfg: DictConfig) -> float | None:
    score = evaluate_model(cfg)
    return score


def evaluate_model(cfg: DictConfig) -> float:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")
    if cfg.evaluation.model_path is None:
        raise ValueError("Please provide evaluation.model_path to define which model to test.")

    # Set cache folder
    os.environ["OPENRETINA_CACHE_DIRECTORY"] = cfg.paths.cache_dir

    # Load model
    model_path = cfg.evaluation.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_core_readout_model(model_path, device)
    model.eval()

    # Set matmul precision if defined
    if "matmul_precision" in cfg:
        hydra.utils.call(cfg.matmul_precision)

    movies_dict = hydra.utils.call(cfg.data_io.stimuli)
    neuron_data_dict = hydra.utils.call(cfg.data_io.responses)

    if cfg.check_stimuli_responses_match:
        for session, neuron_data in neuron_data_dict.items():
            neuron_data.check_matching_stimulus(movies_dict[session])

    dataloaders = hydra.utils.instantiate(
        cfg.dataloader,
        neuron_data_dictionary=neuron_data_dict,
        movies_dictionary=movies_dict,
    )

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    # Note: data_split can be either "train", "validation", or "test", or, if we have multiple test stimuli,
    # the name of the test stimulus (as it appears in the dataloaders)
    data_split = cfg.evaluation.get("data_split", "test")
    if data_split not in dataloaders:
        raise ValueError(f"Dataloader '{data_split}' not found. Available keys: {sorted(dataloaders.keys())}")

    dataloader_dict = dataloaders[data_split]
    results = []  # List of dictionaries, one per neuron
    poisson_loss = PoissonLoss3d(per_neuron=True)
    lag = -1
    n_trials_per_session = []  # Track number of trials/repeats per session
    session_plot_data: dict[str, dict[str, np.ndarray]] = {}

    for session, dl in tqdm(dataloader_dict.items(), desc="Evaluating sessions", unit="session"):
        dataset = dl.dataset

        with torch.no_grad():
            model_responses_torch_array = []
            targets_array = []
            for data_point in dl:
                model_resp_batch = model.forward(data_point[0].to(device), data_key=session)
                model_resp = model_resp_batch.flatten(0, 1)
                model_responses_torch_array.append(model_resp)
                targets_array.append(data_point[1].flatten(0, 1))
            model_responses_torch = torch.concat(model_responses_torch_array)
            targets = torch.concat(targets_array).to(device)
            poisson_loss_session = poisson_loss(model_responses_torch.unsqueeze(0), targets.unsqueeze(0))

        poisson_loss_values = poisson_loss_session.cpu().numpy()
        model_responses = model_responses_torch.cpu().numpy()

        avg_responses = dataset.responses.numpy()
        try:
            responses_by_trial = dataset.test_responses_by_trial.cpu().numpy()
            responses_by_trial = reorder_like_a(a=avg_responses, b=responses_by_trial)
            # rearrange to match feve function expectations
            responses_by_trial = rearrange(responses_by_trial, "trials time neurons -> time trials neurons")
        except RuntimeError as e:
            log.warning(
                f"Could not infer reordering of responses by trial for {session}: {e}. "
                f"Assuming they are ordered as `trials, neurons, time` on data loading.",
                exc_info=True,
            )
            responses_by_trial = rearrange(responses_by_trial, "trials neurons time -> time trials neurons")
        except Exception as e:
            log.warning(
                f"Could not retrieve responses by trial for session {session}: {e}. Using avg responses instead. "
                f"var_ratio will not be computed and var_ratio_cutoff filtering will be skipped for this session.",
                exc_info=True,
            )
            responses_by_trial = avg_responses[:, np.newaxis, :]

        avg_responses, responses_by_trial, lag = align_responses_to_model_output(
            targets=targets,
            model_responses=model_responses,
            avg_responses=avg_responses,
            responses_by_trial=responses_by_trial,
            dataset=dataset,
            lag=lag,
        )

        session_plot_data[session] = {
            "avg_responses": avg_responses,
            "responses_by_trial": responses_by_trial,
            "model_responses": model_responses,
        }

        n_neurons_session = avg_responses.shape[1]

        if model_responses.shape != avg_responses.shape:
            raise ValueError(f"Inconsistent Shapes: {model_responses.shape=}, {avg_responses.shape}, {lag=}")
        if responses_by_trial[:, 0, :].shape != avg_responses.shape:
            raise ValueError(
                f"Inconsistent responses by trial shape: {avg_responses.shape=} {responses_by_trial.shape=}"
            )

        # Compute evaluation metrics (all are arrays of length n_neurons_session)
        corr_to_average = correlation_numpy(avg_responses, model_responses, axis=0)
        mse_to_average = MSE_numpy(avg_responses, model_responses, axis=0)

        n_trials = responses_by_trial.shape[1]
        if n_trials > 1:
            feve_values = feve(responses_by_trial, model_responses)
            # we already cut the frames from responses_by_trial
            jackknife, _ = oracle_corr_jackknife(responses_by_trial, cut_first_n_frames=None)
            var_ratio, explainable_var = explainable_vs_total_var(responses_by_trial)
        else:
            feve_values = np.full(n_neurons_session, np.nan)
            jackknife = np.full(n_neurons_session, np.nan)
            # Cannot compute var_ratio without multiple trials
            var_ratio = np.full(n_neurons_session, np.nan)

        # Compute per-trial metrics
        n_trials_per_session.append(n_trials)
        corr_by_trial = {}
        mse_by_trial = {}
        for i in range(n_trials):
            resp = responses_by_trial[:, i, :]
            corr_by_trial[i] = correlation_numpy(resp, model_responses, axis=0)
            mse_by_trial[i] = MSE_numpy(resp, model_responses, axis=0)

        # Get neuron specific information from the response objects and, if needed, the checkpoint data_info.
        neuron_data_info = _get_session_metadata(session, neuron_data_dict, model.data_info)

        # Preprocess session_kwargs to identify which fields are per-neuron arrays
        # vs scalar values that should be repeated for all neurons
        per_neuron_fields = {}
        scalar_fields = {}
        for key, value in neuron_data_info.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
                if arr.ndim > 0 and len(arr) == n_neurons_session:
                    per_neuron_fields[key] = arr
                else:
                    scalar_fields[key] = value
            elif isinstance(value, (np.ndarray, list, tuple)):
                # Check if it's a per-neuron array (length matches n_neurons_session)
                arr = np.asarray(value)
                if arr.ndim > 0 and len(arr) == n_neurons_session:
                    per_neuron_fields[key] = arr
                else:
                    # Scalar or different length - treat as scalar for this session
                    scalar_fields[key] = value
            else:
                # Scalar value (int, float, str, etc.)
                scalar_fields[key] = value

        # Create a dictionary entry for each neuron
        for neuron_idx in range(n_neurons_session):
            neuron_result = {
                "session": session,
                "neuron_relative_idx": neuron_idx,
                "data_split": data_split,
                "poisson_loss_to_average": float(poisson_loss_values[neuron_idx]),
                "corr_to_average": float(corr_to_average[neuron_idx]),
                "mse_to_average": float(mse_to_average[neuron_idx]),
                "feve": float(feve_values[neuron_idx]),
                "jackknife": float(jackknife[neuron_idx]),
                "var_ratio": float(var_ratio[neuron_idx]),
            }

            # Add per-trial metrics
            for i in range(n_trials):
                neuron_result[f"corr_{i}"] = float(corr_by_trial[i][neuron_idx])
                neuron_result[f"mse_{i}"] = float(mse_by_trial[i][neuron_idx])

            # Add all fields from session_kwargs dynamically
            for key, arr in per_neuron_fields.items():
                if neuron_idx < len(arr):
                    val = arr[neuron_idx]
                    neuron_result[key] = val.item() if hasattr(val, "item") else val
                else:
                    neuron_result[key] = None

            # Scalar fields (same value for all neurons in this session)
            for key, value in scalar_fields.items():
                neuron_result[key] = value

            # Add model responses if requested
            if cfg.evaluation.get("add_responses_to_model_results", False):
                neuron_result["model_response"] = model_responses[:, neuron_idx].tolist()

            results.append(neuron_result)

    # Create DataFrame from list of dictionaries
    df = pandas.DataFrame(results)

    var_ratio_cutoff = cfg.evaluation.get("var_ratio_cutoff", 0.15)

    # Check if var_ratio can be used for filtering
    n_neurons_total = len(df)
    n_neurons_with_var_ratio = df["var_ratio"].notna().sum()

    if n_neurons_with_var_ratio == 0:
        log.warning(
            f"No neurons have valid variance ratio values. "
            f"Skipping var_ratio_cutoff filtering. All {n_neurons_total} neurons will be included in evaluation."
        )
        df_filtered = df.copy()
        filtering_applied = False
    else:
        df_filtered = df[df["var_ratio"].notna() & (df["var_ratio"] >= var_ratio_cutoff)].copy()
        filtering_applied = True

    # Compute dataset statistics by iterating over the actual dataloaders (if enabled)
    if cfg.evaluation.get("compute_dataset_statistics", False):
        dataset_stats = compute_dataloader_statistics(dataloaders)
    else:
        dataset_stats = DatasetStatistics.empty()

    # Extract metadata from config
    model_tag = cfg.evaluation.get("model_tag", cfg.evaluation.model_path)
    exp_name = cfg.get("exp_name", "unknown")
    species = None
    if hasattr(cfg, "data_io") and hasattr(cfg.data_io, "responses"):
        species = cfg.data_io.responses.get("specie", None)
    if species is None and hasattr(cfg, "data_io") and hasattr(cfg.data_io, "stimuli"):
        species = cfg.data_io.stimuli.get("specie", None)

    # Build evaluation summary from DataFrame
    summary = EvaluationSummary.from_dataframe(
        df_filtered,
        df_all=df,
        model_path=str(cfg.evaluation.model_path),
        model_tag=str(model_tag),
        exp_name=str(exp_name),
        species=str(species) if species else None,
        data_split=data_split,
        temporal_lag=lag,
        n_trials_per_session=n_trials_per_session,
        n_neurons_total=n_neurons_total,
        var_ratio_cutoff=var_ratio_cutoff,
        filtering_applied=filtering_applied,
        dataset_stats=dataset_stats,
    )

    # Print formatted report
    summary.print_report()

    group_response_plot_path = cfg.evaluation.get(
        "group_response_plot_path",
        os.path.join(cfg.paths.output_dir, "group_response_traces.png"),
    )
    if group_response_plot_path is not None:
        _plot_group_response_traces(
            df_filtered=df_filtered,
            session_plot_data=session_plot_data,
            group_metric_breakdown=summary.group_metric_breakdown,
            output_path=str(group_response_plot_path),
        )

    # Save per-neuron results
    if cfg.evaluation.get("model_results_path") is not None:
        df.to_csv(cfg.evaluation.model_results_path)
        log.info(f"Per-neuron results saved to {cfg.evaluation.model_results_path}")

    # Save summary results
    if cfg.evaluation.get("summary_results_path") is not None:
        summary.save_json(cfg.evaluation.summary_results_path)
        log.info(f"Summary results saved to {cfg.evaluation.summary_results_path}")

    # Return average correlation computed on filtered neurons
    return summary.corr_to_avg_mean


if __name__ == "__main__":
    main()
