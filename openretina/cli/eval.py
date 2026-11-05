#!/usr/bin/env python3

import logging
import os
import re
from pathlib import Path

import hydra
import lightning.pytorch
import numpy as np
import pandas
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.auto import tqdm

from openretina.data_io.base import DatasetStatistics
from openretina.eval.metrics import MSE_numpy, correlation_numpy, explainable_vs_total_var, feve
from openretina.eval.oracles import oracle_corr_jackknife
from openretina.models.core_readout import BaseCoreReadout, load_core_readout_model
from openretina.modules.losses import PoissonLoss3d
from openretina.utils.eval_utils import EvaluationSummary, align_responses_to_model_output
from openretina.utils.frame_fingerprints import compute_dataloader_statistics
from openretina.utils.misc import reorder_like_a

log = logging.getLogger(__name__)
logging.captureWarnings(True)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="karamanlis_2024_eval",
)
def main(cfg: DictConfig) -> float | None:
    score = evaluate_model(cfg)
    return score


def _tagged_output_path(path: str, data_split: str) -> str:
    """Insert a filesystem-safe form of ``data_split`` before the suffix of ``path``.

    Used when several splits are evaluated in one run, so each writes its own file:
    ``eval/per_neuron.csv`` + split ``Oup5uAZxF2G7zEJkT+ui`` -> ``eval/per_neuron_Oup5uAZxF2G7zEJkT_ui.csv``.
    Split names are not always path-safe -- qiu_2026 keys its test splits by DataJoint ``condition_hash``,
    which is base64 and so can contain ``/`` and ``+``.
    """
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", data_split)
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{safe}{p.suffix}"))


def evaluate_model(cfg: DictConfig) -> float:
    """Run the full evaluation pipeline: load a trained model, compute per-neuron metrics
    (correlation, FEVe, Poisson loss, oracle), and return the mean correlation.

    ``evaluation.data_split`` may be a single split name or a list of them. Loading the stimuli,
    responses and dataloaders costs far more than evaluating one split against them (~10 min vs ~5 min
    for the ten qiu_2026 sessions), and none of it depends on which split is scored, so a list is
    evaluated in one pass over the data rather than one process per split. Each split still gets its
    own report, DataFrame and summary; with a list, ``model_results_path``/``summary_results_path`` are
    suffixed per split so they do not overwrite each other. The returned score is that split's mean
    correlation, or the mean over splits for a list.
    """
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.get("paths", {}).get("cache_dir") is None:
        raise ValueError("Please provide paths.cache_dir for the data in the config file or via the command line.")
    if cfg.get("evaluation", {}).get("model_path") is None:
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

    dataloader_kwargs = {
        "neuron_data_dictionary": neuron_data_dict,
        "movies_dictionary": movies_dict,
    }

    # Some datasets (e.g. qiu_2026) additionally carry a per-session behavioral trace consumed by a
    # shifter; their data_io config exposes it as an extra "pupil" target so it can be loaded and
    # threaded through here. Absent for all other datasets, so this is a no-op elsewhere.
    if "pupil" in cfg.data_io:
        dataloader_kwargs["pupil_dictionary"] = hydra.utils.call(cfg.data_io.pupil)

    # `_partial_` then a plain call, so builders configured with `release_movies` can actually free each
    # session's source movie as they go -- see the comment in cli/train.py for why instantiate(**kwargs)
    # defeats that. `movies_dict` is not read after this point.
    build_dataloaders = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
    dataloaders = build_dataloaders(**dataloader_kwargs)

    if cfg.seed is not None:
        lightning.pytorch.seed_everything(cfg.seed)

    # Note: a data split can be either "train", "validation", or "test", or, if we have multiple test
    # stimuli, the name of the test stimulus (as it appears in the dataloaders)
    requested_split = cfg.evaluation.get("data_split", "test")
    # A bare string keeps the historical single-split behaviour, including unsuffixed output paths.
    single_split = isinstance(requested_split, str)
    if single_split:
        data_splits = [requested_split]
    elif isinstance(requested_split, (list, tuple, ListConfig)):
        data_splits = [str(s) for s in requested_split]
    else:
        # Anything else (notably `data_split=null`, which the config makes easy to reach) would
        # otherwise fail as a bare TypeError on iteration, minutes after the data was loaded.
        raise ValueError(
            f"evaluation.data_split must be a split name or a list of split names, got {requested_split!r}."
        )
    if len(data_splits) == 0:
        raise ValueError("evaluation.data_split is an empty list; give at least one split name.")

    # Validate every split up front: the data is already loaded by this point, but the per-split work
    # below is minutes long, so a typo in the last of six splits should not surface six passes in.
    missing = [s for s in data_splits if s not in dataloaders]
    if missing:
        raise ValueError(f"Dataloader(s) {missing} not found. Available keys: {sorted(dataloaders.keys())}")

    # Two splits whose names differ only where the tag sanitises -- base64 `condition_hash` keys make
    # '+' and '/' both become '_' -- would write to one filename, the second silently overwriting the
    # first and the run reporting six conditions but leaving five files. Refuse instead: the splits
    # can still be evaluated in separate runs, which is cheaper than a result that is quietly wrong.
    if not single_split:
        by_tag: dict[str, list[str]] = {}
        for s in data_splits:
            by_tag.setdefault(_tagged_output_path("x.csv", s), []).append(s)
        collisions = [group for group in by_tag.values() if len(group) > 1]
        if collisions:
            raise ValueError(
                f"These split names would write to the same output file once made path-safe: {collisions}. "
                "Evaluate them in separate runs, or give model_results_path/summary_results_path per run."
            )

    # Dataset statistics describe the dataloaders as a whole, not the split under evaluation, so they
    # are computed once and shared. They are also the single most expensive option here -- they walk
    # every dataloader including train -- which is exactly why this must not sit inside the loop.
    if cfg.evaluation.get("compute_dataset_statistics", False):
        dataset_stats = compute_dataloader_statistics(dataloaders)
    else:
        dataset_stats = DatasetStatistics.empty()

    scores = []
    for data_split in data_splits:
        df, summary = evaluate_split(
            cfg=cfg,
            model=model,
            dataloaders=dataloaders,
            neuron_data_dict=neuron_data_dict,
            data_split=data_split,
            device=device,
            dataset_stats=dataset_stats,
        )

        summary.print_report()

        # Save per-neuron results
        results_path = cfg.evaluation.get("model_results_path")
        if results_path is not None:
            results_path = results_path if single_split else _tagged_output_path(results_path, data_split)
            df.to_csv(results_path)
            log.info(f"Per-neuron results saved to {results_path}")

        # Save summary results
        summary_path = cfg.evaluation.get("summary_results_path")
        if summary_path is not None:
            summary_path = summary_path if single_split else _tagged_output_path(summary_path, data_split)
            summary.save_json(summary_path)
            log.info(f"Summary results saved to {summary_path}")

        scores.append(summary.corr_to_avg_mean)

    # A split whose neurons all fall below var_ratio_cutoff scores nan, which is routine on a strict
    # cutoff. Averaging it in with `mean` would turn the whole run's score nan and hide what the other
    # splits found, so the aggregate skips the nans and says which splits it skipped.
    empty_splits = [s for s, score in zip(data_splits, scores, strict=True) if np.isnan(score)]
    if empty_splits:
        log.warning(
            f"No neurons passed var_ratio_cutoff for split(s) {empty_splits}; they score nan and are "
            f"left out of the mean over splits."
        )

    if not single_split:
        _print_across_split_report(data_splits, scores)

    # Return average correlation computed on filtered neurons, averaged over splits if there were several
    return _mean_over_splits(scores)


def _mean_over_splits(scores: list[float]) -> float:
    """Mean of the per-split scores, ignoring splits that scored nan (nan only if every split did)."""
    if all(np.isnan(score) for score in scores):
        return float("nan")
    return float(np.nanmean(scores))


def _print_across_split_report(data_splits: list[str], scores: list[float]) -> None:
    """Print one line per split plus their mean, after the individual reports."""
    print("=" * 80)
    print(f"Mean correlation across {len(data_splits)} evaluated splits")
    print("=" * 80)
    width = max(len(s) for s in data_splits)
    for data_split, score in zip(data_splits, scores, strict=True):
        print(f"  {data_split:{width}s} : {score:.3f}")
    print("-" * 80)
    print(f"  {'mean':{width}s} : {_mean_over_splits(scores):.3f}")
    print("=" * 80)


def evaluate_split(
    cfg: DictConfig,
    model: BaseCoreReadout,
    dataloaders: dict,
    neuron_data_dict: dict,
    data_split: str,
    device: str,
    dataset_stats: DatasetStatistics,
) -> tuple[pandas.DataFrame, EvaluationSummary]:
    """Score one already-built split: per-neuron metrics plus the summary over them.

    Split out of :func:`evaluate_model` so that several splits can share one load of the data; it
    neither loads nor writes anything, and holds no state between calls.
    """
    dataloader_dict = dataloaders[data_split]
    results = []  # List of dictionaries, one per neuron
    poisson_loss = PoissonLoss3d(per_neuron=True)
    lag = -1
    n_trials_per_session = []  # Track number of trials/repeats per session

    for session, dl in tqdm(dataloader_dict.items(), desc=f"Evaluating sessions ({data_split})", unit="session"):
        dataset = dl.dataset

        with torch.no_grad():
            model_responses_torch_array = []
            targets_array = []
            for data_point in dl:
                # Datasets carrying a behavioral variable (qiu_2026) need it for the shifter; without it
                # the shifter stays inert and the model scores as if it had none.
                pupil_center = getattr(data_point, "pupil_center", None)
                targets_batch = data_point[1]
                model_resp_batch = model.forward(
                    data_point[0].to(device),
                    data_key=session,
                    pupil_center=None if pupil_center is None else pupil_center.to(device),
                )
                # A batch whose batch dimension holds repeats of one stimulus contributes a single
                # prediction: the average over repeats, paired with the (identical) trial-averaged
                # target. Flattening it instead would lay the repeats end to end along time.
                reduce_flag = getattr(data_point, "reduce_over_batch", None)
                if reduce_flag is not None and bool(torch.as_tensor(reduce_flag).all()):
                    model_resp_batch = model_resp_batch.mean(dim=0, keepdim=True)
                    targets_batch = targets_batch[:1]
                model_resp = model_resp_batch.flatten(0, 1)
                model_responses_torch_array.append(model_resp)
                targets_array.append(targets_batch.flatten(0, 1))
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

        # Get neuron specific information from session_kwargs
        neuron_data_info = neuron_data_dict[session].session_kwargs

        # Preprocess session_kwargs to identify which fields are per-neuron arrays
        # vs scalar values that should be repeated for all neurons
        per_neuron_fields = {}
        scalar_fields = {}
        for key, value in neuron_data_info.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                # Check if it's a per-neuron array (length matches n_neurons_session)
                arr = np.asarray(value)
                if len(arr) == n_neurons_session:
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

    return df, summary


if __name__ == "__main__":
    main()
