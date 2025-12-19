#!/usr/bin/env python3

import logging
import os

import hydra
import lightning.pytorch
import numpy as np
import pandas
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from openretina.eval.metrics import MSE_numpy, correlation_numpy, feve
from openretina.eval.oracles import oracle_corr_jackknife
from openretina.models.core_readout import load_core_readout_model
from openretina.modules.losses import PoissonLoss3d
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

    for session, dl in dataloader_dict.items():
        dataset = dl.dataset
        movies = dataset.movies.to(device).unsqueeze(0)

        with torch.no_grad():
            model_responses_torch = model.forward(movies, data_key=session)
            targets = dataset.responses.to(device).unsqueeze(0)
            poisson_loss_session = poisson_loss(model_responses_torch, targets)

        poisson_loss_values = poisson_loss_session.cpu().numpy()
        model_responses = model_responses_torch.squeeze(0).cpu().numpy()

        avg_responses = dataset.responses.numpy()
        try:
            responses_by_trial = dataset.test_responses_by_trial.cpu().numpy()
            responses_by_trial = reorder_like_a(a=avg_responses, b=responses_by_trial)
            # rearrange to match feve function expectations
            responses_by_trial = rearrange(responses_by_trial, "trials time neurons -> time trials neurons")
        except Exception as e:
            log.error(
                f"Could not retrieve responses by trial for session {session}: {e}. Using avg responses instead",
                exc_info=True,
            )
            responses_by_trial = avg_responses[:, np.newaxis, :]

        # adjust responses to lag
        new_lag = avg_responses.shape[0] - model_responses.shape[0]
        if lag < 0:
            lag = new_lag
        elif new_lag != lag:
            raise ValueError(f"Inconsistent lag: {new_lag=} {lag=}")

        avg_responses = avg_responses[lag:]
        n_neurons_session = avg_responses.shape[1]
        responses_by_trial = responses_by_trial[lag:]

        if model_responses.shape != avg_responses.shape:
            raise ValueError(f"Inconsistent Shapes: {model_responses.shape=}, {avg_responses.shape}, {lag=}")
        if responses_by_trial[:, 0, :].shape != avg_responses.shape:
            raise ValueError(
                f"Inconsistent responses by trial shape: {avg_responses.shape=} {responses_by_trial.shape=}"
            )

        # Compute evaluation metrics (all are arrays of length n_neurons_session)
        corr_to_average = correlation_numpy(avg_responses, model_responses, axis=0)
        mse_to_average = MSE_numpy(avg_responses, model_responses, axis=0)
        feve_values = feve(responses_by_trial, model_responses)
        jackknife, _ = oracle_corr_jackknife(responses_by_trial, cut_first_n_frames=lag)

        # Compute per-trial metrics
        n_trials = responses_by_trial.shape[1]
        corr_by_trial = {}
        mse_by_trial = {}
        for i in range(n_trials):
            resp = responses_by_trial[:, i, :]
            corr_by_trial[i] = correlation_numpy(resp, model_responses, axis=0)
            mse_by_trial[i] = MSE_numpy(resp, model_responses, axis=0)

        # Get neuron specific information from session_kwargs
        # This can contain any fields, so we'll add all of them dynamically
        neuron_data_info = neuron_data_dict[session].session_kwargs

        # Preprocess session_kwargs to identify which fields are per-neuron arrays
        # vs scalar values that should be repeated for all neurons
        per_neuron_fields = {}
        scalar_fields = {}
        for key, value in neuron_data_info.items():
            if value is None:
                scalar_fields[key] = None
            elif isinstance(value, (np.ndarray, list, tuple)):
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
            }

            # Add per-trial metrics
            for i in range(n_trials):
                neuron_result[f"corr_{i}"] = float(corr_by_trial[i][neuron_idx])
                neuron_result[f"mse_{i}"] = float(mse_by_trial[i][neuron_idx])

            # Add all fields from session_kwargs dynamically
            # Per-neuron fields (arrays with length matching n_neurons_session)
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

    model_path_display = cfg.paths.get("load_model_path", cfg.evaluation.model_path)
    print(f"{model_path_display} ({data_split=}, {lag=})")
    for k in ["corr_to_average", "mse_to_average", "feve", "poisson_loss_to_average"]:
        print(f"{k}: {np.nanmean(df[k]):.3f}")
    # comparison to individual traces
    for k in ["corr", "mse"]:
        avgs, i = [], 0
        while (key_ := f"{k}_{i}") in df:
            avgs.append(np.nanmean(df[key_]))
            i += 1
        if avgs:
            print(f"{k}: {np.nanmean(avgs):.3f} (+-{np.nanstd(avgs):.3f})")

    if cfg.evaluation.get("model_results_path") is not None:
        df.to_csv(cfg.evaluation.model_results_path)

    avg_correlation = float(np.nanmean(df["corr_to_average"]))
    return avg_correlation


if __name__ == "__main__":
    main()
