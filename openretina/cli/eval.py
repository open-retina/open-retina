#!/usr/bin/env python3

import logging
import os
from collections import defaultdict

import hydra
import lightning.pytorch
import numpy as np
import pandas
import torch
from omegaconf import DictConfig, OmegaConf

from openretina.eval.metrics import MSE_numpy, correlation_numpy, feve
from openretina.eval.oracles import oracle_corr_jackknife
from openretina.models.core_readout import load_core_readout_model
from openretina.modules.losses import PoissonLoss3d

log = logging.getLogger(__name__)
logging.captureWarnings(True)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="hoefling_2024_core_readout_high_res",
)
def main(cfg: DictConfig) -> float | None:
    score = evaluate_model(cfg)
    return score


def evaluate_model(cfg: DictConfig) -> float:
    log.info("Logging full config:")
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.paths.cache_dir is None:
        raise ValueError("Please provide a cache_dir for the data in the config file or as a command line argument.")
    if cfg.paths.load_model_path is None:
        raise ValueError("Please provide paths.load_model_path to define which model to test.")

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

    test_key = "test"
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

    results = defaultdict(list)
    dataloader_dict = dataloaders[test_key]
    poisson_loss = PoissonLoss3d(per_neuron=True)
    lag = -1
    for session, dl in dataloader_dict.items():
        dataset = dl.dataset
        movies = dataset.movies.to(device).unsqueeze(0)

        with torch.no_grad():
            model_responses_torch = model.forward(movies, data_key=session)
            targets = dataset.responses.to(device).unsqueeze(0)
            poisson_loss_session = poisson_loss(model_responses_torch, targets)

        results["poisson_loss_to_average"].extend(poisson_loss_session.cpu().numpy().tolist())

        model_responses = model_responses_torch.squeeze(0).cpu().numpy()
        if cfg.evaluation.get("add_responses_to_model_results", False):
            results["model_response"].extend([x.tolist() for x in model_responses.T])

        avg_responses = dataset.responses.numpy()
        try:
            responses_by_trial = dataset.test_responses_by_trial.cpu().numpy()
            responses_by_trial = np.swapaxes(responses_by_trial, 0, 2)
        except Exception as e:
            print(f"Could not retrieve responses by trial: {e}")
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

        # Add evaluation specific
        results["corr_to_average"].extend(correlation_numpy(avg_responses, model_responses, axis=0).tolist())
        results["mse_to_average"].extend(MSE_numpy(avg_responses, model_responses, axis=0).tolist())
        results["feve"].extend(feve(responses_by_trial, model_responses).tolist())
        jackknife, _ = oracle_corr_jackknife(responses_by_trial, cut_first_n_frames=lag)
        results["jackknife"].extend(jackknife.tolist())

        for i in range(responses_by_trial.shape[1]):
            resp = responses_by_trial[:, i, :]
            results[f"corr_{i}"].extend(correlation_numpy(resp, model_responses, axis=0).tolist())
            results[f"mse_{i}"].extend(MSE_numpy(resp, model_responses, axis=0).tolist())

        # Add neuron specific information
        neuron_data_info = neuron_data_dict[session].session_kwargs
        results["session"].extend([session] * n_neurons_session)
        if "roi_ids" in neuron_data_info:
            results["roi_ids"].extend(neuron_data_info["roi_ids"].tolist())
        if "group_assignment" in neuron_data_info:
            results["group_assignment"].extend(neuron_data_info["group_assignment"].tolist())
        if "scan_sequence_idx" in neuron_data_info:
            scan_sequence_idx = int(neuron_data_info["scan_sequence_idx"])
            results["scan_sequence_idx"].extend([scan_sequence_idx] * n_neurons_session)

    # print statistics
    df = pandas.DataFrame(results)
    print(f"{cfg.paths.load_model_path} ({lag=})")
    for k in ["corr_to_average", "mse_to_average", "feve", "poisson_loss_to_average"]:
        print(f"{k}: {df[k].mean():.3f}")
    # comparison to individual traces
    for k in ["corr", "mse"]:
        avgs, i = [], 0
        while (key_ := f"{k}_{i}") in df:
            avgs.append(df[key_].mean())
            i += 1
        print(f"{k}: {np.mean(avgs):.3f} (+-{np.std(avgs):.3f})")

    if cfg.evaluation.get("model_results_path") is not None:
        df.to_csv(cfg.evaluation.model_results_path)

    avg_correlation = float(df["corr_to_average"].mean())
    return avg_correlation


if __name__ == "__main__":
    main()
