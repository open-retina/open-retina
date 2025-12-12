#!/usr/bin/env python3

from collections import defaultdict
import logging
import os
import pickle

import hydra
import numpy as np
import lightning.pytorch
import torch
from omegaconf import DictConfig, OmegaConf

from openretina.models.core_readout import load_core_readout_model

from openretina.eval.metrics import feve, correlation_numpy, MSE_numpy
from openretina.eval.oracles import oracle_corr_jackknife, global_mean_oracle
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

    model_responses_dict = {}
    results = defaultdict(list)
    dataloader_dict = dataloaders[test_key]
    poisson_loss = PoissonLoss3d(per_neuron=True)
    lag = -1
    for session, dl in dataloader_dict.items():
        dataset = dl.dataset
        movies = dataset.movies.to(device).unsqueeze(0)
        neuron_data = neuron_data_dict[session]
        # other potentially useful keys: "roi_mask", "group_assignment", "eye"
        roi_ids = neuron_data.session_kwargs["roi_ids"]

        with torch.no_grad():
            model_responses_torch = model.forward(movies, data_key=session)
            targets = dataset.responses.to(device).unsqueeze(0)
            poisson_loss_session = poisson_loss(model_responses_torch, targets)

        results["poisson_loss"].append(poisson_loss_session.cpu().numpy())

        model_responses = model_responses_torch.squeeze(0).cpu().numpy()
        model_responses_dict[session] = model_responses

        responses_by_trial = dataset.test_responses_by_trial.cpu().numpy()
        responses_by_trial = np.swapaxes(responses_by_trial, 0, 2)
        avg_responses = dataset.responses.numpy()

        # adjust responses to lag
        new_lag = avg_responses.shape[0] - model_responses.shape[0]
        if lag < 0:
            lag = new_lag
        elif new_lag != lag:
            raise ValueError(f"Inconsistent lag: {new_lag=} {lag=}")

        avg_responses = avg_responses[lag:]
        responses_by_trial = responses_by_trial[lag:]

        if model_responses.shape != avg_responses.shape:
            raise ValueError(f"Inconsistent Shapes: "
                             f"{model_responses.shape=}, {avg_responses.shape}, {lag=}")
        if responses_by_trial[:, 0, :].shape != avg_responses.shape:
            raise ValueError(f"Inconsistent responses by trial shape: "
                             f"{avg_responses.shape=} {responses_by_trial.shape=}")

        results["corr"].append(correlation_numpy(avg_responses, model_responses, axis=0))
        results["mse"].append(MSE_numpy(avg_responses, model_responses, axis=0))
        results["feve"].append(feve(responses_by_trial, model_responses))
        jackknife, _ = oracle_corr_jackknife(responses_by_trial, cut_first_n_frames=lag)
        results["jackknife"].append(jackknife)

        for i in range(responses_by_trial.shape[1]):
            resp = responses_by_trial[:, i, :]
            results[f"corr_{i}"].append(correlation_numpy(resp, model_responses, axis=0))
            results[f"mse_{i}"].append(MSE_numpy(resp, model_responses, axis=0))

    print(f"{cfg.paths.load_model_path} ({lag=})")
    for k, v in results.items():
        avg = np.average(np.concatenate(v))
        print(f"{k}: {avg:.3f}")

    avg_correlation = float(np.average(np.concatenate(results["corr"])))

    if cfg.evaluation.get("model_results_path") is not None:
        with open(cfg.evaluation.model_results_path, "wb") as f:
            pickle.dump(model_responses_dict, f)
    return avg_correlation


if __name__ == "__main__":
    main()
