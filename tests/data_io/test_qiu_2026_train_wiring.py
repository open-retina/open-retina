"""End-to-end validation that the qiu_2026 Hydra configs and the pupil-dictionary wiring added to
``train.py`` fit together, without requiring the real (multi-GB, not locally cached) qiu_2026 dataset.

Builds one synthetic session's worth of movies/responses/pupil in memory, drives it through the real
``configs/dataloader/qiu_2026.yaml`` config and the real ``configs/qiu_2026_core_readout.yaml`` model
config, and runs one training step end-to-end (dataloader -> model forward with the shifter -> loss).
This mirrors the "full training epoch with non-NaN loss" success criterion on synthetic data.
"""

import hydra
import numpy as np
import torch

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.models.core_readout import UnifiedCoreReadout

SESSION_KEY = "dynamic28188-16-3-Fluorescence-7b721b-v4a"
N_NEURONS = 5
CLIP_LENGTH = 300
NUM_TRAIN_CLIPS = 4  # 1 clip is held out via validation_clip_indices below; the rest must fill >= 1 batch
HEIGHT, WIDTH = 36, 64
N_CHANNELS = 3
# The real config's batch_size=32 needs far more synthetic clips than is practical to generate in a fast
# unit test; override it down to fit the tiny synthetic dataset above while keeping every other wiring
# (target_, kwarg names, dataset_cls/extra_arrays hook) exactly as shipped.
DATALOADER_OVERRIDES = ["trainer=debug", "dataloader.batch_size=2"]


def _make_synthetic_session() -> tuple[MoviesTrainTestSplit, ResponsesTrainTestSplit, dict]:
    rng = np.random.default_rng(0)
    train_time = CLIP_LENGTH * NUM_TRAIN_CLIPS

    movies = MoviesTrainTestSplit(
        train=rng.standard_normal((N_CHANNELS, train_time, HEIGHT, WIDTH)).astype(np.float32),
        test_dict={"cond1": rng.standard_normal((N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH)).astype(np.float32)},
        stim_id="qiu_2026",
    )
    responses = ResponsesTrainTestSplit(
        train=rng.random((N_NEURONS, train_time)).astype(np.float32),
        test_dict={"cond1": rng.random((N_NEURONS, CLIP_LENGTH)).astype(np.float32)},
        stim_id="qiu_2026",
        session_kwargs={"validation_clip_indices": [1]},
    )
    pupil = {
        "train": rng.standard_normal((2, train_time)).astype(np.float32),
        "test_dict": {"cond1": rng.standard_normal((2, CLIP_LENGTH)).astype(np.float32)},
    }
    return movies, responses, pupil


def test_qiu_2026_dataloader_config_produces_qiu_datapoints() -> None:
    movies, responses, pupil = _make_synthetic_session()

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout", overrides=DATALOADER_OVERRIDES)

    dataloader_kwargs = {
        "neuron_data_dictionary": {SESSION_KEY: responses},
        "movies_dictionary": {SESSION_KEY: movies},
    }
    # Mirrors the exact conditional added to train.py: qiu_2026 is the only data_io config exposing "pupil".
    assert "pupil" in cfg.data_io
    dataloader_kwargs["pupil_dictionary"] = {SESSION_KEY: pupil}

    dataloaders = hydra.utils.instantiate(cfg.dataloader, **dataloader_kwargs)

    assert set(dataloaders["train"].keys()) == {SESSION_KEY}
    train_batch = next(iter(dataloaders["train"][SESSION_KEY]))
    assert train_batch.inputs.shape[0] == train_batch.targets.shape[0] == train_batch.pupil_center.shape[0]
    assert train_batch.inputs.shape[1] == N_CHANNELS
    assert train_batch.pupil_center.shape[1] == 2


def test_qiu_2026_train_step_runs_end_to_end_with_shifter() -> None:
    movies, responses, pupil = _make_synthetic_session()

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout", overrides=DATALOADER_OVERRIDES)

    dataloaders = hydra.utils.instantiate(
        cfg.dataloader,
        neuron_data_dictionary={SESSION_KEY: responses},
        movies_dictionary={SESSION_KEY: movies},
        pupil_dictionary={SESSION_KEY: pupil},
    )

    n_neurons_dict = {SESSION_KEY: N_NEURONS}
    cfg.model.n_neurons_dict = n_neurons_dict
    model = UnifiedCoreReadout(data_info={}, **cfg.model)
    assert model.shifter is not None

    train_batch = next(iter(dataloaders["train"][SESSION_KEY]))
    total_loss = model.training_step((SESSION_KEY, train_batch), 0)

    assert torch.isfinite(total_loss)
