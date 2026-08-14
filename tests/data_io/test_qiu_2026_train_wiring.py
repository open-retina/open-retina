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

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit, compute_data_info
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


def _build_dataloaders(movies_dictionary: dict, responses, pupil, **overrides):
    """Build dataloaders the way cli/train.py does: `_partial_`, then a plain Python call.

    The call style matters for memory, not just style — see
    ``test_release_movies_needs_the_partial_call_style``.
    """
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout", overrides=DATALOADER_OVERRIDES)
    build = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
    return build(
        neuron_data_dictionary={SESSION_KEY: responses},
        movies_dictionary=movies_dictionary,
        pupil_dictionary={SESSION_KEY: pupil},
        **overrides,
    )


def test_shipped_config_releases_source_movies_during_construction() -> None:
    """The shipped qiu config sets release_movies, which empties the caller's dictionary as it goes.

    This is what keeps peak RSS down on the real dataset (~28 GB of movies, duplicated by the splits).
    Anything needing the raw movies must therefore run before dataloader construction — see the
    compute_data_info ordering in cli/train.py, guarded by the next test.
    """
    movies, responses, pupil = _make_synthetic_session()
    movies_dictionary = {SESSION_KEY: movies}

    dataloaders = _build_dataloaders(movies_dictionary, responses, pupil)

    assert movies_dictionary == {}, "shipped config should release source movies during construction"
    # Releasing must not damage what the dataloaders yield.
    train_batch = next(iter(dataloaders["train"][SESSION_KEY]))
    assert train_batch.inputs.shape[1] == N_CHANNELS
    assert torch.isfinite(train_batch.inputs).all()


def test_release_movies_needs_the_partial_call_style() -> None:
    """Pins WHY cli/train.py uses `_partial_` instead of instantiate(cfg.dataloader, **kwargs).

    Passing the data dictionaries through instantiate() gives the builder OmegaConf-rebuilt copies of
    the container and of each session wrapper (numpy buffers stay shared, so nothing is duplicated) —
    but the builder then releases entries from a copy, while the caller keeps every movie alive. The
    memory win silently disappears with no error anywhere, so it needs a test rather than a comment.
    """
    movies, responses, pupil = _make_synthetic_session()
    movies_dictionary = {SESSION_KEY: movies}

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout", overrides=DATALOADER_OVERRIDES)
    assert cfg.dataloader.release_movies is True, "this test is meaningless if the config disables release"
    hydra.utils.instantiate(
        cfg.dataloader,
        neuron_data_dictionary={SESSION_KEY: responses},
        movies_dictionary=movies_dictionary,
        pupil_dictionary={SESSION_KEY: pupil},
    )

    assert movies_dictionary != {}, (
        "instantiate(**kwargs) now propagates mutations to the caller's dict; if this ever changes, "
        "the _partial_ dance in cli/train.py and cli/eval.py can be simplified away"
    )


def test_release_movies_does_not_change_the_batches() -> None:
    movies, responses, pupil = _make_synthetic_session()
    kept = _build_dataloaders({SESSION_KEY: movies}, responses, pupil, release_movies=False)
    released = _build_dataloaders({SESSION_KEY: movies}, responses, pupil, release_movies=True)

    for split in ("train", "validation"):
        # The train sampler shifts chunk starts with np.random (MovieSampler.__iter__), so seed numpy —
        # not torch — before each iterator, or the two builds draw different offsets of the same movie.
        np.random.seed(0)
        kept_batch = next(iter(kept[split][SESSION_KEY]))
        np.random.seed(0)
        released_batch = next(iter(released[split][SESSION_KEY]))
        torch.testing.assert_close(kept_batch.inputs, released_batch.inputs)
        torch.testing.assert_close(kept_batch.targets, released_batch.targets)
        torch.testing.assert_close(kept_batch.pupil_center, released_batch.pupil_center)


def test_compute_data_info_does_not_depend_on_dataloader_construction() -> None:
    """cli/train.py computes data_info BEFORE building dataloaders; this is why that is safe."""
    movies, responses, pupil = _make_synthetic_session()
    movies_dictionary = {SESSION_KEY: movies}
    neuron_data_dictionary = {SESSION_KEY: responses}

    before = compute_data_info(neuron_data_dictionary, movies_dictionary)
    _build_dataloaders(movies_dictionary, responses, pupil, release_movies=False)
    after = compute_data_info(neuron_data_dictionary, movies_dictionary)

    assert before.keys() == after.keys()
    assert before["n_neurons_dict"] == after["n_neurons_dict"]
    assert before["input_shape"] == after["input_shape"]
    assert before["stim_mean"] == after["stim_mean"] and before["stim_std"] == after["stim_std"]
    assert before["sessions_kwargs"] == after["sessions_kwargs"]
    torch.testing.assert_close(before["mean_activity_dict"][SESSION_KEY], after["mean_activity_dict"][SESSION_KEY])
