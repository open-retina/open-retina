"""End-to-end validation that the qiu_2026 Hydra configs and the pupil-dictionary wiring added to
``train.py`` fit together, without requiring the real (multi-GB, not locally cached) qiu_2026 dataset.

Builds one synthetic session's worth of movies/responses/pupil in memory, drives it through the real
``configs/dataloader/qiu_2026.yaml`` config and the real ``configs/qiu_2026_core_readout.yaml`` model
config, and runs one training step end-to-end (dataloader -> model forward with the shifter -> loss).
This mirrors the "full training epoch with non-NaN loss" success criterion on synthetic data.
"""

import hydra
import numpy as np
import pytest
import torch

from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit, compute_data_info
from openretina.models.core_readout import UnifiedCoreReadout

SESSION_KEY = "dynamic28188-16-3-Fluorescence-7b721b-v4a"
N_NEURONS = 5
CLIP_LENGTH = 300
NUM_TRAIN_CLIPS = 4  # 1 clip is held out via validation_clip_indices below; the rest must fill >= 1 batch
HEIGHT, WIDTH = 36, 64
N_CHANNELS = 3
N_TEST_REPEATS = 3  # the real dataset has 15-20 repeats per test condition
# The real config's batch_size=32 needs far more synthetic clips than is practical to generate in a fast
# unit test; override it down to fit the tiny synthetic dataset above while keeping every other wiring
# (target_, kwarg names, dataset_cls/extra_arrays hook) exactly as shipped.
DATALOADER_OVERRIDES = ["trainer=debug", "dataloader.batch_size=2"]


def _make_synthetic_session() -> tuple[MoviesTrainTestSplit, ResponsesTrainTestSplit, dict]:
    rng = np.random.default_rng(0)
    train_time = CLIP_LENGTH * NUM_TRAIN_CLIPS

    # Per-repeat behavior/pupil for the one test condition, mirroring what the real loaders produce:
    # the movie is identical on every repeat, the behavior and eye position are not.
    behavior_by_repeat = rng.standard_normal((N_TEST_REPEATS, N_CHANNELS - 1, CLIP_LENGTH)).astype(np.float32)
    pupil_by_repeat = rng.standard_normal((N_TEST_REPEATS, 2, CLIP_LENGTH)).astype(np.float32)

    # Test movie channel 0 is the video; channels 1+ carry the across-repeat mean behavior, broadcast
    # over every pixel, exactly as load_stimuli_for_session assembles it.
    test_movie = np.empty((N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH), dtype=np.float32)
    test_movie[0] = rng.standard_normal((CLIP_LENGTH, HEIGHT, WIDTH)).astype(np.float32)
    test_movie[1:] = behavior_by_repeat.mean(axis=0)[:, :, None, None]

    movies = MoviesTrainTestSplit(
        train=rng.standard_normal((N_CHANNELS, train_time, HEIGHT, WIDTH)).astype(np.float32),
        test_dict={"cond1": test_movie},
        stim_id="qiu_2026",
        test_behavior_by_repeat_dict={"cond1": behavior_by_repeat},
    )
    responses = ResponsesTrainTestSplit(
        train=rng.random((N_NEURONS, train_time)).astype(np.float32),
        test_dict={"cond1": rng.random((N_NEURONS, CLIP_LENGTH)).astype(np.float32)},
        stim_id="qiu_2026",
        session_kwargs={"validation_clip_indices": [1]},
    )
    pupil = {
        "train": rng.standard_normal((2, train_time)).astype(np.float32),
        "test_dict": {"cond1": pupil_by_repeat.mean(axis=0)},
        "test_by_repeat_dict": {"cond1": pupil_by_repeat},
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


def test_both_clis_build_the_same_dataloader_kwargs() -> None:
    """`openretina eval` must thread the pupil trace exactly as `openretina train` does.

    eval.py used to build its dataloaders from two fixed keyword arguments, with no way to pass a
    third stream, so the qiu_2026 builder raised on the missing `pupil_dictionary` and the dataset
    could not be evaluated from the CLI at all. This reads the conditional out of both modules'
    source so the two cannot drift apart again.
    """
    import inspect

    from openretina.cli import eval as eval_cli
    from openretina.cli import train as train_cli

    pupil_wiring = 'dataloader_kwargs["pupil_dictionary"] = hydra.utils.call(cfg.data_io.pupil)'
    for module in (train_cli, eval_cli):
        source = inspect.getsource(module)
        assert 'if "pupil" in cfg.data_io:' in source, f"{module.__name__} does not check for a pupil target"
        assert pupil_wiring in source, f"{module.__name__} does not pass pupil_dictionary to the builder"
        assert "build_dataloaders(**dataloader_kwargs)" in source, (
            f"{module.__name__} does not pass the assembled kwargs to the builder"
        )


def test_eval_config_path_builds_qiu_dataloaders() -> None:
    """The kwargs eval.py assembles must actually satisfy the shipped qiu_2026 builder."""
    movies, responses, pupil = _make_synthetic_session()

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="qiu_2026_core_readout", overrides=DATALOADER_OVERRIDES)

    # Exactly what cli/eval.py now assembles.
    dataloader_kwargs = {
        "neuron_data_dictionary": {SESSION_KEY: responses},
        "movies_dictionary": {SESSION_KEY: movies},
    }
    if "pupil" in cfg.data_io:
        dataloader_kwargs["pupil_dictionary"] = {SESSION_KEY: pupil}

    build_dataloaders = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
    dataloaders = build_dataloaders(**dataloader_kwargs)

    assert "train" in dataloaders
    assert set(dataloaders["train"].keys()) == {SESSION_KEY}


def test_test_dataloader_yields_one_batch_holding_every_repeat() -> None:
    """The shipped config evaluates a test condition as R repeats in the batch dimension."""
    movies, responses, pupil = _make_synthetic_session()
    behavior_by_repeat = movies.test_behavior_by_repeat_dict["cond1"]
    pupil_by_repeat = pupil["test_by_repeat_dict"]["cond1"]

    dataloaders = _build_dataloaders({SESSION_KEY: movies}, responses, pupil)

    assert "cond1" in dataloaders, "each test condition gets its own top-level dataloader key"
    batches = list(dataloaders["cond1"][SESSION_KEY])
    assert len(batches) == 1, "the whole condition must be one batch, or the repeats cannot be averaged"
    batch = batches[0]

    assert batch.inputs.shape == (N_TEST_REPEATS, N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH)
    assert batch.targets.shape == (N_TEST_REPEATS, CLIP_LENGTH, N_NEURONS)
    assert batch.pupil_center.shape == (N_TEST_REPEATS, 2, CLIP_LENGTH)
    assert bool(torch.as_tensor(batch.reduce_over_batch).all()), "batch must be marked for reduction"

    # Every repeat carries the same trial-averaged target...
    assert torch.equal(batch.targets[0], batch.targets[-1])
    # ...the same video...
    assert torch.equal(batch.inputs[0, 0], batch.inputs[-1, 0])
    # ...and its own behavior and pupil trace, spatially constant across the frame.
    for repeat in range(N_TEST_REPEATS):
        assert torch.allclose(batch.inputs[repeat, 1:, :, 0, 0], torch.as_tensor(behavior_by_repeat[repeat]), atol=1e-6)
        assert torch.allclose(batch.pupil_center[repeat], torch.as_tensor(pupil_by_repeat[repeat]), atol=1e-6)
    assert not torch.equal(batch.inputs[0, 1:], batch.inputs[-1, 1:]), "behavior must differ across repeats"


def test_repeat_test_dataset_stores_the_video_once() -> None:
    """The video is identical across repeats, so storing it per repeat would waste ~R x the test memory."""
    movies, responses, pupil = _make_synthetic_session()

    dataloaders = _build_dataloaders({SESSION_KEY: movies}, responses, pupil)
    dataset = dataloaders["cond1"][SESSION_KEY].dataset

    assert len(dataset) == N_TEST_REPEATS
    assert dataset.video.shape == (1, CLIP_LENGTH, HEIGHT, WIDTH)
    # The stored per-repeat arrays are traces, not movies: no spatial dimensions.
    assert dataset.behavior_by_repeat.shape == (N_TEST_REPEATS, N_CHANNELS - 1, CLIP_LENGTH)
    assert dataset.pupil_by_repeat.shape == (N_TEST_REPEATS, 2, CLIP_LENGTH)
    # cli/eval.py reads both of these off the dataset object.
    assert dataset.responses.shape == (CLIP_LENGTH, N_NEURONS)
    assert hasattr(dataset, "test_responses_by_trial")


def test_per_repeat_test_behavior_false_restores_the_averaged_input_path() -> None:
    """The escape hatch must reproduce the old single-clip test batch exactly."""
    movies, responses, pupil = _make_synthetic_session()

    dataloaders = _build_dataloaders({SESSION_KEY: movies}, responses, pupil, per_repeat_test_behavior=False)
    batches = list(dataloaders["cond1"][SESSION_KEY])

    assert len(batches) == 1
    batch = batches[0]
    assert batch.inputs.shape == (1, N_CHANNELS, CLIP_LENGTH, HEIGHT, WIDTH)
    assert batch.pupil_center.shape == (1, 2, CLIP_LENGTH)
    assert not hasattr(batch, "reduce_over_batch"), "the averaged-input path must not mark its batches"
    # Inputs are the repeat-averaged ones the loaders put in test_dict.
    assert torch.allclose(batch.inputs[0], torch.as_tensor(movies.test_dict["cond1"]), atol=1e-6)
    assert torch.allclose(
        batch.pupil_center[0], torch.as_tensor(pupil["test_by_repeat_dict"]["cond1"]).mean(dim=0), atol=1e-6
    )


def test_per_repeat_test_behavior_without_per_repeat_arrays_is_an_error() -> None:
    """Falling back silently would score the wrong estimator with no sign anything happened."""
    movies, responses, pupil = _make_synthetic_session()
    movies_without_repeats = MoviesTrainTestSplit(
        train=movies.train,
        test_dict=movies.test_dict,
        stim_id=movies.stim_id,
    )

    with pytest.raises(ValueError, match="per_repeat_test_behavior"):
        _build_dataloaders({SESSION_KEY: movies_without_repeats}, responses, pupil)
