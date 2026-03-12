from pathlib import Path

import hydra
import numpy as np
import jax.numpy as jnp
import pytest

from openretina.data_io.base_dataloader import get_movie_dataloader
from openretina.data_io.hoefling_2024.dataloaders import natmov_dataloaders_v2
from openretina.data_io.hoefling_2024.responses import filter_responses, make_final_responses
from openretina.data_io.hoefling_2024.stimuli import movies_from_pickle
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.utils.misc import check_server_responding
from lightning.pytorch import seed_everything



def test_get_movie_dataloader_jax_backend():
    movie = np.random.randn(4, 2, 150, 18, 16).astype(np.float32)
    responses = np.random.randn(150, 3).astype(np.float32)

    loader = get_movie_dataloader(
        movie=movie,
        responses=responses,
        split="train",
        scene_length=10,
        chunk_size=5,
        batch_size=2,
        array_backend="jax",
        num_workers=0,
    )

    batch = next(iter(loader))
    inputs, targets = batch

    assert type(inputs).__module__.startswith("jax")
    assert type(targets).__module__.startswith("jax")
    assert inputs.shape == (2, 2, 150, 18, 16)
    assert targets.shape == (2, 150, 3)


@pytest.mark.parametrize(
    "batch_size,train_chunk_size,allow_over_boundaries",
    [
        (128, 50, True),
    ],
)
def test_natmov_dataloaders_jax_backend_real_data(batch_size: int, train_chunk_size: int, allow_over_boundaries: bool):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name="hoefling_2024_core_readout_low_res.yaml")

    movies_path = get_local_file_path(file_path=cfg.paths.movies_path, cache_folder=cfg.paths.data_dir)
    movies_dict = movies_from_pickle(movies_path)

    responses_path = get_local_file_path(file_path=cfg.paths.responses_path, cache_folder=cfg.paths.data_dir)
    responses_dict = load_h5_into_dict(file_path=responses_path)
    filtered_responses_dict = filter_responses(responses_dict, **cfg.quality_checks)
    final_responses = make_final_responses(filtered_responses_dict, response_type="natural")

    seed_everything(42)
    dataloaders = natmov_dataloaders_v2(
        neuron_data_dictionary=final_responses,
        movies_dictionary=movies_dict,
        allow_over_boundaries=allow_over_boundaries,
        batch_size=batch_size,
        train_chunk_size=train_chunk_size,
        validation_clip_indices=cfg.dataloader.validation_clip_indices,
        array_backend="jax",
    )

    session_key = next(iter(dataloaders["train"].keys()))
    batch = next(iter(dataloaders["train"][session_key]))
    inputs, targets = batch

    assert isinstance(inputs,jnp.ndarray)
    assert isinstance(targets,jnp.ndarray)
    assert inputs.shape[2] == train_chunk_size
    assert targets.shape[1] == train_chunk_size

    seed_everything(42)
    dataloaders_torch = natmov_dataloaders_v2(
        neuron_data_dictionary=final_responses,
        movies_dictionary=movies_dict,
        allow_over_boundaries=allow_over_boundaries,
        batch_size=batch_size,
        train_chunk_size=train_chunk_size,
        validation_clip_indices=cfg.dataloader.validation_clip_indices,
        array_backend="torch",
    )
    batch_torch = next(iter(dataloaders_torch["train"][session_key]))
    inputs_torch, targets_torch = batch_torch
    assert np.allclose(np.asarray(inputs),inputs_torch.detach().cpu().numpy())
    assert np.allclose(np.asarray(targets),targets_torch.detach().cpu().numpy())


