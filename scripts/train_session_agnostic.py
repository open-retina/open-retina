import os
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

from openretina.cyclers import LongCycler
from openretina.hoefling_2024.configs import model_config, trainer_config
from openretina.hoefling_2024.data_io import get_mb_dataloaders
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
from openretina.neuron_data_io import filter_responses, make_final_responses
from openretina.training import save_model
from openretina.training import standard_early_stop_trainer as trainer
from openretina.utils.h5_handling import load_h5_into_dict


@hydra.main(config_path="../configs", config_name="minimal_session_agnostic")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load data, filter responses
    data_path = os.path.join(cfg.data.base_path, "2024-08-14_neuron_data_responses_484c12d_djimaging.h5")
    movies_path = os.path.join(cfg.data.base_path, "2024-01-11_movies_dict_c285329.pkl")
    responses = load_h5_into_dict(data_path)
    movies_dict = pickle.load(open(movies_path, "rb"))
    filtered_responses = filter_responses(responses, **OmegaConf.to_object(cfg.quality_checks))  # type: ignore

    # Get dataloaders for mb model, train MB model from which we will extract the spatial masks.
    mb_data_dict = make_final_responses(filtered_responses, response_type="mb", trace_type="spikes")
    mb_dataloaders = get_mb_dataloaders(mb_data_dict)

    mb_dataloaders = {
        "train": mb_dataloaders["train"],
        "validation": mb_dataloaders["train"],
        "test": mb_dataloaders["train"],
    }

    mb_model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=mb_dataloaders, seed=42)  # type: ignore
    trainer_config["max_iter"] = 15

    test_score, val_score, output, model_state = trainer(  # type: ignore
        model=mb_model,
        dataloaders=mb_dataloaders,
        seed=1000,
        **trainer_config,
        wandb_logger=None,
    )
    print(f"Moving bar model trained with test correlation: {test_score:.2f}")

    save_model(mb_model, mb_dataloaders, save_folder=cfg.data.save_path, model_name="mb_model")

    # Now can train the session agnostic model

    movie_data_dict = make_final_responses(
        filtered_responses,
        response_type="natural",
        trace_type="spikes",
    )
    dataloaders = hydra.utils.call(
        cfg.dataloader, neuron_data_dictionary=movie_data_dict, movies_dictionary=movies_dict
    )

    ct_model = hydra.utils.call(cfg.model, dataloaders=dataloaders, seed=42, readout_mask_from=mb_model)

    # Specify scheduler kwargs for the trainer
    cfg.trainer.scheduler_kwargs = {
        "max_lr": cfg.trainer.lr_init * 10,
        "epochs": cfg.trainer.max_iter,
        "steps_per_epoch": len(LongCycler(dataloaders["train"])),
    }

    test_score, val_score, output, model_state = hydra.utils.instantiate(
        cfg.trainer,
        model=ct_model,
        dataloaders=dataloaders,
        seed=1000,
    )

    print(f"Session agnostic model trained with test correlation: {test_score:.2f}")

    save_model(ct_model, dataloaders=None, save_folder=cfg.data.save_path, model_name="session_agnostic_model")


if __name__ == "__main__":
    train()  # type: ignore
