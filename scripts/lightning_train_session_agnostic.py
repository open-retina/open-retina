import os
import pickle

import hydra
import lightning
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf

from openretina.cyclers import LongCycler
from openretina.hoefling_2024.data_io import get_mb_dataloaders
from openretina.neuron_data_io import filter_responses, make_final_responses
from openretina.training import save_model
from openretina.utils.h5_handling import load_h5_into_dict


@hydra.main(config_path="../configs", config_name="minimal_session_agnostic_lightning")
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

    mb_train_loader = data.DataLoader(LongCycler(mb_dataloaders["train"], shuffle=True), batch_size=None, num_workers=0)

    # Assign missing n_neurons_dict to the model
    cfg.mb_model.n_neurons_dict = {name: data_point.targets.shape[-1] for name, data_point in iter(mb_train_loader)}

    mb_model = hydra.utils.instantiate(cfg.mb_model)

    mb_trainer = lightning.Trainer(
        max_epochs=15,
        default_root_dir=cfg.data.save_path,
    )
    mb_trainer.fit(model=mb_model, train_dataloaders=mb_train_loader, val_dataloaders=mb_train_loader)

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

    train_loader = data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0)
    valid_loader = data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), batch_size=None, num_workers=0)
    test_loader = data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), batch_size=None, num_workers=0)

    # Specify scheduler kwargs for the trainer
    cfg.scheduler.steps_per_epoch = len(LongCycler(dataloaders["train"]))

    # These are instantiated with partial here. Will get params inside the model
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    scheduler = hydra.utils.instantiate(cfg.scheduler)

    ct_model = hydra.utils.instantiate(
        cfg.model,
        readout_mask_from=mb_model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Logging
    log_folder = os.path.join(cfg.data.save_path, cfg.exp_name)
    os.makedirs(log_folder, exist_ok=True)
    logger_array = []
    for logger_name, logger_params in cfg.loggers.items():
        logger = hydra.utils.instantiate(logger_params, save_dir=log_folder, name=logger_name)
        logger_array.append(logger)

    callbacks = [
        hydra.utils.instantiate(callback_params) for callback_params in cfg.get("training_callbacks", {}).values()
    ]
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger_array, callbacks=callbacks)
    trainer.fit(model=ct_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Test the model
    trainer.test(ct_model, dataloaders=[train_loader, valid_loader, test_loader])
    trainer.save_checkpoint(os.path.join(log_folder, "model.ckpt"))
    save_model(ct_model, dataloaders=None, save_folder=cfg.data.save_path, model_name="session_agnostic_model")


if __name__ == "__main__":
    train()  # type: ignore
