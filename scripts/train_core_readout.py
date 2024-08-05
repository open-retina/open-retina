#!/usr/bin/env python3

import argparse
import os
import pickle

import torch
import lightning
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from openretina.cyclers import LongCycler
from openretina.hoefling_2024.data_io import (
    natmov_dataloaders_v2,
)
from openretina.neuron_data_io import make_final_responses
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.models.core_readout import CoreReadout


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default = "cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main(
    data_folder: str,
    save_folder: str,
    device: str,
) -> None:
    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging.h5")
    responses = load_h5_into_dict(data_path_responses)

    data_dict = make_final_responses(responses, response_type="natural")  # type: ignore
    dataloaders = natmov_dataloaders_v2(data_dict, movies_dictionary=movies_dict, train_chunk_size=100, seed=1000)

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), batch_size=None, num_workers=0)
    test_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), batch_size=None, num_workers=0)

    torch.Size([1, 2, 750, 18, 16])
    torch.Size([1, 750, 131])

    n_neurons_dict = {
       name: data_point.targets.shape[-1] for name, data_point in iter(train_loader)
    }
    model = CoreReadout(
        in_channels=2,
        features_core=(8, 8, 8),
        in_shape=(2, 750, 18, 16),
        n_neurons_dict=n_neurons_dict,
        scale=True,
        bias=True,
        gaussian_masks = True,
        gaussian_mean_scale=6.0,
        gaussian_var_scale=4.0,
        positive=True,
        gamma_readout=0.4,
    )

    lightning_folder = f"{save_folder}"
    os.makedirs(lightning_folder, exist_ok=True)
    csv_logger = CSVLogger(lightning_folder)
    tensorboard_logger = TensorBoardLogger(f"models/tensorboard", name="foo")
    trainer = lightning.Trainer(max_epochs=50, default_root_dir=lightning_folder,
                                logger=[csv_logger, tensorboard_logger])
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
    test_res = trainer.test(model, dataloaders=[train_loader, valid_loader, test_loader])
    print(test_res)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
