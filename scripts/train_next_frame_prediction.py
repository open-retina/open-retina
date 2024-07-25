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
from openretina.models.next_frame_prediction import NextFramePredictionModel, TorchSTSeparableConv3D


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

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

    channel_sizes = [2, 16, 2]
    conv_models = []
    for in_channels, out_channels in zip(channel_sizes, channel_sizes[1:]):
        conv_models.append(
            TorchSTSeparableConv3D(in_channels, out_channels, 15, (9, 9)))
    reconstruction_model = torch.nn.Sequential(*conv_models)

    next_frame_pred_model = NextFramePredictionModel(reconstruction_model)

    # when num_workers > 0 the docker container needs more shared memory
    train_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["train"], shuffle=True), batch_size=None, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["validation"], shuffle=False), batch_size=None, num_workers=0)
    test_loader = torch.utils.data.DataLoader(LongCycler(dataloaders["test"], shuffle=False), batch_size=None, num_workers=0)

    lightning_folder = f"{save_folder}_next_frame_pred"
    os.makedirs(lightning_folder)
    csv_logger = CSVLogger(lightning_folder)
    tensorboard_logger = TensorBoardLogger("models/tensorboard", name="next_frame_pred")
    trainer = lightning.Trainer(max_epochs=40, default_root_dir=lightning_folder,
                                logger=[csv_logger, tensorboard_logger])
    trainer.fit(model=next_frame_pred_model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
    test_res = trainer.test(next_frame_pred_model, dataloaders=[train_loader, valid_loader, test_loader])
    print(test_res)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
