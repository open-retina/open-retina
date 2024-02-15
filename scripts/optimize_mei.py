#!/usr/bin/env python3

import os
import pickle
from functools import partial

import torch
from openretina.hoefling_2022_configs import model_config
from openretina.hoefling_2022_data_io import natmov_dataloaders_v2
from openretina.hoefling_2022_models import SFB3d_core_SxF3d_readout

from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.objective import SingleNeuronObjective


def main() -> None:
    data_folder = "/gpfs01/euler/data/SharedFiles/projects/TP12/"
    data_path = os.path.join(data_folder, "2024-01-11_neuron_data_stim_8c18928_responses_99c71a0.pkl")
    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(data_path, "rb") as f:
        neuron_data_dict = pickle.load(f)

    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    dataloaders = natmov_dataloaders_v2(neuron_data_dict, movies_dict)
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)
    state_dict_path = "model_state_dict.tmp"
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    print(f"Init model from {state_dict_path=}")

    objective = SingleNeuronObjective(model, neuron_idx=0)
    stimulus = torch.rand(1, 3, 64, 64).cuda()
    optimizer_init_fn = partial(torch.optim.SGD, lr=100.0)
    # Throws: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    # reason probably: not all model parameters are on gpu(?)
    optimize_stimulus(
        stimulus,
        optimizer_init_fn,
        objective,
        stimulus_regularizing_fn=None,
    )



if __name__ == "__main__":
    main()

