#!/usr/bin/env python3

import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import torch
from openretina.hoefling_2024.configs import model_config
from openretina.hoefling_2024.data_io import natmov_dataloaders_v2
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
from openretina.optimization.objective import SingleNeuronObjective
from openretina.optimization.optimizer import optimize_stimulus
from openretina.optimization.optimization_stopper import OptimizationStopper
from openretina.optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    RangeRegularizationLoss,
)
from openretina.plotting import plot_stimulus_composition


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

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=dataloaders, seed=42)  # type: ignore
    state_dict_path = "model_state_dict.tmp"
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.cuda()
    print(f"Init model from {state_dict_path=}")

    device = "cuda"
    # from controversial stimuli: (2, 50, 18, 16): (channels, time, height, width)
    stimulus_shape = (1, 2, 50, 18, 16)

    for session_id in model.readout.keys():
        for neuron_id in range(model.readout[session_id].outdims):
            print(f"Generating MEI for {session_id=} {neuron_id=}")
            objective = SingleNeuronObjective(model, neuron_idx=neuron_id, data_key=session_id)
            stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
            stimulus_postprocessor = ChangeNormJointlyClipRangeSeparately()
            stimulus.data = stimulus_postprocessor.process(stimulus.data)
            optimizer_init_fn = partial(torch.optim.SGD, lr=10.0)
            stimulus_regularizing_loss = RangeRegularizationLoss()
            # Throws: RuntimeError: Expected all tensors to be on the same device,
            # but found at least two devices, cuda:0 and cpu!
            # reason probably: not all model parameters are on gpu(?)
            optimize_stimulus(
                stimulus,
                optimizer_init_fn,
                objective,
                OptimizationStopper(max_iterations=100),
                stimulus_regularization_loss=stimulus_regularizing_loss,
                stimulus_postprocessor=stimulus_postprocessor,
            )
            stimulus_np = stimulus[0].cpu().numpy()
            fig, axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
            plot_stimulus_composition(
                stimulus=stimulus_np,
                temporal_trace_ax=axes[0, 0],
                freq_ax=axes[0, 1],
                spatial_ax=axes[1, 0],
                highlight_x_list=[(40, 49)],
            )
            img_path = f"meis/mei_{session_id}_{neuron_id}.pdf"
            fig.savefig(img_path, bbox_inches="tight", facecolor="w", dpi=300)


if __name__ == "__main__":
    main()
