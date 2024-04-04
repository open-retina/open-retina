#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from openretina.plotting import save_figure

from openretina.stimuli import load_moving_bar


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--model_path", type=str,
                        required=True, help="Path to the saved model")
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


def main(model_path: str, device: str) -> None:
    model = torch.load(model_path, map_location=torch.device(device))
    print(f"Initialized model from {model_path=}")
    # Begin evaluation
    model.eval()
    stimulus = torch.Tensor(load_moving_bar())
    if args.device == "cuda":
        model.cuda()
        stimulus.cuda()
    else:
        model.cpu()
        stimulus.cpu()
        # for feature in model.core.features:
        #    feature.conv.device = "cpu"

    all_responses_array = []
    for session_id in model.readout.keys():
        with torch.no_grad():
            # responses.shape: directions, time_steps, neurons
            responses = model.forward(stimulus, data_key=session_id)
            all_responses_array.append(responses.cpu().numpy())
    all_responses = np.concatenate(all_responses_array, axis=-1)
    score_per_direction = np.max(all_responses, axis=1)
    direction_pseudo_index = np.max(score_per_direction, axis=0) / np.min(score_per_direction, axis=0)
    sorted_idc = np.argsort(direction_pseudo_index)

    best_neuron_response = all_responses[:, :, sorted_idc[-1]]
    for dir_id in range(all_responses.shape[0]):
        plt.plot(best_neuron_response[dir_id], label=f"Dir{dir_id}")
    plt.legend()

    save_figure("ds_neuron.pdf", ".")
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
