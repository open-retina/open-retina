#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from openretina.plotting import save_figure
from openretina.constants import FRAME_RATE_MODEL

from openretina.stimuli import load_moving_bar_stack


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
    mb_stack = load_moving_bar_stack()
    stimulus = torch.Tensor(mb_stack.transpose(0, 4, 1, 2, 3))
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
    # The moving bars "next" to each other are always off by 180 degrees
    # A direction selective cell should strongly respond to one direction, but not its opposing direction
    direction_minus_opposing_direction = np.abs(all_responses[::2] - all_responses[1::2])
    direction_pseudo_index = np.max(np.sum(direction_minus_opposing_direction, axis=1), axis=0)
    sorted_idc = np.argsort(direction_pseudo_index)

    best_neuron_response = all_responses[:, :, sorted_idc[-1]]
    for dir_id in range(all_responses.shape[0]):
        resp = best_neuron_response[dir_id]
        time = 1.0 + np.linspace(0, resp.shape[0] / FRAME_RATE_MODEL, resp.shape[0])
        plt.plot(time, best_neuron_response[dir_id], label=f"Dir{dir_id}")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Response")

    save_figure("ds_neuron.jpg", ".")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
