#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from openretina.hoefling_2024.constants import FRAME_RATE_MODEL
from openretina.plotting import save_figure
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.stimuli import load_moving_bar_stack


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_folder", type=str, default=None, help="Path to the base data folder")

    return parser.parse_args()


def main(model_path: str, device: str, data_folder: str | None) -> None:
    model = torch.load(model_path, map_location=torch.device(device))
    print(f"Initialized model from {model_path=}")
    # Begin evaluation
    model.eval()
    mb_stack = load_moving_bar_stack()
    stimulus = torch.Tensor(mb_stack.transpose(0, 4, 1, 2, 3))
    model.to(args.device)
    model.to(args.device)

    all_responses_array = []
    session_id_list = list(model.readout.keys())
    for session_id in session_id_list:
        with torch.no_grad():
            # responses.shape: directions, time_steps, neurons
            responses = model.forward(stimulus, data_key=session_id)
            all_responses_array.append(responses.cpu().numpy())
    all_responses = np.concatenate(all_responses_array, axis=-1)
    # The moving bars "next" to each other are always off by 180 degrees
    # A direction selective cell should strongly respond to one direction, but not its opposing direction
    direction_minus_opposing_direction = np.abs(all_responses[::2] - all_responses[1::2])
    direction_pseudo_index = np.max(np.sum(direction_minus_opposing_direction, axis=1), axis=0)

    print(f"Avg {np.average(direction_pseudo_index)=} {direction_pseudo_index.max()} {direction_pseudo_index.min()}")

    min_confidence = 0.25
    if data_folder is not None:
        data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging.h5")
        ground_truth_responses = load_h5_into_dict(data_path_responses)
        gt_celltype_assignments_array: list[np.ndarray] = []
        for session_id in session_id_list:
            cut_session_id = session_id.replace("_mb", "").replace("_chirp", "")
            gt = ground_truth_responses[cut_session_id]
            gt_celltype = gt["group_assignment"]
            gt_mask = gt["group_confidences"].max(axis=1) >= min_confidence
            gt_celltype_masked = gt_celltype * gt_mask
            if cut_session_id == "session_2_ventral1_20200226":
                print(f"{gt_celltype[101]=} {sum(x.shape[0] for x in gt_celltype_assignments_array)}")
            gt_celltype_assignments_array.append(gt_celltype_masked)

    # This is date: 2020-02-26, exp_num=1, field=GCL1, roi_id=102, and has type 26
    scid = 2347
    desired_celltype_indices = [scid]
    best_neuron_response = all_responses[:, :, desired_celltype_indices[0]]
    for idx in desired_celltype_indices:
        print(f"{idx}: {direction_pseudo_index[idx]}")
    DIRS_DEGREE = [0, 180, 45, 225, 90, 270, 135, 315]
    for dir_id in range(all_responses.shape[0]):
        resp = best_neuron_response[dir_id]
        time = 1.0 + np.linspace(0, resp.shape[0] / FRAME_RATE_MODEL, resp.shape[0])
        plt.plot(time, best_neuron_response[dir_id], label=f"Dir {DIRS_DEGREE[dir_id]}°")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Response")

    save_figure("ds_neuron.jpg", ".")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
