import argparse
import os
import random

import h5py
import numpy as np


def add_parser_arguments(parser: argparse.ArgumentParser):
    parser.description = "Create artificial data to use for testing openretina and as an example to add new datasets"

    parser.add_argument(
        "directory",
        type=str,
        help="Directory to write test data to",
    )

    parser.add_argument(
        "--num_colors",
        type=int,
        default=3,
        help="Color channels in stimulus",
    )

    parser.add_argument(
        "--num_stimuli",
        type=int,
        default=3,
        help="Color channels in stimulus",
    )

    parser.add_argument(
        "--num_sessions",
        type=int,
        default=2,
        help="Number of sessions to generate",
    )


def _generate_response(stimulus: np.ndarray) -> np.ndarray:
    """Sum over the stimulus in an area and multiply this sum by a scale and add a bias"""
    num_colors, time_steps, h, w = stimulus.shape
    color_idx = random.randrange(num_colors)
    receptive_field_width = random.randint(1, 5)
    h_loc = random.randrange(0, h)
    w_loc = random.randrange(0, w)
    stim_rec_field = stimulus[
        color_idx, :, h_loc : h_loc + receptive_field_width, w_loc : w_loc + receptive_field_width
    ]
    resp = np.sum(stim_rec_field, axis=-1).sum(axis=-1)
    scale = 0.5 + random.random()
    bias = random.random()
    resp_scaled = resp * scale + bias
    return resp_scaled


def write_data_to_directory(directory: str, num_colors: int, num_stimuli: int, num_sessions: int):
    stimulus_shape = (num_colors, 40, 16, 16)
    neurons_per_session = np.random.choice(np.arange(20) + 10, size=num_sessions)

    os.makedirs(directory, exist_ok=True)

    stimuli_folder = os.path.join(directory, "stimuli")
    os.makedirs(stimuli_folder, exist_ok=True)

    stimuli_map = {}
    for i in range(num_stimuli):
        stim_name = f"random_noise_{i}"
        stim_path = os.path.join(stimuli_folder, stim_name)
        rand_noise = np.random.randn(*stimulus_shape)
        stimuli_map[stim_name] = rand_noise
        np.save(stim_path, rand_noise, allow_pickle=False)

    # generate responses
    for session_id, num_neurons in enumerate(neurons_per_session):
        session_path = os.path.join(directory, f"session_{session_id}.hdf5")
        with h5py.File(session_path, "w") as f:
            for stim_name, stim in stimuli_map.items():
                responses = np.stack([_generate_response(stim) for _ in range(num_neurons)])
                resp_name = f"responses_{stim_name}"
                f[resp_name] = responses
            f["session_info/celltypes"] = np.random.choice(40, size=num_neurons)
            print(f"Wrote the following entries to the file {session_path}: {list(f.keys())}")
