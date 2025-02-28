import argparse
import os

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
        "--num-colors",
        type=int,
        default=3,
        help="Color channels in stimulus",
    )

    parser.add_argument(
        "--num-stimuli",
        type=int,
        default=4,
        help="Color channels in stimulus",
    )

    parser.add_argument(
        "--num-sessions",
        type=int,
        default=2,
        help="Number of sessions to generate",
    )


def _generate_response(stimuli: list[np.ndarray], num_neurons: int) -> list[np.ndarray]:
    """Sum over the stimulus in a different area for each neuron and multiply this sum by a scale and add a bias"""
    stimulus_shapes = set((s.shape[0], s.shape[2], s.shape[3]) for s in stimuli)
    assert len(stimulus_shapes) == 1
    stimulus_shape = next(iter(stimulus_shapes))

    num_colors, h, w = stimulus_shape

    # first generate neuron scale, bias and receptive field locations
    scale = np.random.random(num_neurons) + 0.5
    bias = np.random.random(num_neurons)
    color_idc = np.random.randint(0, num_colors, size=num_neurons)
    receptive_field_widths = np.random.randint(1, 6, size=num_neurons)
    h_locations = np.random.randint(0, h, size=num_neurons)
    w_locations = np.random.randint(0, w, size=num_neurons)

    responses = []
    for stim in stimuli:
        resp_neurons = []
        for i in range(num_neurons):
            stim_rec_field = stim[
                color_idc[i],
                :,
                h_locations[i] : h_locations[i] + receptive_field_widths[i],
                w_locations[i] : w_locations[i] + receptive_field_widths[i],
            ]
            resp = np.sum(stim_rec_field, axis=-1).sum(axis=-1)
            resp_neurons.append(resp)
        resp_neurons_np = np.stack(resp_neurons)
        resp_neurons_np = resp_neurons_np * scale[:, np.newaxis] + bias[:, np.newaxis]
        responses.append(resp_neurons_np)

    return responses


def write_data_to_directory(directory: str, num_colors: int, num_stimuli: int, num_sessions: int):
    stimulus_shape_array = [(num_colors, int(t * 120), 16, 8) for t in np.arange(30, 30 + num_stimuli)]
    neurons_per_session = np.random.choice(np.arange(20) + 10, size=num_sessions)

    os.makedirs(directory, exist_ok=True)

    stimuli_folder = os.path.join(directory, "stimuli")
    os.makedirs(stimuli_folder, exist_ok=True)

    name_stimulus_list: list[tuple[str, np.ndarray]] = []
    for i, stimulus_shape in enumerate(stimulus_shape_array):
        stim_name = f"random_noise_{i}"
        stim_path = os.path.join(stimuli_folder, stim_name)
        rand_noise = np.random.randn(*stimulus_shape)
        name_stimulus_list.append((stim_name, rand_noise))
        np.save(stim_path, rand_noise, allow_pickle=False)
    print(f"Wrote the following files to the folder {stimuli_folder}: {[x[0] + '.npy' for x in name_stimulus_list]}")

    # generate responses
    for session_id, num_neurons in enumerate(neurons_per_session):
        session_path = os.path.join(directory, f"session_{session_id}.hdf5")
        stimuli = [x[1] for x in name_stimulus_list]
        responses = _generate_response(stimuli, num_neurons)
        with h5py.File(session_path, "w") as f:
            for (stim_name, stim), resp in zip(name_stimulus_list, responses, strict=True):
                resp_name = f"responses_{stim_name}"
                f[resp_name] = resp
            f["session_info/celltypes"] = np.random.choice(40, size=num_neurons)
            print(f"Wrote the following entries to the file {session_path}: {list(f.keys())}")
