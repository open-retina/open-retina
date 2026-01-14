"""
Minimal stimulus loading utilities to train a model on the data used in Goldin et al. 2022.

Paper: https://doi.org/10.1038/s41467-022-33242-8
Data: https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted]

OpenRetina provides a mirror of the dataset on huggingface:
https://huggingface.co/datasets/open-retina/open-retina/tree/main/marre_lab/goldin_2022
"""

import os
from typing import Literal

import h5py
import numpy as np
from einops import rearrange

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_dataset_from_h5


def load_all_responses(
    base_data_path: str | os.PathLike,
    response_type: Literal["firing_rate_300ms"] = "firing_rate_300ms",
    stim_type: Literal["naturalscene"] = "naturalscene",
    specie: Literal["mouse", "axolotl"] = "mouse",
    fr_normalization: float = 1,
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Load all stimuli from sessions within subfolders in a given base data path.

    The base data path should point to the location of the `neural_code_data` folder.
    (See https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted] for dataset download).

    Alternatively, base_data_path can point directly to our huggingface mirror of the dataset, which will then
    be downloaded and extracted automatically to the openretina cache directory.
    """
    # Resolve data path
    base_data_path = get_local_file_path(str(base_data_path))
    full_data_path = base_data_path

    responses_all_sessions = {}
    for session in [x.name for x in os.scandir(os.fspath(full_data_path))]:
        if specie not in session:
            continue
        session_path = os.path.join(full_data_path, session)
        recording_file = session_path

        print(f"Loading responses from {recording_file}")

        # Load neural responses
        train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")
        test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")

        assert train_session_data.shape[0] == test_session_data.shape[0], (
            "Train and test responses should have the same number of neurons."
        )

        test_by_trial = load_test_repeats_for_session(recording_file, fr_normalization)

        responses_all_sessions[str(session).removesuffix(".h5")] = ResponsesTrainTestSplit(
            train=train_session_data / fr_normalization,
            test=test_session_data / fr_normalization,
            test_by_trial=test_by_trial,
            stim_id=stim_type,
        )
    return responses_all_sessions


def load_test_repeats_for_session(
    session_path: str | os.PathLike,
    fr_normalization: float = 1,
) -> np.ndarray | None:
    """
    Load repeated test responses stored under /test/repeats/cell_{idx}.

    Returns repeats x neurons x time or None if no repeats are present.
    """
    with h5py.File(session_path, "r") as f:
        test_group = f.get("test")
        if not isinstance(test_group, h5py.Group):
            return None
        repeats_group = test_group.get("repeats")
        if not isinstance(repeats_group, h5py.Group):
            return None

        repeat_groups = [k for k in repeats_group.keys() if "cell" in k]
        if len(repeat_groups) == 0:
            return None

        # Each dataset is expected to be shape (repeats, time)
        repeat_groups_sorted = sorted(repeat_groups, key=lambda x: int(x.split("cell")[-1]))
        stacked_list: list[np.ndarray] = []
        for cell_key in repeat_groups_sorted:
            cell_dataset = repeats_group.get(cell_key)
            if not isinstance(cell_dataset, h5py.Dataset):
                continue
            cell_repeats = cell_dataset[...]
            stacked_list.append(cell_repeats)

        if len(stacked_list) == 0:
            return None

        # shape: neurons x repeats x time -> repeats x neurons x time
        stacked = np.stack(stacked_list, axis=0) / fr_normalization
        stacked = rearrange(stacked, "neurons repeats time -> repeats neurons time")
        return stacked
