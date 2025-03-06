"""
Minimal responses loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1016/j.neuron.2023.06.007
Data: https://doi.org/10.25740/rk663dm5577

OpenRetina provides a mirror of the dataset on huggingface:
https://huggingface.co/datasets/open-retina/open-retina/resolve/main/baccus_lab/maheswaranathan_2023/neural_code_data.zip
"""

import os
from typing import Literal

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import NeuronDataSplit
from openretina.utils.file_utils import get_local_file_path
from openretina.utils.h5_handling import load_dataset_from_h5


def load_all_responses(
    base_data_path: str | os.PathLike,
    response_type: Literal["firing_rate_5ms", "firing_rate_10ms", "firing_rate_20ms"] = "firing_rate_20ms",
    stim_type: Literal["naturalscene", "whitenoise"] = "naturalscene",
    fr_normalization: float = 1,
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Load all neural responses from sessions within subfolders in a given base data path.

    The base data path should point to the location of of the `neural_code_data` folder.
    (See https://doi.org/10.25740/rk663dm5577 for dataset download).

    Alternatively, base_data_path can point directly to our huggingface mirror of the dataset, which will then
    be downloaded and extracted automatically to the openretina cache directory.
    https://huggingface.co/datasets/open-retina/open-retina/resolve/main/baccus_lab/maheswaranathan_2023/neural_code_data.zip
    """
    # Resolve data path
    base_data_path = get_local_file_path(str(base_data_path))
    full_data_path = os.path.join(base_data_path, "ganglion_cell_data")

    responses_all_sessions = {}
    for session in [x.name for x in os.scandir(os.fspath(full_data_path)) if x.is_dir()]:
        session_path = os.path.join(full_data_path, session)
        for recording_file in [x for x in os.listdir(session_path) if str(x).endswith(f"{stim_type}.h5")]:
            recording_file = os.path.join(session_path, recording_file)

            print(f"Loading responses from {recording_file}")

            # Load neural responses
            train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")
            test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")

            assert train_session_data.shape[0] == test_session_data.shape[0], (
                "Train and test responses should have the same number of neurons."
            )

            responses_all_sessions[str(session)] = ResponsesTrainTestSplit(
                train=train_session_data / fr_normalization,
                test=test_session_data / fr_normalization,
                stim_id=stim_type,
            )
    return responses_all_sessions


class NeuronDataBaccus(NeuronDataSplit):
    pass
