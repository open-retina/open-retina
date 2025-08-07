"""
Minimal stimulus loading utilities to train a model on the data used in Maheswaranathan et al. 2023

Paper: https://doi.org/10.1038/s41467-022-33242-8
Data: https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted]

OpenRetina provides a mirror of the dataset on huggingface:
### TODO: Update this link when the dataset is available
"""

import os
from typing import Literal

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import NeuronDataSplit
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
    TO DO: Update this link when the dataset is available
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

        responses_all_sessions[str(session).replace(".h5", "")] = ResponsesTrainTestSplit(
            train=train_session_data / fr_normalization,
            test=test_session_data / fr_normalization,
            stim_id=stim_type,
        )
    return responses_all_sessions
