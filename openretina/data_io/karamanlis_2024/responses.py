"""
Minimal responses loading utilities to train a model on the data used in Karamanlis et al. 2024

Paper: https://doi.org/10.1038/s41586-024-08212-3
Data: https://doi.org/10.12751/g-node.ejk8kx
"""

import os
from typing import Literal

import numpy as np
from einops import rearrange
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.utils.h5_handling import load_dataset_from_h5


def load_all_responses(
    base_data_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"] = "fixationmovie",
    specie: Literal["mouse", "marmoset"] = "mouse",
    fr_normalisation: float = 1.0,
) -> dict[str, ResponsesTrainTestSplit]:
    responses_all_sessions = {}
    exp_sessions = [
        path
        for path in os.listdir(base_data_path)
        if os.path.isdir(os.path.join(base_data_path, path)) and specie in path
    ]

    assert len(exp_sessions) > 0, (
        f"No data directories found in {base_data_path} for animal {specie}."
        "Please check the path and the animal species provided, and that you unrared the data files."
    )

    for session in tqdm(exp_sessions, desc="Processing sessions"):
        session_path = os.path.join(base_data_path, session)

        # Check all files in the session path to identify relevant files
        for recording_file in os.listdir(session_path):
            full_path = os.path.join(session_path, recording_file)

            if str(recording_file).endswith(f"{stim_type}_data.mat"):
                tqdm.write(f"Loading responses from {full_path}")

                testing_responses = load_dataset_from_h5(full_path, "frozenbin")
                training_responses = load_dataset_from_h5(full_path, "runningbin")

                testing_responses = (
                    rearrange(testing_responses, "trials time neurons -> trials neurons time") / fr_normalisation
                )
                mean_test_response = np.mean(testing_responses, axis=0)

                training_responses = (
                    rearrange(training_responses, "block time neurons -> neurons (block time)") / fr_normalisation
                )

            else:
                continue  # Skip session if no relevant file found

            responses_all_sessions["".join(session.split("/")[-1])] = ResponsesTrainTestSplit(
                train=training_responses,
                test=mean_test_response,
                test_by_trial=testing_responses,
                stim_id=stim_type,
            )

    return responses_all_sessions
