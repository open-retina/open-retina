"""
Minimal responses loading utilities to train a model on the data used in Karamanlis et al. 2024

Paper: https://doi.org/10.1038/s41586-024-08212-3
Data: https://doi.org/10.12751/g-node.ejk8kx
"""

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from einops import rearrange
from tqdm.auto import tqdm

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path, unzip_and_cleanup
from openretina.utils.h5_handling import load_dataset_from_h5


def load_responses_for_session(
    session_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"],
    fr_normalisation: float,
) -> ResponsesTrainTestSplit | None:
    """
    Load responses for a single session.

    Args:
        session_path (str): Path to the session directory.
        stim_type (str): The stimulus type to filter files.
        fr_normalisation (float): Normalization factor for firing rates.

    Returns:
        ResponsesTrainTestSplit: Loaded responses for the session.

    Raises:
        IOError: If multiple relevant files are found.
    """
    if str(session_path).endswith(".zip"):
        session_path = unzip_and_cleanup(Path(session_path))
    recording_file_names = [x for x in os.listdir(session_path) if x.endswith(f"{stim_type}_data.mat")]

    if len(recording_file_names) == 0:
        warnings.warn(
            f"No file with postfix {f'{stim_type}_data.mat'} found in folder {session_path}", UserWarning, stacklevel=2
        )
        return None
    elif len(recording_file_names) > 1:
        raise IOError(
            f"Multiple files with postfix {f'{stim_type}_data.mat'} found in folder {session_path}: "
            f"{recording_file_names=}"
        )

    full_path = os.path.join(session_path, recording_file_names[0])
    tqdm.write(f"Loading responses from {full_path}")

    testing_responses = load_dataset_from_h5(full_path, "frozenbin")
    training_responses = load_dataset_from_h5(full_path, "runningbin")

    testing_responses = rearrange(testing_responses, "trials time neurons -> trials neurons time") / fr_normalisation
    mean_test_response = np.mean(testing_responses, axis=0)

    training_responses = rearrange(training_responses, "block time neurons -> neurons (block time)") / fr_normalisation

    return ResponsesTrainTestSplit(
        train=training_responses,
        test=mean_test_response,
        test_by_trial=testing_responses,
        stim_id=stim_type,
    )


def load_all_responses(
    base_data_path: str | os.PathLike,
    stim_type: Literal["fixationmovie", "frozencheckerflicker", "gratingflicker", "imagesequence"] = "fixationmovie",
    specie: Literal["mouse", "marmoset"] = "mouse",
    fr_normalization: float = 1.0,
) -> dict[str, ResponsesTrainTestSplit]:
    """
    Load responses for all sessions.

    Args:
        base_data_path (str | os.PathLike): Base directory containing session data.
                                            Can also be the path to the "sessions" folder in the huggingface mirror.
        "https://huggingface.co/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024/sessions"
        stim_type (str): The stimulus type to filter files.
        specie (str): Animal species (e.g., "mouse", "marmoset").
        fr_normalization (float): Normalization factor for firing rates.

    Returns:
        dict[str, ResponsesTrainTestSplit]: Dictionary mapping session names to response data.
    """
    base_data_path = get_local_file_path(str(base_data_path))

    responses_all_sessions = {}
    exp_sessions = [
        path
        for path in os.listdir(base_data_path)
        if (os.path.isdir(os.path.join(base_data_path, path)) or path.endswith(".zip")) and specie in path
    ]

    assert len(exp_sessions) > 0, (
        f"No data directories found in {base_data_path} for animal {specie}."
        "Please check the path and the animal species provided, and that you unrared the data files."
    )

    for session in tqdm(exp_sessions, desc="Processing sessions"):
        session_path = os.path.join(base_data_path, session)
        responses = load_responses_for_session(session_path, stim_type, fr_normalization)
        if responses:
            responses_all_sessions[str(os.path.basename(os.path.normpath(session)))] = responses

    return responses_all_sessions
