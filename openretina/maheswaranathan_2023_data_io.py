import os

import h5py as h5
import numpy as np

from .misc import load_dataset_from_h5


def load_all_sessions(base_data_path, response_type="firing_rate_20ms", stim_type="naturalscene", fr_normalization=1):
    """
    Load all sessions from subfolders in a given base data path
    """
    responses_all_sessions = {}
    video_stimuli = {}
    for session in os.listdir(base_data_path):
        # List sessions in the base data path
        session_path = os.path.join(base_data_path, session)
        if not os.path.isdir(session_path):
            continue
        for recording_file in os.listdir(
            session_path,
        ):
            print(f"Recording file: {str(recording_file)}")
            if str(recording_file).endswith(f"{stim_type}.h5"):
                recording_file = os.path.join(session_path, recording_file)
                print(f"Loading data from {recording_file}")
                train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")
                test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")
                responses_all_sessions["".join(session.split("/")[-1])] = {
                    "responses_final": {
                        "train": train_session_data / fr_normalization,
                        "test": test_session_data / fr_normalization,
                    }
                }

                # Load video stimuli
                train_video = load_dataset_from_h5(recording_file, "/train/stimulus")
                test_video = load_dataset_from_h5(recording_file, "/test/stimulus")
                if "train" in video_stimuli:
                    if not np.array_equal(video_stimuli["train"], train_video):
                        raise ValueError("Train videos are different across sessions")
                else:
                    video_stimuli["train"] = train_video
                if "test" in video_stimuli:
                    if not np.array_equal(video_stimuli["test"], test_video):
                        raise ValueError("Test videos are different across sessions")
                else:
                    video_stimuli["test"] = test_video

    return responses_all_sessions, video_stimuli
