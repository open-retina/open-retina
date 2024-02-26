import os

import h5py as h5
import numpy as np

from .misc import load_dataset_from_h5


def load_all_sessions(base_data_path, response_type="firing_rate_20ms", stim_type="naturalscene", fr_normalization=1):
    """
    Load all sessions from subfolders in a given base data path
    """
    common_length_train = -1
    common_length_test = -1
    skip_sessions = []
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
            if str(recording_file).endswith(f"{stim_type}.h5"):

                recording_file = os.path.join(session_path, recording_file)

                # Load video stimuli
                train_video = load_dataset_from_h5(recording_file, "/train/stimulus")
                test_video = load_dataset_from_h5(recording_file, "/test/stimulus")

                if "train" in video_stimuli:
                    common_length_train = min(common_length_train, train_video.shape[0])
                    if not np.allclose(video_stimuli["train"][:, :common_length_train].squeeze(), train_video[:common_length_train]):  # type: ignore
                        print(f"Train videos are different across sessions, skipping {session}")
                        skip_sessions.append(session)
                        continue
                        # raise ValueError("Train videos are different across sessions")
                else:
                    common_length_train = train_video.shape[0]
                video_stimuli["train"] = train_video[None, :common_length_train, ...]  # type: ignore

                if "test" in video_stimuli:
                    common_length_test = min(common_length_test, test_video.shape[0])
                    if not np.allclose(video_stimuli["test"][:, :common_length_test].squeeze(), test_video[:common_length_test]):  # type: ignore
                        # raise ValueError("Test videos are different across sessions")
                        raise ValueError("Test videos are different across sessions")
                else:
                    common_length_test = test_video.shape[0]
                video_stimuli["test"] = test_video[None, :common_length_test, ...]  # type: ignore

        # Load responses after common length is determined
        for recording_file in os.listdir(
            session_path,
        ):
            if str(recording_file).endswith(f"{stim_type}.h5") and session not in skip_sessions:
                recording_file = os.path.join(session_path, recording_file)
                print(f"Loading data from {recording_file}")
                train_session_data = load_dataset_from_h5(recording_file, f"/train/response/{response_type}")[:, :common_length_train]  # type: ignore
                test_session_data = load_dataset_from_h5(recording_file, f"/test/response/{response_type}")[:, :common_length_test]  # type: ignore
                responses_all_sessions["".join(session.split("/")[-1])] = {
                    "responses_final": {
                        "train": train_session_data / fr_normalization,  # type: ignore
                        "test": test_session_data / fr_normalization,  # type: ignore
                    },
                    "stim_id": "salamander_natural",
                }

    return responses_all_sessions, video_stimuli


def find_first_different_frame(video1, video2):
    # Ensure the videos have the same shape
    assert video1.shape == video2.shape, "Videos must have the same shape."

    # Compare the videos to get a boolean array: True where elements match, False otherwise
    comparison = np.equal(video1, video2)

    # Reduce the comparison to check if all elements in each frame are True (equal)
    frames_equal = np.all(comparison, axis=(1, 2))

    # Find the first frame that is not entirely equal (where frames_equal is False)
    different_frame_index = np.where(~frames_equal)[0]

    # Check if there is at least one frame that is different and return its index
    if different_frame_index.size > 0:
        return different_frame_index[0]
    else:
        return None
