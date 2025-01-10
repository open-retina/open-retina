import os
import pickle

import numpy as np

_CURRENT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
_STIMULUS_FOLDER_PATH = os.path.join(_CURRENT_FOLDER_PATH, "data/stimuli")
_CHIRP_PATH = _STIMULUS_FOLDER_PATH + "/RGC_Chirp_30Hz_18_16.pickle"
_CHIRP_72_64_PATH = _STIMULUS_FOLDER_PATH + "/RGC_Chirp_60Hz_72_64.pickle"
_MOVING_BAR_PATH = _STIMULUS_FOLDER_PATH + "/RGC_MovingBar_30Hz_18_16.pickle"
_MOVING_BAR_72_64_PATH = _STIMULUS_FOLDER_PATH + "/RGC_MovingBar_60Hz_72_64.pickle"
_DEFAULT_FRAME_RATE = 30  # Hz

"""
There are three stimuli this module deals with:
- noise stimulus: todo
- chirp stimulus: traces 32.9s, 0.128
- moving bar stimulus: 8 directions, each 3.968s, in total
"""


def normalize_stimulus(stimulus: np.ndarray) -> np.ndarray:
    # max value is 255, min value 0, normalize to range between -1.0 and 1.0
    normalized = (stimulus - 128.0).astype(np.float32) / 128.0
    return normalized


def discretize_triggers(trigger_times: np.ndarray, frame_rate: int = _DEFAULT_FRAME_RATE) -> list[int]:
    assert len(trigger_times.shape) == 1, "trigger_times should be one dimensional"
    discrete_trigger_times = (trigger_times * frame_rate).astype(int).tolist()
    return discrete_trigger_times


def align_stimulus_to_trigger(stimulus: np.ndarray, start_trigger_times: list[int]) -> np.ndarray:
    # TODO: needs fixing!
    stimulus_length = stimulus.shape[0]

    stimulus_array = []
    current_idx = 0
    for trigger_time in start_trigger_times:
        dark_between_time = 0
        if trigger_time > current_idx:
            dark_between_time = trigger_time - current_idx
            dark_shape = (dark_between_time,) + stimulus.shape[1:]
            dark_between = np.zeros(dark_shape)
            stimulus_array.append(dark_between)
        elif trigger_time < current_idx:
            print(f"Warning: trigger_time was smaller than current_idx {trigger_time=} {current_idx=}")
        stimulus_array.append(stimulus)

        current_idx += dark_between_time + stimulus_length

    aligned_stimulus = np.concatenate(stimulus_array)
    # Maybe todo: potential time after stimulus filled with dark, too
    return aligned_stimulus


def load_stimulus(
    file_path: str,
    normalize: bool,
    trigger_times: np.ndarray | None,
    num_triggers_per_repetition: int,
    downsample_t_factor: int = 1,
) -> np.ndarray:
    with open(file_path, "rb") as f:
        stimulus_uint8 = pickle.load(f)

    stimulus = stimulus_uint8.astype(np.float32)
    # only keep two color channels (which ones does not matter as the chirp and moving bars are achromatic)
    stimulus = stimulus[::downsample_t_factor, :, :, :2]

    if trigger_times is not None:
        discrete_trigger_times = discretize_triggers(trigger_times)
        start_trigger_times = discrete_trigger_times[::num_triggers_per_repetition]
        stimulus = align_stimulus_to_trigger(stimulus, start_trigger_times)

    if normalize:
        stimulus = normalize_stimulus(stimulus)

    return stimulus


def load_chirp(
    normalize: bool = True,
    trigger_times: np.ndarray | None = None,
) -> np.ndarray:
    """The chirp has 2 triggers per repetition"""
    chirp = load_stimulus(
        file_path=_CHIRP_PATH,
        normalize=normalize,
        trigger_times=trigger_times,
        num_triggers_per_repetition=2,
    )
    return chirp


def load_chirp_30hz_72_64px(
    normalize: bool = True,
    trigger_times: np.ndarray | None = None,
) -> np.ndarray:
    chirp = load_stimulus(
        file_path=_CHIRP_72_64_PATH,
        normalize=normalize,
        trigger_times=trigger_times,
        num_triggers_per_repetition=2,
        downsample_t_factor=2,
    )
    return chirp


def load_moving_bar(
    normalize: bool = True,
    trigger_times: np.ndarray | None = None,
) -> np.ndarray:
    """Moving bar has 8 triggers, one for each direction.
    Its directions are [0,180, 45,225, 90,270, 135,315] (in degrees)
    """
    moving_bar = load_stimulus(
        file_path=_MOVING_BAR_PATH,
        normalize=normalize,
        trigger_times=trigger_times,
        num_triggers_per_repetition=8,
    )
    # Remove the frames that are not part of the moving bar, i.e. black frames at the beginning and end
    # See: https://github.com/eulerlab/QDSpy/blob/master/Stimuli/RGC_MovingBar_2.py
    frame_rate = 30
    frames_before_first_mb = frame_rate * 3  # 3s before first moving bar
    frames_after_last_mb = frame_rate * 1  # 1s after last moving bar
    moving_bar_content = moving_bar[frames_before_first_mb:-frames_after_last_mb]

    return moving_bar_content


def load_moving_bar_30hz_72_64px(
    normalize: bool = True,
    trigger_times: np.ndarray | None = None,
) -> np.ndarray:
    """Moving bar has 8 triggers, one for each direction.
    Its directions are [0,180, 45,225, 90,270, 135,315] (in degrees)
    """
    moving_bar = load_stimulus(
        file_path=_MOVING_BAR_72_64_PATH,
        normalize=normalize,
        trigger_times=trigger_times,
        num_triggers_per_repetition=8,
        downsample_t_factor=2,
    )
    # Remove the frames that are not part of the moving bar, i.e. black frames at the beginning and end
    # See: https://github.com/eulerlab/QDSpy/blob/master/Stimuli/RGC_MovingBar_2.py
    frame_rate = 30
    frames_before_first_mb = frame_rate * 3  # 3s before first moving bar
    frames_after_last_mb = frame_rate * 1  # 1s after last moving bar
    moving_bar_content = moving_bar[frames_before_first_mb:-frames_after_last_mb]

    return moving_bar_content


def load_moving_bar_stack(normalize: bool = True, number_of_moving_bars: int = 8) -> np.ndarray:
    moving_bar = load_moving_bar(normalize)

    assert moving_bar.shape[0] % number_of_moving_bars == 0, (
        "Moving bar timesteps are not divisible by number_of_moving_bars, something went wrong"
    )
    return np.stack(np.split(moving_bar, number_of_moving_bars, axis=0), axis=0)


def colored_stimulus(
    channel_idx: int,
    pad_front: int,
    stimulus_length: int,
    pad_end: int,
    size_x: int,
    size_y: int,
) -> np.ndarray:
    total_length_time = pad_front + stimulus_length + pad_end
    stimulus = np.zeros((2, total_length_time, size_x, size_y), dtype=np.float32)
    stimulus[channel_idx, pad_front : pad_front + stimulus_length] = 1.0
    stimulus_5d = np.expand_dims(stimulus, 0)

    return stimulus_5d
