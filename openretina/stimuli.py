import numpy as np

_ORIGINAL_STIMULI_FOLDER = "/gpfs01/berens/data/data/optimalChirp/stimuli/original"
# _ORIGINAL_STIMULI_FOLDER = "/home/tzenkel/Data/Stimuli"
_MOVING_BAR_STIMULUS_PATH = f"{_ORIGINAL_STIMULI_FOLDER}/mb_array_range_0_1__8_110_18_16_corrected.npy"
_CHIRP_STIMULUS_PATH = f"{_ORIGINAL_STIMULI_FOLDER}/chirp_stim.npy"

_MEAN = 36.774321197604586  # from MovieStimulus.fetch("stimulus_info")
_STD = 34.811117676511635  # from MovieStimulus.fetch("stimulus_info")

_LENGTH_X = 18
_LENGTH_Y = 16


def load_chirp(
        file_path: str = _CHIRP_STIMULUS_PATH,
        receptive_field_ratio: float = 1.0,
        use_predefined_normalization: bool = False
) -> np.array:
    assert 0 < receptive_field_ratio <= 1.0

    chirp_stim = np.load(file_path)

    if use_predefined_normalization:
        mean, std = _MEAN, _STD
    else:
        mean, std = chirp_stim.mean(), chirp_stim.std()
    chirp_stim_normalized = (chirp_stim - mean) / std

    chirp_3d = np.expand_dims(chirp_stim_normalized, (1, 2))
    chirp_4d = np.expand_dims(chirp_3d, 0)
    chirp_4d_full = np.repeat(chirp_4d, 2, axis=0)
    chirp_4d_full = np.tile(chirp_4d_full, (1, 1, _LENGTH_X, _LENGTH_Y))
    chirp_5d = np.expand_dims(chirp_4d_full, 0)
    chirp_5d_f32 = chirp_5d.astype(np.float32)

    if receptive_field_ratio < 1:
        r_half = chirp_5d_f32.shape[-2] // 2
        c_half = chirp_5d_f32.shape[-1] // 2
        r_d = int((chirp_5d_f32.shape[-2] * receptive_field_ratio) // 2)  # // 2 as r_d goes in both directions
        c_d = int((chirp_5d_f32.shape[-1] * receptive_field_ratio) // 2)
        z = np.zeros_like(chirp_5d_f32)
        z[:, :, :, r_half - r_d:r_half + r_d, c_half - c_d : c_half + c_d] = chirp_5d_f32[:, :, :, r_half - r_d:r_half + r_d, c_half - c_d : c_half + c_d]
        chirp_5d_f32 = z

    return chirp_5d_f32


def load_moving_bar(file_path: str = _MOVING_BAR_STIMULUS_PATH) -> np.array:
    mb_stack = np.load(file_path)
    mb_stack_5d = np.repeat(np.expand_dims(mb_stack, 1), 2, axis=1)
    mb_stack_5d_f32 = mb_stack_5d.astype(np.float32)

    return mb_stack_5d_f32


def colored_stimulus(channel_idx: int, pad_front: int, stimulus_length: int, pad_end: int) -> np.array:
    total_length_time = pad_front + stimulus_length + pad_end
    stimulus = np.zeros((2, total_length_time, _LENGTH_X, _LENGTH_Y), dtype=np.float32)
    stimulus[channel_idx, pad_front:pad_front+stimulus_length] = 1.0
    stimulus_5d = np.expand_dims(stimulus, 0)

    return stimulus_5d
