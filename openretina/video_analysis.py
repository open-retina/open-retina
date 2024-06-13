from typing import Tuple

import numpy as np
import scipy


def butter_lowpass_filter(data: np.ndarray, lowpass_cutoff: float, fs: float, order: int = 5):
    b, a = scipy.signal.butter(order, Wn=lowpass_cutoff, fs=fs, btype="low")
    y = scipy.signal.filtfilt(b, a, data)
    return y


def calculate_fft(
    temporal_kernel: np.ndarray,
    sampling_frequency: float = 30.0,
    lowpass_cutoff: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    y = butter_lowpass_filter(temporal_kernel, lowpass_cutoff=lowpass_cutoff, fs=sampling_frequency)
    n_samples = temporal_kernel.shape[0]
    fft_frequencies = scipy.fft.fftfreq(n_samples, 1 / sampling_frequency)
    fft_values = scipy.fft.fft(y)

    fft_frequencies = fft_frequencies[: n_samples // 2]
    fft_weights = np.abs(fft_values[: n_samples // 2])

    return fft_frequencies, fft_weights


def weighted_main_frequency(fft_frequencies: np.ndarray, fft_weights: np.ndarray) -> float:
    assert len(fft_frequencies.shape) == 1
    assert fft_frequencies.shape == fft_weights.shape
    return np.average(fft_frequencies, weights=fft_weights)


def decompose_kernel(
    space_time_kernel: np.ndarray, scaling_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale spatial and temporal components such that spatial component is in the range [-1, 1]
    Args:
        space_time_kernel: shape=[time, y_shape, x.shape]
        scaling_factor: optional scaling factor
    Returns:
    """
    assert len(space_time_kernel.shape) == 3
    dt, dy, dx = space_time_kernel.shape

    space_time_kernel_flat = space_time_kernel.reshape((dt, dy * dx))
    U, S, V = scipy.linalg.svd(space_time_kernel_flat)
    temporal = U[:, 0]
    spatial = V[0]
    scaling_factor *= S[0]
    abs_max_val = max(np.abs(spatial.min()), np.abs(spatial.max()))
    spatial = spatial / abs_max_val
    temporal = temporal * abs_max_val * scaling_factor
    reshaped_spatial = spatial.reshape((dy, dx))
    center_x, center_y = int(dy / 2), int(dx / 2)
    if np.mean(reshaped_spatial[center_x - 3: center_x + 2, center_y - 3: center_y + 2]) < 0:
        spatial *= -1
        temporal *= -1
    singular_values = S[1]
    return temporal, reshaped_spatial, singular_values
