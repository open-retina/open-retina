"""Constants for the Qiu et al. 2026 (Franke lab) mouse dataset.

The data ships in Sensorium FileTree trial format (one ``.npy`` per trial), NaN-padded
to ``CLIP_LENGTH_FULL`` frames. Train/validation trials carry ``CLIP_LENGTH_CUT`` valid
frames; test trials are internally uniform at either 300 or 450 valid frames.
"""

# Frame rate of the stimulus / responses (Hz). Used as the model frame rate.
FRAME_RATE_MODEL = 30

# Per-trial temporal layout (frames @ 30 fps).
CLIP_LENGTH_FULL = 450  # padded length every trial array is stored at
CLIP_LENGTH_CUT = 300  # valid length of train/validation trials after removing NaN padding

# Single-channel video geometry (channels, height, width).
VIDEO_SHAPE = (1, 36, 64)

# Behavior channels to fold into the video input as extra channels: pupil size (0) and
# locomotion (2). Channel 1 (Delta-pupil) is dropped, matching the reference notebook.
BEHAVIOR_CHANNELS = (0, 2)

# Total input channels seen by the core: 1 video + len(BEHAVIOR_CHANNELS) behavior.
N_INPUT_CHANNELS = 1 + len(BEHAVIOR_CHANNELS)

# One zip per session under franke_lab/qiu_2026/; sessions are discovered by listing the
# folder and keeping entries with this suffix (never hard-code a single session).
SESSION_ZIP_SUFFIX = "-Fluorescence-7b721b-v4a.zip"

# Neuron quality masks live under this sub-folder as INDEX arrays (not boolean), named
# "<session-prefix>_neurons_fluor_good.npy"; the prefix truncates the hash to 7 chars.
DATA_QUALITY_DIRNAME = "data-quality"
QUALITY_MASK_SUFFIX = "_neurons_fluor_good.npy"

# --- In-silico stimulus constraints ------------------------------------------------------------
#
# Range and scale of the *normalized* (z-scored) video channel, measured over the concatenated
# train movie of all 10 sessions (2.3e9 pixels) with
# `openretina.data_io.qiu_2026.stimuli.measure_video_range` on 2026-08-24. The percentiles are
# the 0.1/99.9 across sessions (lowest low, highest high); `rms_video` is pooled exactly over
# every pixel.
#
# Cross-checked against the 8-bit envelope implied by each session's own shipped mean/std --
# `-mean/std` to `(255 - mean)/std`, which spans [-1.2893, 2.9959] -- and both measured bounds
# fall inside it. Note the asymmetry: the raw movie's mean sits well below mid-grey, so the
# normalized video reaches +2.96 but only -1.29. Clipping symmetrically would be wrong.
STIMULUS_RANGE_CONSTRAINTS = {
    "x_min_video": -1.29,
    "x_max_video": 2.96,
    "rms_video": 0.996,
}

# Sweep range for the z-scored behavior channels (pupil size, locomotion) and pupil position,
# in standard deviations around each session's mean state.
BEHAVIOR_SWEEP_RANGE = (-2.0, 2.0)


def video_range_and_norm(
    time_steps: int,
    height: int = VIDEO_SHAPE[1],
    width: int = VIDEO_SHAPE[2],
    rms_factor: float = 1.0,
) -> tuple[list[tuple[float, float]], float]:
    """Range and norm constraints for optimizing a ``(1, 1, time_steps, height, width)`` video.

    Returns ``(min_max_values, norm)`` ready to hand to ``ChangeNormJointlyClipRangeSeparately``
    and ``RangeRegularizationLoss``. Both expect the *optimized* tensor, so pair this with
    :class:`~openretina.insilico.stimulus_optimization.fixed_channel_model.FixedChannelStimulusModel`,
    which hides the behavior channels; passing a full 3-channel stimulus instead makes the joint
    norm meaningless (see that module's docstring).

    The norm is **derived**, not tabulated: ``rms_factor * rms_video * sqrt(time_steps*height*width)``.
    An MEI's length is a free choice, and a norm hard-coded for one length is silently wrong for
    another -- 50 frames gives 338, 40 frames gives 302. Deriving it also states the intent
    plainly: hold the optimized stimulus at the same per-pixel RMS as the training video.
    ``rms_factor`` scales that target (``2.0`` for a deliberately high-contrast MEI).

    Args:
        time_steps: length of the stimulus in frames.
        height: stimulus height; defaults to the dataset's.
        width: stimulus width; defaults to the dataset's.
        rms_factor: multiple of the training video's RMS to target.

    Returns:
        ``([(x_min_video, x_max_video)], norm)`` -- a one-entry range list, since the optimized
        tensor has exactly one channel.
    """
    if time_steps < 1 or height < 1 or width < 1:
        raise ValueError(f"time_steps, height and width must all be >= 1, got {time_steps=} {height=} {width=}.")
    min_max_values = [(STIMULUS_RANGE_CONSTRAINTS["x_min_video"], STIMULUS_RANGE_CONSTRAINTS["x_max_video"])]
    norm = rms_factor * STIMULUS_RANGE_CONSTRAINTS["rms_video"] * (time_steps * height * width) ** 0.5
    return min_max_values, norm
