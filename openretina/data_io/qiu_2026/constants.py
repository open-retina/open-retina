"""Constants for the Qiu et al. 2026 (Franke lab) mouse dataset.

The data ships in Sensorium FileTree trial format (one ``.npy`` per trial), NaN-padded at the end to
a length that is uniform within a session but differs between sessions (300, 450 or 960 frames: the
longest trial that session holds). Every clip trial carries ``CLIP_LENGTH_CUT`` valid frames in
every tier, so the loaders trim by counting valid frames rather than assuming a padded length.
"""

# Frame rate of the stimulus / responses (Hz). Used as the model frame rate.
FRAME_RATE_MODEL = 30

# Per-trial temporal layout (frames @ 30 fps). There is deliberately no constant for the padded
# length: it is 300, 450 or 960 depending on the session (see the module docstring).
CLIP_LENGTH_CUT = 300  # valid length of a clip trial, in every tier, after removing NaN padding

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
