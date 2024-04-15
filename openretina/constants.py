FRAME_RATE_MODEL = 30.0  # Frames/s

RGC_GROUP_NAMES_DICT = {
    1: "OFF local, OS",
    2: "OFF DS",
    3: "OFF step",
    4: "OFF slow",
    5: "OFF alpha sust.",
    6: "(ON-)OFF Jam-B mix",
    7: "OFF sust",
    8: "OFF alpha trans",
    9: "OFF mini alpha trans",
    10: "ON-OFF local-edge W3",
    11: "ON-OFF local",
    12: "ON-OFF DS 1",
    13: "ON-OFF DS 2",
    14: "(ON-)OFF local, OS",
    15: "ON step",
    16: "ON DS trans",
    17: "ON local trans, OS",
    18: "ON trans",
    19: "ON trans, large",
    20: "ON high freq",
    21: "ON low freq",
    22: "ON sust",
    23: "ON mini alpha",
    24: "ON alpha",
    25: "ON DS sust 1",
    26: "ON DS sust 2",
    27: "ON slow",
    28: "ON contrast suppr.",
    29: "ON DS sust 3",
    30: "ON local sust Os",
    31: "OFF suppr. 1",
    32: "OFF suppr. 2",
    33: "OFF AC",
    34: "ON high freq. sust. 1 AC",
    35: "ON high freq. trans. AC",
    36: "ON-OFF high freq. AC",
    37: "ON high freq. sust. 2 AC",
    38: "ON sust. 1 AC",
    39: "ON sust. 2 AC",
    40: "ON sust. 3 AC",
    41: "ON sust. 4 AC",
    42: "ON starburst AC",
    43: "ON-OFF local AC",
    44: "ON step AC",
    45: "ON local 1 AC",
    46: "ON local 2 AC",
}

SCENE_LENGTH = 150  # Frames
NUM_CLIPS = 108
NUM_VAL_CLIPS = 15
CLIP_LENGTH = 150  # Frames


EPSILON = 1e-8

STIMULI_IDS = {"natural": 5, "chirp": 1, "mb": 2}
