import os

import numpy as np
from tqdm import tqdm

from openretina.data_io.base import MoviesTrainTestSplit
from openretina.utils.file_utils import get_local_file_path


def load_frames(img_dir_name: str | os.PathLike, frame_file: str, full_img_w: int, full_img_h: int):
    """
    loads all stimulus frames of the movie into memory
    """
    img_dir_name = get_local_file_path(str(img_dir_name))
    print("Loading all frames from:", img_dir_name, "into memory")
    images = os.listdir(img_dir_name)
    images = [frame for frame in images if frame_file in frame]
    all_frames = np.zeros((len(images), full_img_w, full_img_h), dtype=np.float16)
    i = 0
    for img_file in tqdm(sorted(images)):
        img = np.load(f"{img_dir_name}/{img_file}")

        all_frames[i] = img / 255
        i += 1
    return all_frames


def process_fixations(fixations, flip_imgs=False, select_flip=None):
    if not flip_imgs:
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0]),
            }
            for f in fixations
        ]
    else:
        print("FLIPPING THE OTHER WAY AROUND!")
        fixations = [
            {
                "img_index": int(float(f.split(" ")[0])),
                "center_x": int(float(f.split(" ")[3])),
                "center_y": int(float(f.split(" ")[4])),
                "flip": int(f.split(" ")[-1][0] == "0"),
            }
            for f in fixations
        ]
    if select_flip is not None:
        fixations = [x for x in fixations if x["flip"] == select_flip]
    return fixations


def build_placeholder_movies(
    session_ids,
    *,
    channels: int,
    height: int,
    width: int,
    time_bins: int = 1,
    test_time_bins: int | None = None,
    stim_id_prefix: str = "sridhar_2025",
    norm_mean: float = 0.0,
    norm_std: float = 1.0,
):
    """
    Create lightweight MoviesTrainTestSplit placeholders that encode spatial dimensions of the stimuli.
    Will not be used directly by the dataloader and model, but rather for `data_info` computation.
    """
    if test_time_bins is None:
        test_time_bins = time_bins

    movies = {}
    for session_id in session_ids:
        train = np.zeros((channels, time_bins, height, width), dtype=np.float32)
        test = np.zeros((channels, test_time_bins, height, width), dtype=np.float32)
        movies[session_id] = MoviesTrainTestSplit(
            train=train,
            test=test,
            stim_id=f"{stim_id_prefix}_{session_id}",
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    return movies
