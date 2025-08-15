import os

import numpy as np
from tqdm import tqdm


def load_frames(img_dir_name, frame_file, full_img_w, full_img_h):
    """
    loads all stimulus frames of the movie into memory
    """
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
