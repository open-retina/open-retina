import os

import datasets  # type: ignore
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Sampler

from openretina.data_io.sridhar_2025.constants import NM_DATASET, WN_DATASET
from openretina.utils.misc import set_seed


def download_nm_dataset(cache_dir, base_path, hf_token):
    """Download the natural movie (NM) marmoset dataset from Hugging Face.
    Original data source: https://gin.g-node.org/gollischlab/Sridhar_Gollisch_2025_Marmoset_RGC_Responses_Naturalistic_Movies/src/master/README.md
    Original paper: https://www.biorxiv.org/content/10.1101/2024.03.05.583449v2
    Creates a folder structure to work with the MarmosetMovieDataset class."""
    datasets.logging.set_verbosity_debug()
    load_dataset(
        NM_DATASET,
        name="nm_marmoset_data",
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
        unzip_path=base_path,
    )


def download_wn_dataset(cache_dir, base_path, hf_token):
    """Download the white noise (WN) marmoset dataset from Hugging Face.
    Original data source: https://gin.g-node.org/gollischlab/Sridhar_Gollisch_2025_Marmoset_RGC_Responses_Naturalistic_Movies/src/master/README.md
    Original paper: https://www.biorxiv.org/content/10.1101/2024.03.05.583449v2
    Creates a folder structure to work with the NoiseDataset class."""
    load_dataset(
        WN_DATASET,
        name="wn_marmoset_data",
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
        unzip_path=base_path,
    )


def get_trial_wise_validation_split(train_responses, train_frac, seed=None, final_training=False, hard_coded=None):
    """
    Get a trial-wise validation split of the training responses.
    """
    if hard_coded is not None:
        print(f"hard coded train trials:{hard_coded[0]}")
        print(f"hard coded validation trials: {hard_coded[1]}")
        return hard_coded[0], hard_coded[1]
    num_of_trials = train_responses.shape[-1]
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(1000)
    train_idx, val_idx = np.split(np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)])
    # in the marmoset dataset for seed 2022, the last ~3000 frames are the same as the first ~3000 frames, therefore,
    # they need to be in the same dataset part in the marmoset dataset

    # for seed 2023, it's the same, therefore, they need to be in
    # the same dataset part, this then holds for trials 10 and 19 if all trials are used or 0 and 9 if only the 2023
    # seed trials are used. This however is by ok with the default seed so we don't worry about it now
    while (
        ((0 in train_idx) and (9 in val_idx))
        or ((9 in train_idx) and (0 in val_idx))
        or ((10 in train_idx) and (19 in val_idx) or ((19 in train_idx) and (10 in val_idx)))
    ):
        train_idx, val_idx = np.split(np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)])

    if final_training:
        train_idx = list(train_idx) + list(val_idx)
    else:
        assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"

    print(f"train idx: {train_idx}")
    print(f"val idx: {val_idx}")
    return train_idx, val_idx


def flatten_collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # [B, ...]
    ys = torch.stack(ys)

    # Flatten batch and time
    ys = ys.flatten(0, 1)
    return xs, ys


def get_location_from_sta(sta_dir, file_name, flip_sta=False, crop=0):
    """Get the location of the pixel with maximum variance of the spatial-temporal average (STA).
    The STA is loaded from a file."""
    sta = np.load(os.path.join(sta_dir, file_name))
    if isinstance(crop, int):
        crop = (crop, crop, crop, crop)

    if flip_sta:
        sta = np.flip(sta, axis=1)

    if sum(crop) > 0:
        num_of_imgs, h, w = sta.shape
        sta = sta[:, crop[0] : h - crop[1], crop[2] : w - crop[3]]
    temporal_variances = np.var(sta, axis=0)
    location = np.unravel_index(np.argmax(temporal_variances), (sta.shape[1], sta.shape[2]))
    return location


def make_file_name(cell, retina_index):
    """
    Create a file name based on the provided dataset index and cell index.
    Works for the natural movie marmoset dataset "nm_marmoset_data" and
    white noise marmoset "wn_marmoset_data" datasets.
    """
    return f"cell_data_{retina_index}_WN_stas_cell_{cell}.npy"


def get_locations_from_stas(sta_dir, retina_index, cells, crop=0, flip_sta=False):
    """
    Get the source grid -- i.e. all the locations from the STAs in the sta directory for
    the specified cells and file name.
    Args:
        sta_dir (str): Directory where the STAs are stored.
        retina_index (str): Key of the retina dataset.
        cells (list): List of cell indices for which to get the locations.
        crop (int or tuple): Amount to crop from the STA. If int, crops the same amount from all sides.
                             If tuple, crops (top, bottom, left, right).
        flip_sta (bool): Whether to flip the STA horizontally. - this is useful as the STAs are all
                         computed from the white noise dataset and the natural movie dataset is flipped horizontally.
    """
    all_locations = []
    for cell in cells:
        file_name = make_file_name(cell, retina_index)
        location = get_location_from_sta(sta_dir, file_name, flip_sta=flip_sta, crop=crop)
        all_locations.append(location)
    return np.array(all_locations)


def filter_trials(
    train_responses, all_train_ids, all_validation_ids, hard_coded=None, num_of_trials_to_use=None, starting_trial=0
):
    """
    Selects a subset of trials based on the provided parameters.
    Args:
        train_responses (np.ndarray): The training responses array.
        all_train_ids (np.ndarray): Array of all training trial IDs.
        all_validation_ids (np.ndarray): Array of all validation trial IDs.
        hard_coded (tuple, optional): If provided, uses these IDs instead of filtering.
        num_of_trials_to_use (int, optional): Number of trials to use. If None, all trials are used.
        starting_trial (int): The starting trial index to consider.
    """
    if num_of_trials_to_use is None:
        # If num_of_trials_to_use is not provided, use all trials
        num_of_trials_to_use = len(all_train_ids) + len(all_validation_ids)
    if hard_coded is None:
        train_ids = np.isin(
            all_train_ids,
            np.arange(starting_trial, min(train_responses.shape[2], num_of_trials_to_use + starting_trial)),
        )
        train_ids = np.asarray(all_train_ids)[train_ids]
        valid_ids = np.isin(
            all_validation_ids,
            np.arange(starting_trial, min(train_responses.shape[2], num_of_trials_to_use + starting_trial)),
        )
        valid_ids = all_validation_ids[valid_ids]
    else:
        # If hard_coded is provided, use those IDs directly
        num_trials = train_responses.shape[-1]
        train_ids = [int(x) for x in all_train_ids if int(x) < min(num_trials, num_of_trials_to_use + starting_trial)]
        valid_ids = [
            int(x) for x in all_validation_ids if int(x) < min(num_trials, num_of_trials_to_use + starting_trial)
        ]

    return train_ids, valid_ids


class ChunkedSampler(Sampler):
    def __init__(self, dataset, seed=42):
        """
        Custom sampler that shuffles indices within n chunks and shuffles the chunk order.
        This is to speed up the NoiseDataset training by reducing the number of times a trial is loaded from disk
        It loads one trial file, shuffles chunks within trial and then continues on to the next trial.
        Args:
            dataset (Dataset): The dataset.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.dataset = dataset
        self.num_of_imgs = dataset.num_of_imgs
        self.num_of_frames = dataset.num_of_frames
        self.len = len(dataset)
        self.chunk_size = int(
            np.floor(dataset.num_of_imgs - dataset.frame_overhead) / (dataset.time_chunk_size - dataset.frame_overhead)
        )
        self.seed = seed
        self.indices = np.arange(len(dataset))
        self.num_chunks = int(np.ceil(self.len / self.chunk_size))
        print(f"chunk size: {self.chunk_size}, frames in trial: {self.dataset.num_of_imgs}")  # Size of each chunk

    def _create_shuffled_chunks(self):
        """Create shuffled chunks of indices for each epoch."""
        chunks = [self.indices[i * self.chunk_size : (i + 1) * self.chunk_size] for i in range(self.num_chunks)]

        # Shuffle within each chunk
        for chunk in chunks:
            np.random.shuffle(chunk)

        # Shuffle the order of chunks
        np.random.shuffle(chunks)

        # Flatten shuffled chunks into a single ordered list
        return np.concatenate(chunks)

    def __iter__(self):
        """Reshuffle chunks every epoch."""
        self.final_indices = self._create_shuffled_chunks()
        return iter(self.final_indices)

    def __len__(self):
        return len(self.dataset)
