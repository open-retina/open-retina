from openretina.data_io.sridhar_2025.constants import NM_DATASET, WN_DATASET
from openretina.utils.misc import set_seed
import numpy as np
import datasets
from datasets import load_dataset
# from openretina.env import HF_TOKEN


def download_nm_dataset(cache_dir, base_path, hf_token):
    datasets.logging.set_verbosity_debug()
    load_dataset(NM_DATASET, name='nm_marmoset_data', cache_dir=cache_dir, token=hf_token,
                      trust_remote_code=True, unzip_path=base_path)


def download_wn_dataset(cache_dir, base_path, hf_token):
    ds = load_dataset(WN_DATASET, name='wn_marmoset_data', cache_dir=cache_dir, token=hf_token,
                             trust_remote_code=True, unzip_path=base_path)

def get_trial_wise_validation_split(
    train_responses, train_frac, seed=None, final_training=False, hard_coded=None
):
    if hard_coded is not None:
        print(f'hard coded train trials:{hard_coded[0]}')
        print(f'hard coded validation trials: {hard_coded[1]}')
        return hard_coded[0], hard_coded[1]
    num_of_trials = train_responses.shape[-1]
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(1000)
    train_idx, val_idx = np.split(
        np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)]
    )
    # in the marmoset dataset for seed 2022, the last ~3000 frames are the same as the first ~3000 frames, therefore,
    # they need to be in the same dataset part in the marmoset dataset

    # for seed 2023, it's the same, therefore, they need to be in
    # the same dataset part, this then holds for trials 10 and 19 if all trials are used or 0 and 9 if only the 2023
    # seed trials are used. This however is by ok with the default seed so we don't worry about it now
    while ((0 in train_idx) and (9 in val_idx)) or (
        (9 in train_idx) and (0 in val_idx)
    ) or ((10 in train_idx) and (19 in val_idx) or ((19 in train_idx) and (10 in val_idx))):
        train_idx, val_idx = np.split(
            np.random.permutation(int(num_of_trials)), [int(num_of_trials * train_frac)]
        )

    if final_training:
        train_idx = list(train_idx) + list(val_idx)
    else:
        assert not np.any(
            np.isin(train_idx, val_idx)
        ), "train_set and val_set are overlapping sets"

    print(f"train idx: {train_idx}")
    print(f"val idx: {val_idx}")
    return train_idx, val_idx