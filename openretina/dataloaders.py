from collections import namedtuple
from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler


class MovieDataSet(Dataset):
    def __init__(self, movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size):
        if split == "test":
            self.samples = tuple((movies, responses["avg"]))
            self.test_responses_by_trial = responses["by_trial"]
            self.roi_ids = roi_ids
            self.group_assignment = group_assignment
        else:
            self.samples = tuple((movies, responses))
            self.roi_coords = roi_coords
        self.DataPoint = namedtuple("DataPoint", ("inputs", "targets"))
        self.chunk_size = chunk_size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.DataPoint(*[self.samples[0][:, idx, ...], self.samples[1][idx, ...]])
        else:
            return self.DataPoint(
                *[
                    self.samples[0][:, idx : idx + self.chunk_size, ...],
                    self.samples[1][idx : idx + self.chunk_size, ...],
                ]
            )

    def __len__(self):
        return len(self.samples[0])


class MovieSampler(Sampler):
    def __init__(self, start_indices, split, chunk_size):
        self.indices = start_indices
        self.split = split
        self.chunk_size = chunk_size

    def __iter__(self):
        if self.split == "train":  # randomize order for training?
            shift = np.random.randint(0, self.chunk_size)
        else:
            shift = 0
        return iter([idx + shift for idx in self.indices])

    def __len__(self):
        return len(self.indices)


def get_movie_dataloader(
    movies,
    responses,
    roi_ids,
    roi_coords,
    group_assignment,
    scan_sequence_idx,
    split,
    chunk_size,
    start_indices,
    batch_size,
):
    # for right movie: flip second frame size axis!
    if split == "train":
        dataset = MovieDataSet(
            movies[scan_sequence_idx], responses, roi_ids, roi_coords, group_assignment, split, chunk_size
        )
        sampler = MovieSampler(start_indices[scan_sequence_idx], split, chunk_size)
    else:
        dataset = MovieDataSet(movies, responses, roi_ids, roi_coords, group_assignment, split, chunk_size)
        sampler = MovieSampler(start_indices, split, chunk_size)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def natmov_dataloaders_v2(
    experimenters: Tuple[str],
    dates: Tuple[str],
    exp_nums: Tuple[int],
    stim_id: int,
    stim_hash: str,
    keys: Tuple[str],
    dataset_idxss: Tuple[List[int]],
    detrend_param_set_id: int,
    quality_threshold_movie: float,
    quality_threshold_chirp: float,
    quality_threshold_ds: float,
    cell_types: Tuple[str],
    cell_type_crit: str,
    spikes: bool,
    qi_link: str,
    chunk_size: int,  # standard: 50 (frames)
    batch_size: int,  # standard: 32
    seed: int,
    return_data_info: bool = False,
    stim_table: Callable = stim_table,
):
    # make sure movies and responses arrive as torch tensors!!!
    rnd = np.random.RandomState(seed)  # make sure whether we want the validation set to depend on the seed
    if stim_id == 0:
        num_clips, clip_length = NUM_NOISE_CLIPS, NOISE_CLIP_LENGTH
        val_clip_idx = list(rnd.choice(NUM_NOISE_CLIPS, NUM_NOISE_VAL_CLIPS, replace=False))
    else:
        num_clips, clip_length = NUM_CLIPS, CLIP_LENGTH
        val_clip_idx = list(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))

    # if not stim_id == 5:
    #     raise ValueError('Stimulus loading is only implemented for stim_id 5 so far')
    chunk_dict = {"train": chunk_size, "validation": clip_length, "test": 5 * clip_length}
    dataloaders = {"train": {}, "validation": {}, "test": {}}
    # draw validation indices so that a validation movie can be returned!
    movies, random_sequences = (stim_table() & dict(stimulus_hash=stim_hash)).get_all_movie_seqs(val_clip_idx)
    start_indices = gen_start_indices(random_sequences, val_clip_idx, clip_length, chunk_size, num_clips)
    if not (len(keys) == len(experimenters) and len(keys) == len(dates) and len(keys) == len(exp_nums)):
        raise Exception("Your dataset config is inconsistent")
    data_info = {}
    for n, key in enumerate(keys):
        experimenter = experimenters[n]
        date = dates[n]
        exp_num = exp_nums[n]
        dataset_idxs = dataset_idxss[n]
        for i in dataset_idxs:
            neuron_data = NeuronData(
                experimenter,
                date,
                exp_num,
                i,
                detrend_param_set_id,
                quality_threshold_movie,
                quality_threshold_chirp,
                quality_threshold_ds,
                cell_types,
                cell_type_crit,
                qi_link,
                spikes,
                stim_id,
                random_sequences,
                val_clip_idx,
                num_clips,
                clip_length,
            )
            try:
                session_key = "_".join([str(i), key, "".join(date.split("-"))])
            except (
                TypeError
            ):  # The session key versioning was changed in https://github.com/eulerlab/mei/commit/d1f24ef89bdeb4643057ead2ee6b1fb651a90ebb
                session_key = "_".join(["".join(date.split("-")), str(exp_num), str(i)])

            if neuron_data.responses_train.shape[-1] == 0:
                print("skipped: {}".format(session_key))
                break
            for tier in ["train", "validation", "test"]:
                if not (hasattr(neuron_data, "roi_coords")):
                    neuron_data.roi_mask = []
                dataloaders[tier][session_key] = get_movie_dataloader(
                    movies[neuron_data.eye][tier],
                    neuron_data.response_dict[tier],
                    neuron_data.roi_ids,
                    neuron_data.roi_coords,
                    neuron_data.group_assignment,
                    neuron_data.scan_sequence_idx,
                    tier,
                    chunk_dict[tier],
                    start_indices[tier],
                    batch_size,
                )
    in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields
    session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
    n_neurons_dict = {k: v[out_name][-1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}
    for session_key in session_shape_dict:
        data_info[session_key] = dict(
            input_dimensions=in_shapes_dict[session_key],
            input_channels=input_channels[session_key],
            output_dimension=n_neurons_dict[session_key],
            roi_coords=deepcopy(neuron_data.roi_coords.numpy()),
            mean_response=deepcopy(dataloaders["train"][session_key].dataset[:]._asdict()["targets"].mean(0).numpy()),
        )
    return dataloaders if not return_data_info else data_info


def get_dims_for_loader_dict(dataloaders):
    """
    Borrowed from nnfabrik/utility/nn_helpers.py.

    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.

    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.

    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def get_io_dims(data_loader):
    """
    Borrowed from nnfabrik/utility/nn_helpers.py.

    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.

    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.

    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)
