from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from openretina.data_io.artificial_stimuli import load_chirp, load_moving_bar
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit
from openretina.data_io.base_dataloader import get_movie_dataloader
from openretina.data_io.hoefling_2024.constants import CLIP_LENGTH, NUM_CLIPS
from openretina.data_io.hoefling_2024.responses import NeuronDataSplitHoefling
from openretina.data_io.hoefling_2024.stimuli import gen_start_indices, get_all_movie_combinations


def get_dims_for_loader_dict(dataloaders: dict[str, dict[str, Any]]) -> dict[str, dict[str, tuple[int, ...]] | tuple]:
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


def get_io_dims(data_loader) -> dict[str, tuple[int, ...]] | tuple:
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
        return {k: v.shape for k, v in items.items() if isinstance(v, (torch.Tensor, np.ndarray))}
    else:
        return tuple(v.shape for v in items)


def filter_nan_collate(batch):
    """
    Filters out batches containing NaN values and then calls the default_collate function.
    Can happen for inferred spikes exported with CASCADE.
    To be used as a collate_fn in a DataLoader.

    Args:
        batch (list): A list of tuples representing the batch.

    Returns:
        tuple of torch.Tensor: The collated batch after filtering out NaN values.

    """
    batch = list(filter(lambda x: not np.isnan(x[1]).any(), batch))
    return default_collate(batch)


def filter_different_size(batch):
    """
    Filters out batches that do not have the same shape as most of the other batches.
    """
    # Get the shapes of all the elements in the batch
    shapes = [element[1].shape for element in batch]

    # Find the most common shape in the batch
    most_common_shape = max(set(shapes), key=shapes.count)

    # Filter out elements that do not have the most common shape
    filtered_batch = [element for element in batch if element[1].shape == most_common_shape]

    # If the filtered batch is empty, return None
    return default_collate(filtered_batch) if filtered_batch else None


def extract_data_info_from_dataloaders(
    dataloaders: dict[str, dict[str, Any]] | dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Extracts the data_info dictionary from the provided dataloaders.
    Args:
        dataloaders: A dictionary of dataloaders for different sessions.
    Returns:
        data_info: A dictionary containing input_dimensions, input_channels, and output_dimension for each session,
                   nested with these attributes as the first level keys and sessions as the second level.
    """
    # Ensure train loader is used if available and not provided directly
    dataloaders = dataloaders.get("train", dataloaders)

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name, *_ = next(iter(list(dataloaders.values())[0]))._fields  # type: ignore

    # Get the input and output dimensions for each session
    session_shape_dict = get_dims_for_loader_dict(dataloaders)

    # Initialize the new structure
    data_info: dict[str, dict[str, Any]] = {k: {} for k in session_shape_dict.keys()}

    # Populate the new structure
    for session_key, shapes in session_shape_dict.items():
        data_info[session_key]["input_dimensions"] = shapes[in_name]
        data_info[session_key]["input_channels"] = shapes[in_name][1]
        data_info[session_key]["output_dimension"] = shapes[out_name][-1]
        data_info[session_key]["mean_response"] = np.array(dataloaders[session_key].dataset.mean_response)  # type: ignore
        data_info[session_key]["roi_coords"] = np.array(dataloaders[session_key].dataset.roi_coords)  # type: ignore

    return data_info


def natmov_dataloaders_v2(
    neuron_data_dictionary: dict[str, ResponsesTrainTestSplit],
    movies_dictionary: MoviesTrainTestSplit,
    validation_clip_indices: list[int],
    train_chunk_size: int = 50,
    batch_size: int = 32,
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
):
    stim_ids = {x.stim_id for x in neuron_data_dictionary.values()}
    assert stim_ids == {"natural"}, (
        "This function only supports natural movie stimuli. Stimuli type found",
        f" in neural responses: {stim_ids}",
    )

    clip_chunk_sizes = {
        "train": train_chunk_size,
        "validation": clip_length,
        "test": movies_dictionary.test.shape[1],
    }
    dataloaders: dict[str, dict[str, DataLoader]] = {"train": {}, "validation": {}, "test": {}}

    # Get the random sequences of movies presentations for each session if available
    if movies_dictionary.random_sequences is None:
        movie_length = movies_dictionary.train.shape[1]
        random_sequences = np.arange(0, movie_length // clip_length)[:, np.newaxis]
    else:
        random_sequences = movies_dictionary.random_sequences

    movies = get_all_movie_combinations(
        movies_dictionary.train,
        movies_dictionary.test,
        random_sequences,
        validation_clip_indices=validation_clip_indices,
        num_clips=num_clips,
        clip_length=clip_length,
    )

    start_indices = gen_start_indices(
        random_sequences, validation_clip_indices, clip_length, train_chunk_size, num_clips
    )

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
        # Extract training, validation, and test responses
        neuron_data = NeuronDataSplitHoefling(
            neural_responses=session_data,
            random_sequences=random_sequences,
            **session_data.session_kwargs,
            val_clip_idx=validation_clip_indices,
            num_clips=num_clips,
            clip_length=clip_length,
        )
        _eye = neuron_data.eye

        if session_key == "session_2_ventral2_20200626":
            # session incorrectly labeled as left
            _eye = "right"
        for fold in ["train", "validation", "test"]:
            dataloaders[fold][session_key] = get_movie_dataloader(
                movies=movies[_eye][fold],
                responses=neuron_data.response_dict[fold],
                roi_ids=neuron_data.roi_ids,
                roi_coords=neuron_data.roi_coords,
                group_assignment=neuron_data.group_assignment,
                scan_sequence_idx=neuron_data.scan_sequence_idx,
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                start_indices=start_indices[fold],
                batch_size=batch_size,
                scene_length=clip_length,
            )

    return dataloaders


def get_chirp_dataloaders(
    neuron_data_dictionary,
    train_chunk_size: Optional[int] = None,
    batch_size: int = 32,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert all(
        field in next(iter(neuron_data_dictionary.values()))
        for field in ["responses_final", "stim_id", "chirp_trigger_times"]
    ), (
        "Check the neuron data dictionary sub-dictionaries for the minimal required fields: "
        "'responses_final', 'stim_id' and 'chirp_trigger_times'."
    )

    assert next(iter(neuron_data_dictionary.values()))["stim_id"] == 1, "This function only supports chirp stimuli."

    dataloaders: dict[str, Any] = {"train": {}}

    chirp_triggers = next(iter(neuron_data_dictionary.values()))["chirp_trigger_times"][0]
    # 2 triggers per chirp presentation
    num_chirps = len(chirp_triggers) // 2

    # Get it into chan, time, height, width
    chirp_stimulus = torch.tensor(load_chirp(), dtype=torch.float32).permute(3, 0, 1, 2)

    chirp_stimulus = chirp_stimulus.repeat(1, num_chirps, 1, 1)

    # Use full chirp for training if no chunk size is provided
    clip_chunk_sizes = {
        "train": train_chunk_size if train_chunk_size is not None else chirp_stimulus.shape[1] // num_chirps,
    }

    # 5 chirp presentations
    start_indices = np.arange(0, chirp_stimulus.shape[1] - 1, chirp_stimulus.shape[1] // num_chirps).tolist()

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating chirp dataloaders"):
        neuron_data = NeuronDataSplitHoefling(
            **session_data,
            random_sequences=None,
            val_clip_idx=None,
            num_clips=None,
            clip_length=None,
        )

        session_key += "_chirp"

        dataloader = get_movie_dataloader(
            movies=chirp_stimulus if neuron_data.eye == "right" else torch.flip(chirp_stimulus, [-1]),
            responses=neuron_data.response_dict["train"],
            roi_ids=neuron_data.roi_ids,
            roi_coords=neuron_data.roi_coords,
            group_assignment=neuron_data.group_assignment,
            scan_sequence_idx=neuron_data.scan_sequence_idx,
            split="train",
            chunk_size=clip_chunk_sizes["train"],
            start_indices=start_indices,
            batch_size=batch_size,
            scene_length=chirp_stimulus.shape[1] // num_chirps,
            drop_last=False,
        )
        if dataloader is not None:
            dataloaders["train"][session_key] = dataloader
        else:
            print(f"Ignoring session {session_key} for stimulus chirp")

    return dataloaders


def get_mb_dataloaders(
    neuron_data_dictionary,
    train_chunk_size: Optional[int] = None,
    batch_size: int = 32,
):
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert all(
        field in next(iter(neuron_data_dictionary.values()))
        for field in ["responses_final", "stim_id", "mb_trigger_times"]
    ), (
        "Check the neuron data dictionary sub-dictionaries for the minimal required fields: "
        "'responses_final', 'stim_id' and 'mb_trigger_times'."
    )

    assert (
        next(iter(neuron_data_dictionary.values()))["stim_id"] == 2
    ), "This function only supports moving bar stimuli."

    dataloaders: dict[str, Any] = {"train": {}}

    mb_triggers = next(iter(neuron_data_dictionary.values()))["mb_trigger_times"][0]
    num_repeats = len(mb_triggers) // 8

    # Get it into chan, time, height, width
    mb_stimulus = torch.tensor(load_moving_bar(), dtype=torch.float32).permute(3, 0, 1, 2)

    mb_stimulus = mb_stimulus.repeat(1, num_repeats, 1, 1)

    # 8 directions
    total_num_mbs = 8 * num_repeats

    # Default to each mb for training if no chunk size provided.
    clip_chunk_sizes = {
        "train": train_chunk_size if train_chunk_size is not None else mb_stimulus.shape[1] // total_num_mbs,
    }

    start_indices = np.arange(0, mb_stimulus.shape[1] - 1, step=mb_stimulus.shape[1] // total_num_mbs).tolist()

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating moving bars dataloaders"):
        neuron_data = NeuronDataSplitHoefling(
            **session_data,
            random_sequences=None,
            val_clip_idx=None,
            num_clips=None,
            clip_length=None,
        )

        session_key += "_mb"

        dataloaders["train"][session_key] = get_movie_dataloader(
            movies=mb_stimulus if neuron_data.eye == "right" else torch.flip(mb_stimulus, [-1]),
            responses=neuron_data.response_dict["train"],
            roi_ids=neuron_data.roi_ids,
            roi_coords=neuron_data.roi_coords,
            group_assignment=neuron_data.group_assignment,
            scan_sequence_idx=neuron_data.scan_sequence_idx,
            split="train",
            chunk_size=clip_chunk_sizes["train"],
            start_indices=start_indices,
            batch_size=batch_size,
            scene_length=mb_stimulus.shape[1] // total_num_mbs,
            drop_last=False,
        )

    return dataloaders
