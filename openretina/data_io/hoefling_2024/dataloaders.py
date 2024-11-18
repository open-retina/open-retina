from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate


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
