import json
import os

import h5py as h5
import numpy as np
import psutil
from tqdm.auto import tqdm

from .misc import CustomPrettyPrinter


def count_items(group: h5.Group):
    count = 0
    for _, item in group.items():
        if isinstance(item, h5.Dataset):
            count += 1
        elif isinstance(item, h5.Group):
            count += count_items(item)
    return count


def load_h5_into_dict(file_path):
    """
    Recursively loads the structure and data of the HDF5 file into a dictionary.
    """

    def load_group(group, progress_bar, path=""):
        items = {}
        for key, item in group.items():
            if isinstance(item, h5.Dataset):
                # Load the entire dataset
                items[key] = item[...]
                progress_bar.update(1)
            elif isinstance(item, h5.Group):
                items[key] = load_group(item, progress_bar, f"{path}/{key}")

            # Add group attributes to the dictionary
            if group.attrs:
                for attr_name, attr_value in group.attrs.items():
                    items[attr_name] = attr_value
        return items

    # First assert that we have enough space in RAM to load the entire file size
    file_size = os.path.getsize(file_path)
    if file_size > psutil.virtual_memory().available:
        raise MemoryError(
            f"File size of {file_size / (1024**3):.2f} GB is larger than the "
            f"available memory ({psutil.virtual_memory().available / (1024**3):.2f} GB)."
        )

    with h5.File(file_path, "r") as file:
        total_items = count_items(file)
        with tqdm(total=total_items, unit="item", desc="Loading HDF5 file contents") as progress_bar:
            structure = load_group(file, progress_bar)

    return structure


def h5_to_folders(file_path, output_dir):
    """Converts an HDF5 file to a folder structure.

    Args:
        file_path (str): The path to the HDF5 file.
        output_dir (str): The directory where the folder structure will be created.
    """

    def save_dataset(dataset, path):
        """Save a dataset to a given path."""
        # Saving as a NumPy binary file
        np.save(path, dataset[...])

    def save_attributes(attrs, path):
        """Save attributes of a group or dataset as a JSON file, with type conversion for non-serializable types."""

        def convert(item):
            """Convert non-serializable item types."""
            if isinstance(item, np.generic):
                return item.item()  # Convert NumPy scalars to Python scalars
            elif isinstance(item, np.ndarray):
                return item.tolist()  # Convert arrays to lists
            else:
                return item

        attrs_dict = {attr: convert(attrs[attr]) for attr in attrs}

        with open(path, "w") as f:
            json.dump(attrs_dict, f, ensure_ascii=False)

    def explore_and_save(group, current_path, progress_bar):
        """Recursively explore groups and datasets, saving them to disk."""
        os.makedirs(current_path, exist_ok=True)

        # Save group attributes if any
        if group.attrs:
            attrs_path = os.path.join(current_path, "__attributes__.json")
            save_attributes(group.attrs, attrs_path)

        for key, item in group.items():
            item_path = os.path.join(current_path, key)
            if isinstance(item, h5.Dataset):
                save_dataset(item, item_path + ".npy")
                progress_bar.update(1)
            elif isinstance(item, h5.Group):
                explore_and_save(item, item_path, progress_bar)

    with h5.File(file_path, "r") as file:
        total_items = count_items(file)
        with tqdm(total=total_items, unit="item") as progress_bar:
            explore_and_save(file, output_dir, progress_bar)


def print_h5_structure(file_path):
    def explore_group(group, path=""):
        """Recursively explores and prints the structure of the HDF5 file."""
        items = {}
        for key, item in group.items():
            if isinstance(item, h5.Dataset):
                items[key] = f"h5.Dataset(shape={item.shape}), {item.dtype}"
            elif isinstance(item, h5.Group):
                items[key] = explore_group(item, f"{path}/{key}")

        # Add group attributes to the dictionary
        if group.attrs:
            attributes = {}
            for attr_name, attr_value in group.attrs.items():
                attributes[attr_name] = attr_value
            items["__attributes__"] = attributes

        return items

    with h5.File(file_path, "r") as file:
        structure = explore_group(file)

    printer = CustomPrettyPrinter()
    printer.pprint(structure)


def load_dataset_from_h5(file_path, dataset_path: str):
    """
    Loads a dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.

    Returns:
        data: Data of the loaded dataset.
    """
    with h5.File(file_path, "r") as file:
        if dataset_path in file:
            data = file[dataset_path][()]  # type: ignore
            return data
        else:
            raise FileNotFoundError(f"Dataset path {dataset_path} not found in the file.")
