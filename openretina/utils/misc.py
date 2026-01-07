"""
Miscellaneous utility functions and classes.
"""

import contextlib
import os
import pprint
import random
import sys
from typing import Iterable, Union

import numpy as np
import omegaconf
import requests
import torch
import yaml
from einops import rearrange


def set_seed(seed: int | None = None, seed_torch: bool = True) -> int:
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
    seed : Integer
            A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Default is `True`.

    Returns:
    seed : Integer corresponding to the random state.
    """
    if seed is None:
        seed = np.random.choice(2**16)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")
    return seed


class MaxLinesExceededException(Exception):
    pass


class CustomPrettyPrinter(pprint.PrettyPrinter):
    """
    A custom pretty printer that provides specialized formatting for certain types of objects.

    Args:
        indent (int): Number of spaces for each indentation level.
        width (int): Maximum number of characters per line.
        depth (int): Maximum depth to print nested structures.
        stream (file-like object): Stream to write the formatted output to.

    Attributes:
        indent (int): Number of spaces for each indentation level.
        width (int): Maximum number of characters per line.
        depth (int): Maximum depth to print nested structures.
        stream (file-like object): Stream to write the formatted output to.

    Methods:
        _format(object, stream, indent, allowance, context, level):
            Formats the given object and writes the formatted output to the stream.

    Example:
        printer = CustomPrettyPrinter()
        printer.pprint(np.array([1, 2, 3]))
        # Output: numpy.ndarray(shape=(3,))
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, max_lines=None):
        super().__init__(indent, width, depth, stream)
        self.max_lines = max_lines
        self.current_line = 0

    def _format(self, object, stream, indent, allowance, context, level):
        if self.max_lines is not None and self.current_line >= self.max_lines:
            stream.write("\n ... Exceeded maximum number of lines ...")
            raise MaxLinesExceededException

        if isinstance(object, np.ndarray):
            # Print the shape of the array instead of its contents
            stream.write(f"numpy.ndarray(shape={object.shape})")
            self.current_line += 1
        elif isinstance(object, torch.Tensor):
            # Print the shape of the tensor instead of its contents
            stream.write(f"torch.Tensor(shape={list(object.shape)})")
            self.current_line += 1
        elif isinstance(object, list) and len(object) > 10:
            stream.write(f"list(len={len(object)})")
            self.current_line += 1
        elif isinstance(object, torch.utils.data.DataLoader):
            # Print the dataset name instead of the DataLoader object
            stream.write(f"torch.utils.data.DataLoader(Dataset: {object.dataset})")
            self.current_line += 1
        elif isinstance(object, omegaconf.DictConfig):
            # Convert the OmegaConf object to a dictionary and print it
            self._format(omegaconf.OmegaConf.to_container(object), stream, indent, allowance, context, level)
        elif hasattr(object, "_fields"):
            # Namedtuple: print type and fields
            stream.write(f"{type(object).__name__}(")
            for i, field in enumerate(object._fields):
                if i > 0:
                    stream.write(", ")
                stream.write(f"{field}=")
                self._format(
                    getattr(object, field),
                    stream,
                    indent + len(field) + 2,
                    allowance if i == len(object._fields) - 1 else 1,
                    context,
                    level + 1,
                )
            stream.write(")")
            self.current_line += 1
        else:
            # Use the standard pretty printing for other types
            super()._format(object, stream, indent, allowance, context, level)
            self.current_line += 1

    def pprint(self, object):
        self.current_line = 0
        with contextlib.suppress(MaxLinesExceededException):
            super().pprint(object)


@contextlib.contextmanager
def redirect_stdout(file=open(os.devnull, "w")):
    stdout_fd = sys.stdout.fileno()
    stdout_fd_dup = os.dup(stdout_fd)
    os.dup2(file.fileno(), stdout_fd)
    file.close()
    try:
        yield
    finally:
        os.dup2(stdout_fd_dup, stdout_fd)
        os.close(stdout_fd_dup)


def tensors_to_device(tensors: Union[Iterable[torch.Tensor], torch.Tensor], device=None):
    """
    Move a collection of tensors to a specified device. Handles lists and tuples.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)

    if isinstance(tensors, list):
        return [tensors_to_device(item, device) for item in tensors]

    if isinstance(tensors, tuple):
        return tuple(tensors_to_device(item, device) for item in tensors)

    raise TypeError("Input must be a tensor, list of tensors, or tuple of tensors.")


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


class SafeLoaderWithTuple(yaml.SafeLoader):
    pass


def check_server_responding(url: str) -> bool:
    try:
        response = requests.get(url)
    except ConnectionError:
        return False

    return response.status_code == 200


def reorder_like_a(a, b):
    """
    Reorder b so that it matches a's two dims, with the extra dim first.

    a: shape (D1, D2)
    b: shape (X, Y, Z), where two of these match (D1, D2) in any order

    Returns:
        b_reordered: shape (extra, D1, D2)
    """
    D1, D2 = a.shape
    shape_to_axis = {dim: i for i, dim in enumerate(b.shape)}

    if D1 not in shape_to_axis or D2 not in shape_to_axis:
        raise ValueError("b does not contain both dimensions of a")

    extra_dim = [d for d in b.shape if d not in (D1, D2)]
    if len(extra_dim) != 1:
        raise RuntimeError("Could not uniquely identify extra dimension")

    extra = extra_dim[0]

    # Map dimension sizes to einops axis names
    pattern_in = []
    for s in b.shape:
        if s == extra:
            pattern_in.append("extra")
        elif s == D1:
            pattern_in.append("d1")
        elif s == D2:
            pattern_in.append("d2")

    pattern_in = " ".join(pattern_in)

    return rearrange(b, f"{pattern_in} -> extra d1 d2")
