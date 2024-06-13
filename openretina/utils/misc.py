"""
Miscellaneous utility functions and classes.
"""

import contextlib
import os
import pprint
import random
import sys

import numpy as np
import torch


def set_seed(seed=None, seed_torch=True):
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
