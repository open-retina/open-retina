import os
import pprint
import random

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


class CustomPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, np.ndarray):
            # Print the shape of the array instead of its contents
            stream.write(f"numpy.ndarray(shape={object.shape})")
        else:
            # Use the standard pretty printing for other types
            super()._format(object, stream, indent, allowance, context, level)
