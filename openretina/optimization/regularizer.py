from typing import List, Tuple

import torch


# regularization functions, should map stimulus to penalty term
# e.g. apply penalty if

def no_op_regularizer(stimulus: torch.tensor) -> torch.tensor:
    return 0.0


"""
So far use the defaults for the mouse retina model
postprocessing_config = {
    "norm": 30,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}
"""
def range_regularizer_fn(
        stimulus: torch.tensor,
        min_max_values: List[Tuple[float, float]] = [(-0.654, 6.269), (-0.913, 6.269)],
        max_norm: float = 30.0,
        factor: float = 0.1,
) -> torch.tensor:
    """ Penalizes the stimulus if it is outside the range defined by min_max_values. """
    penalty = 0.0
    for i, (min_val, max_val) in enumerate(min_max_values):
        stimulus_i = stimulus[:, i]
        min_penalty = torch.sum(torch.relu(min_val - stimulus_i))
        max_penalty = torch.sum(torch.relu(stimulus_i - max_val))
        print(f"{i=} {min_penalty=} {max_penalty=}")
        penalty += min_penalty + max_penalty

    # Add a penalty such that the norm of the stimulus is lower than max_norm
    norm_penalty = torch.relu(torch.norm(stimulus) - max_norm)
    x_norm = torch.norm(stimulus.view(len(stimulus), -1), dim=-1)
    penalty += norm_penalty
    print(f"{norm_penalty=}")

    penalty *= factor
    return penalty



