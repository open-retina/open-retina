import torch


# regularization functions, should map stimulus to penalty term
# e.g. apply penalty if

def no_op_regularizer(stimulus: torch.tensor) -> torch.tensor:
    return 0.0
