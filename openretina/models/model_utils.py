import torch
from contextlib import contextmanager


def get_module_output_shape(
        model: torch.nn.Module,
        input_shape: tuple[int, ...],
        use_cuda: bool = True
) -> tuple[int, ...]:
    """
    Return the output shape of the model when fed in an array of `input_shape`.
    Note that a zero array of shape `input_shape` is fed into the model and the
    shape of the output of the model is returned.

    Args:
        model (nn.Module): PyTorch module for which to compute the output shape
        input_shape (tuple): Shape specification for the input array into the model
        use_cuda (bool, optional): If True, model will be evaluated on CUDA if available. Othewrise
            model evaluation will take place on CPU. Defaults to True.

    Returns:
        tuple: output shape of the model

    """
    # infer the original device
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            inp_ = torch.zeros(1, *input_shape[1:], device=device)
            output = model.to(device)(inp_)
    model.to(initial_device)
    return output.shape


@contextmanager
def eval_state(model: torch.nn.Module):
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)
