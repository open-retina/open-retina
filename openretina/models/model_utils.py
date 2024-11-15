from contextlib import contextmanager

import torch
from lightning.pytorch.callbacks import Callback


def get_module_output_shape(
    model: torch.nn.Module, input_shape: tuple[int, ...], use_cuda: bool = True
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


class OptimizerResetCallback(Callback):
    def __init__(self):
        super().__init__()
        self.prev_lr = None  # This will store the previous learning rate

    def on_validation_end(self, trainer, pl_module):
        # Get the current learning rate from the optimizer
        optims = pl_module.optimizers()
        try:
            optim = optims[0]
        except:  # noqa
            optim = optims
        current_lr = optim.param_groups[0]["lr"]

        # Compare with the previous learning rate
        if self.prev_lr is not None and current_lr < self.prev_lr:
            print(f"Learning rate decreased from {self.prev_lr} to {current_lr}. Resetting optimizer.")
            # Reset the optimizer if the learning rate has decreased
            params_dict = optim.param_groups[0]
            # below could be written shorter
            new_optimizer = torch.optim.AdamW(
                pl_module.parameters(),
                lr=current_lr,
                betas=params_dict["betas"],
                eps=params_dict["eps"],
                weight_decay=params_dict["weight_decay"],
                amsgrad=params_dict["amsgrad"],
                maximize=params_dict["maximize"],
                foreach=params_dict["foreach"],
                capturable=params_dict["capturable"],
                differentiable=params_dict["differentiable"],
                fused=params_dict["fused"],
            )
            trainer.optimizers = [new_optimizer]  # Replace the optimizer in the trainer

        self.prev_lr = current_lr
