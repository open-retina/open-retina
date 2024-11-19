from contextlib import contextmanager

import torch
from lightning.pytorch.callbacks import Callback


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
