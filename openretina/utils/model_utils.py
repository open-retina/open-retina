import math
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


def get_core_output_based_on_dimensions(model_config):
    if len(model_config.in_shape) == 5:
        input_shape = model_config.in_shape[1:]
    else:
        input_shape = model_config.in_shape
    ch, t, h, w = input_shape[:]
    layers = len(model_config.core.temporal_kernel_sizes)

    for i in range(layers):
        temp_kernel_size = model_config.core.temporal_kernel_sizes[i]
        spatial_kernel_size = model_config.core.spatial_kernel_sizes[i]
        if isinstance(spatial_kernel_size, int):
            spatial_kernel_size = (spatial_kernel_size, spatial_kernel_size)
        temporal_dilation = (
            1 if "temporal_dilation" not in model_config.core.keys() else model_config.core.temporal_dilation
        )
        spatial_dilation = (
            1 if "spatial_dilation" not in model_config.core.keys() else model_config.core.spatial_dilation
        )
        stride = [1] * layers if "stride" not in model_config.core.keys() else model_config.core.stride

        t = math.floor(t - (temp_kernel_size - 1) * temporal_dilation)
        h = math.floor((h - (spatial_kernel_size[0] - 1) * spatial_dilation - 1) / stride[i] + 1)
        w = math.floor((w - (spatial_kernel_size[1] - 1) * spatial_dilation - 1) / stride[i] + 1)

    output_shape = model_config.core.channels[layers], t, h, w
    print("factorized core output shape:", output_shape)
    return output_shape
