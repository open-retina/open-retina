from functools import partial

import numpy as np
import torch
from neuralpredictors.training import (
    LongCycler,
    MultipleObjectiveTracker,
    early_stopping,
)
from tqdm.auto import tqdm

from . import measures, metrics
from .misc import set_seed


def standard_early_stop_trainer(
    model: torch.nn.Module,
    dataloaders,
    seed: int,
    scale_loss: bool = True,  # trainer args
    loss_function: str = "PoissonLoss3d",
    stop_function: str = "corr_stop",
    loss_accum_batch_n=None,
    device: str = "cuda",
    verbose: bool = True,
    interval: int = 1,
    patience: int = 5,
    epoch: int = 0,
    lr_init: float = 0.005,  # early stopping args
    max_iter: int = 100,
    maximize: bool = True,
    tolerance: float = 1e-6,
    restore_best: bool = True,
    lr_decay_steps: int = 3,
    lr_decay_factor: float = 0.3,
    min_lr: float = 0.0001,  # lr scheduler args
    detach_core: bool = False,
    cb=None,
    **kwargs
):
    # Defines objective function; criterion is resolved to the loss_function that is passed as input
    def full_objective(model, data_key, inputs, targets, detach_core):
        regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        if scale_loss:
            m = len(trainloaders[data_key].dataset)
            k = inputs.shape[0]
            loss_scale = np.sqrt(m / k)
        else:
            loss_scale = 1.0
        return (
            loss_scale * criterion(model(inputs.to(device), data_key, detach_core=detach_core), targets.to(device))
            + regularizers
        )

    trainloaders = dataloaders["train"]
    valloaders = dataloaders.get("validation", dataloaders["val"] if "val" in dataloaders.keys() else None)
    testloaders = dataloaders["test"]

    ##### Model training ####################################################################################################
    model.to(device)
    set_seed(seed)
    model.train()

    criterion = getattr(measures, loss_function)()
    stop_closure = partial(getattr(metrics, stop_function), model, valloaders, device=device)

    n_iterations = len(LongCycler(trainloaders))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(trainloaders.keys()) if loss_accum_batch_n is None else loss_accum_batch_n
    )  # will be equal to number of sessions - why?

    # define some trackers
    tracker_dict = dict(
        val_correlation=partial(metrics.corr_stop3d, model, valloaders, device=device),
        val_poisson_loss=partial(metrics.poisson_stop3d, model, valloaders, device=device),
        val_MSE_loss=partial(metrics.MSE_stop3d, model, valloaders, device=device),
    )

    if hasattr(model, "tracked_values"):
        tracker_dict.update(model.tracked_values)

    tracker = MultipleObjectiveTracker(**tracker_dict)

    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        #         with torch.autograd.detect_anomaly():
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(trainloaders)), total=n_iterations, desc="Epoch {}".format(epoch)
        ):
            loss = full_objective(model, data_key, *data, detach_core)
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize()

    # Compute avg validation and test correlation
    avg_val_corr = metrics.corr_stop3d(model, valloaders, avg=True, device=device)
    avg_test_corr = metrics.corr_stop3d(model, testloaders, avg=True, device=device)

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()}

    return avg_test_corr, avg_val_corr, output, model.state_dict()
