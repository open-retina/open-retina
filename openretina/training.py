import datetime
import os
from functools import partial

try:
    import wandb
except ImportError:
    print("wandb not installed, not logging to wandb")

import numpy as np
import torch
from tqdm.auto import tqdm

from . import measures, metrics
from .cyclers import LongCycler
from .early_stopping import early_stopping
from .tracking import MultipleObjectiveTracker
from .utils.misc import set_seed


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
    wandb_logger=None,
    cb=None,
    **kwargs,
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

        predictions = model(inputs.to(device), data_key, detach_core=detach_core)
        loss_criterion = criterion(predictions, targets.to(device))
        res = loss_scale * loss_criterion + regularizers
        return res

    trainloaders = dataloaders["train"]
    valloaders = dataloaders.get("validation", dataloaders["val"] if "val" in dataloaders.keys() else None)
    testloaders = dataloaders["test"]

    # Model training
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
    )  # will be equal to number of sessions (dict keys) if not specified

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
    for epoch, val_obj in tqdm(
        early_stopping(
            model,
            stop_closure,
            interval=interval,
            patience=patience,
            start=epoch,
            max_iter=max_iter,
            maximize_objective=maximize,
            tolerance=tolerance,
            restore_best=restore_best,
            tracker=tracker,
            scheduler=scheduler,
            lr_decay_steps=lr_decay_steps,
        ),
        desc="Epochs",
        total=max_iter,
        position=0,
        leave=True,
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
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(trainloaders)),
            total=n_iterations,
            desc=f"Epoch {epoch}",
            position=1,
            leave=True,
            disable=not verbose,
        ):
            clean_data_key = clean_session_key(data_key)
            loss = full_objective(model, clean_data_key, *data, detach_core)  # type: ignore
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()
            if np.isnan(loss.item()):
                raise ValueError(f"Loss is NaN on batch {batch_no} from {data_key}, stopping training.")
        if wandb_logger is not None:
            tracker_info = tracker.asdict(make_copy=True)
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "val_corr": tracker_info["val_correlation"][-1],
                    "val_poisson_loss": tracker_info["val_poisson_loss"][-1],
                    "val_MSE_loss": tracker_info["val_MSE_loss"][-1],
                }
            )

    # Model evaluation
    model.eval()
    tracker.finalize()

    # Compute avg validation and test correlation
    avg_val_corr = metrics.corr_stop3d(model, valloaders, avg=True, device=device)
    avg_test_corr = metrics.corr_stop3d(model, testloaders, avg=True, device=device)

    # return the whole tracker output as a dict
    output = tracker.asdict()

    return avg_test_corr, avg_val_corr, output, model.state_dict()


def save_checkpoint(model, optimizer, epoch, loss, save_folder: str, model_name: str) -> None:
    if not os.path.exists(save_folder):
        # only create the lower level directory if it does not exist
        os.mkdir(save_folder)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(save_folder, f"{model_name}_{date}_checkpoint.pt"),
    )


def save_model(model: torch.nn.Module, save_folder: str, model_name: str) -> None:
    os.makedirs(save_folder, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save(model.state_dict(), os.path.join(save_folder, f"{model_name}_{date}_model_weights.pt"))
    torch.save(model, os.path.join(save_folder, f"{model_name}_{date}_model.pt"))


def clean_session_key(session_key: str) -> str:
    # Ignore this function when only training on the chirp or movingbar by uncommenting the following line
    # return session_key
    if "_chirp" in session_key:
        session_key = session_key.split("_chirp")[0]
    if "_mb" in session_key:
        session_key = session_key.split("_mb")[0]
    return session_key
