import warnings

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float

from openretina.models.model_utils import eval_state
from openretina.utils.constants import EPSILON

from .utils.misc import tensors_to_device


def correlation_numpy(
    y1: np.ndarray, y2: np.ndarray, axis: None | int | tuple[int, ...] = -1, eps: float = 1e-8, **kwargs
) -> np.ndarray:
    """Compute the correlation between two NumPy arrays along the specified dimension(s)."""
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    corr = (y1 * y2).mean(axis=axis, **kwargs)
    return corr


def MSE_numpy(y1: np.ndarray, y2: np.ndarray, axis: None | int | tuple[int, ...] = -1, **kwargs) -> np.ndarray:
    """Compute the mean squared error between two NumPy arrays along the specified dimension(s)."""
    return ((y1 - y2) ** 2).mean(axis=axis, **kwargs)


def poisson_loss_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8, mean_axis: None | int | tuple[int, ...] = -1
) -> np.ndarray:
    """Compute the Poisson loss between two NumPy arrays."""
    return (y_pred - y_true * np.log(y_pred + eps)).mean(axis=mean_axis)


def model_predictions(loader, model: torch.nn.Module, data_key, device) -> tuple[np.ndarray, np.ndarray]:
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """
    target, output = torch.empty(0), torch.empty(0)
    for *inputs, responses in loader[data_key]:  # necessary for group assignments
        # code necessary to allow additional pre Ca kernel L1:
        #             curr_output = model(images.to(device), data_key=data_key)
        #             if (type(curr_output) == tuple):
        #                 curr_output = curr_output[0]
        #             output = torch.cat((output, curr_output.detach().cpu()), dim=0)
        output = torch.cat(
            (output, (model(*tensors_to_device(inputs, device), data_key=data_key).detach().cpu())), dim=0
        )
        target = torch.cat((target, responses.detach().cpu()), dim=0)
    output_np = output.numpy()
    target_np = target.numpy()
    lag = target_np.shape[1] - output_np.shape[1]

    return target_np[:, lag:, ...], output_np


def corr_stop(model: torch.nn.Module, loader, avg: bool = True, device: str = "cpu"):
    """
    Returns either the average correlation of all neurons or the correlations per neuron.
        Gets called by early stopping and the model performance evaluation
    """

    n_neurons, correlations_sum = 0, 0
    if not avg:
        all_correlations = np.array([])

    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = correlation_numpy(target, output, axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn(f"{np.isnan(ret).mean() * 100}% NaNs ")
        ret[np.isnan(ret)] = 0

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += output.shape[1]
            correlations_sum += ret.sum()

    corr_ret = correlations_sum / n_neurons if avg else all_correlations
    return corr_ret


def corr_stop3d(model: torch.nn.Module, loader, avg: bool = True, device: str = "cpu"):
    """
    Returns either the average correlation of all neurons or the correlations per neuron.
        Gets called by early stopping and the model performance evaluation
    """

    n_neurons, correlations_sum = 0, 0
    if not avg:
        all_correlations = np.array([])

    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        # Correlation over time axis (1)
        ret = correlation_numpy(target, output, axis=1)

        # Average over batches
        ret = ret.mean(axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn(f"{np.isnan(ret).mean() * 100}% NaNs in corr_stop3d")
        ret[np.isnan(ret)] = 0

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += output.shape[-1]
            correlations_sum += ret.sum()

    corr_ret = correlations_sum / n_neurons if avg else all_correlations
    return corr_ret


def poisson_stop(model: torch.nn.Module, loader, avg: bool = False, device: str = "cpu"):
    poisson_losses = np.array([])
    n_neurons = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = output - target * np.log(output + EPSILON)
        if np.any(np.isnan(ret)):
            warnings.warn(f" {np.isnan(ret).mean() * 100}% NaNs ")

        poisson_losses = np.append(poisson_losses, np.nansum(ret, 0))
        n_neurons += output.shape[1]
    return poisson_losses.sum() / n_neurons if avg else poisson_losses.sum()


def poisson_stop3d(model: torch.nn.Module, loader, avg: bool = True, device: str = "cpu"):
    poisson_losses = np.array([])
    n_neurons = 0
    n_batch = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = output - target * np.log(output + EPSILON)
        if np.any(np.isnan(ret)):
            warnings.warn(f"{np.isnan(ret).mean() * 100:.2f} % NaNs in poisson_stop3d")

        poisson_losses = np.append(poisson_losses, np.nansum(ret, axis=(0, 1)))  # sum over batches (0) and time (1)
        n_neurons += output.shape[-1]
        n_batch += output.shape[0]
    return poisson_losses.sum() / (n_neurons * n_batch) if avg else poisson_losses.sum()


def MSE_stop3d(model: torch.nn.Module, loader, avg: bool = True, device: str = "cpu"):
    mse_losses = np.array([])
    n_neurons = 0
    n_batch = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = (output - target) ** 2
        if np.any(np.isnan(ret)):
            warnings.warn(f"{np.isnan(ret).mean:.1%} NaNs")

        mse_losses = np.append(mse_losses, np.nansum(ret, axis=(0, 1)))  # sum over batches (0) and time (1)
        n_neurons += output.shape[-1]
        n_batch += output.shape[0]
    return mse_losses.sum() / (n_neurons * n_batch) if avg else mse_losses.sum()


def oracle_corr_jackknife(
    repeated_outputs: Float[np.ndarray, "frames repeats neurons"], return_oracle: bool = False
) -> Float[np.ndarray, " neurons"] | tuple[Float[np.ndarray, " neurons"], Float[np.ndarray, " frames repeats neurons"]]:
    """
    Adapted from neuralpredictors.
    Compute the oracle correlations per neuron by averaging over repeated outputs in a leave one out fashion.
    Note that an unequal number of repeats will introduce bias as it distorts assumptions made about the dataset.
    Note that oracle_corr_jackknife underestimates the true oracle correlation.

    Args:
        repeated_outputs (array-like): numpy array with shape (images/time, repeats, neuron responses).

    Returns:
        array: Oracle correlations per neuron. If return_oracle is True, also returns the oracle.
    """

    oracles = []
    oracle_corr = []
    for outputs in repeated_outputs:
        num_repeats = outputs.shape[0]
        oracle = (outputs.sum(axis=0, keepdims=True) - outputs) / (num_repeats - 1)
        if np.any(np.isnan(oracle)):
            oracle[np.isnan(oracle)] = 0
        oracles.append(oracle)
        oracle_corr.append(correlation_numpy(outputs, oracle, axis=0))

    oracle_score = correlation_numpy(
        rearrange(repeated_outputs, "t r n -> (t r) n"),
        rearrange(np.stack(oracles), "t r n -> (t r) n", t=repeated_outputs.shape[0]),
        axis=0,
    )

    return (oracle_score, np.stack(oracles)) if return_oracle else oracle_score


def explainable_vs_total_var(
    repeated_outputs: Float[np.ndarray, "frames repeats neurons"], eps: float = 1e-9, return_explainable: bool = False
) -> Float[np.ndarray, " neurons"] | tuple[Float[np.ndarray, " neurons"], Float[np.ndarray, " neurons"]]:
    """
    Adapted from neuralpredictors.
    Compute the ratio of explainable to total variance per neuron.
    See Cadena et al., 2019: https://doi.org/10.1371/journal.pcbi.1006897

    Args:
        repeated_outputs (array): numpy array with shape (images/time, repeats, neurons) containing the responses.

    Returns:
        array: Ratio of explainable to total variance per neuron. If return_explainable is True, also returns the
        explainable variance for each neuron.
    """
    total_var = np.var(repeated_outputs, axis=(0, 1), ddof=1)
    repeats_var = np.var(repeated_outputs, axis=1, ddof=1)
    noise_var = np.mean(repeats_var, axis=0)
    # Clip. In some bad cases, noise_var can be larger than total_var.
    explainable_var = np.clip(total_var - noise_var, eps, None)
    var_ratio = explainable_var / (total_var + eps)
    return (var_ratio, explainable_var) if return_explainable else var_ratio


def fev(
    targets: Float[np.ndarray, "frames repeats neurons"],
    predictions: Float[np.ndarray, "frames repeats neurons"] | Float[np.ndarray, "frames neurons"],
) -> Float[np.ndarray, " neurons"]:
    """
    Adapted from neuralpredictors.
    Compute the fraction of explainable variance explained per neuron

    Args:
        targets (array-like): Neuron responses (ground truth) over time / different images across repetitions.
        Dimensions: np.array(images/time, num_repeats, num_neurons)
        predictions (array-like): Model predictions to the repeated images, either including or excluding
        repetitions. Dimensions: np.array(images/time, num_repeats, num_neurons) or np.array(images/time, num_neurons)
    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron

    """
    if predictions.shape[1] != targets.shape[1] and predictions.ndim == 2:
        predictions = np.repeat(predictions[:, np.newaxis, :], targets.shape[1], axis=1)

    assert (
        targets.shape == predictions.shape
    ), f"Targets and predictions must have the same shape, got {targets.shape} and {predictions.shape}"

    sum_square_res = [(target - prediction) ** 2 for target, prediction in zip(targets, predictions)]
    sum_square_res = np.concatenate(sum_square_res, axis=0)

    var_ratio, explainable_var = explainable_vs_total_var(targets, return_explainable=True)
    # Invert the formula to get the noise variance
    total_var = explainable_var / var_ratio
    noise_var = total_var - explainable_var

    mse = np.mean(sum_square_res, axis=0)  # mean over time and reps
    fev_e = 1 - np.clip(mse - noise_var, 0, None) / (explainable_var)
    return np.clip(fev_e, 0, None)


def crop_responses(responses: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """

    :param responses: array of responses, last axis is time
    :param predictions: array of predictions, first axis is time
    :return: responses cropped to same shape as predictions
    """

    lag = responses.shape[-1] - predictions.shape[0]
    return responses[..., lag:]
