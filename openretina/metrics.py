import warnings

import numpy as np
import torch

from neuralpredictors.measures import corr
from neuralpredictors.training import eval_state

from .utils.constants import EPSILON


def model_predictions(loader, model, data_key, device):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """
    target, output = torch.empty(0), torch.empty(0)
    for images, responses, *_ in loader[data_key]:  # necessary for group assignments
        ### code necessary to allow additional pre Ca kernel L1
        #             curr_output = model(images.to(device), data_key=data_key)
        #             if (type(curr_output) == tuple):
        #                 curr_output = curr_output[0]
        #             output = torch.cat((output, curr_output.detach().cpu()), dim=0)
        output = torch.cat((output, (model(images.to(device), data_key=data_key).detach().cpu())), dim=0)
        target = torch.cat((target, responses.detach().cpu()), dim=0)
    output = output.numpy()
    target = target.numpy()
    lag = target.shape[1] - output.shape[1]

    return target[:, lag:, ...], output


def corr_stop(model, loader, avg=True, device="cpu"):
    """
    Returns either the average correlation of all neurons or the the correlations per neuron.
        Gets called by early stopping and the model performance evaluation
    """

    n_neurons, correlations_sum = 0, 0
    if not avg:
        all_correlations = np.array([])

    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = corr(target, output, axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn("{}% NaNs ".format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += output.shape[1]
            correlations_sum += ret.sum()

    corr_ret = correlations_sum / n_neurons if avg else all_correlations
    return corr_ret


def corr_stop3d(model, loader, avg=True, device="cpu"):
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

        ret = corr(target, output, axis=1)
        ret = ret.mean(axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn("{}% NaNs in corr_stop3d".format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += output.shape[-1]
            correlations_sum += ret.sum()

    corr_ret = correlations_sum / n_neurons if avg else all_correlations
    return corr_ret


def poisson_stop(model, loader, avg=False, device="cpu"):
    poisson_losses = np.array([])
    n_neurons = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = output - target * np.log(output + EPSILON)
        if np.any(np.isnan(ret)):
            warnings.warn(" {}% NaNs ".format(np.isnan(ret).mean() * 100))

        poisson_losses = np.append(poisson_losses, np.nansum(ret, 0))
        n_neurons += output.shape[1]
    return poisson_losses.sum() / n_neurons if avg else poisson_losses.sum()


def poisson_stop3d(model, loader, avg=True, device="cpu"):
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


def MSE_stop3d(model, loader, avg=True, device="cpu"):
    MSE_losses = np.array([])
    n_neurons = 0
    n_batch = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = (output - target) ** 2
        if np.any(np.isnan(ret)):
            warnings.warn(" {}% NaNs ".format(np.isnan(ret).mean() * 100))

        MSE_losses = np.append(MSE_losses, np.nansum(ret, axis=(0, 1)))  # sum over batches (0) and time (1)
        n_neurons += output.shape[-1]
        n_batch += output.shape[0]
    return MSE_losses.sum() / (n_neurons * n_batch) if avg else MSE_losses.sum()


def evaluate_fev(model, loader, device="cpu", ddof=0):
    """
    Evaluates the Fraction explainable variance explained by a model
    as described in Cadena et al 2019
    :param model: a nnfabrik model
    :param loader: a test dataloader
    :param device:
    :param ddof: 0 or 1; whether to calculate noise variance without (0) or with (1) bias correction
    :return:
    """
    noise_variance_dict = dict.fromkeys(loader.keys())
    total_variance_dict = dict.fromkeys(loader.keys())
    mse_dict = dict.fromkeys(loader.keys())
    fev_dict = dict.fromkeys(loader.keys())
    repeated_predictions_dict = dict.fromkeys(loader.keys())
    for i, data_key in enumerate(loader.keys()):
        with eval_state(model):
            _, predictions = model_predictions(loader, model, data_key, device)
        predictions = np.squeeze(predictions)  # should now be time x neurons
        test_responses_by_trial = loader[
            data_key
        ].dataset.test_responses_by_trial.numpy()  # shape is neurons x repetitions x time
        cropped_responses = crop_responses(test_responses_by_trial, predictions)

        noise_variance = np.mean(  # mean across time
            np.var(cropped_responses, axis=1, ddof=ddof), axis=-1  # variance across repetitions
        )
        total_variance = np.var(cropped_responses, axis=(-1, -2))

        flattened_responses = cropped_responses.reshape(
            cropped_responses.shape[0], cropped_responses.shape[1] * cropped_responses.shape[2]
        )
        repeated_predictions = np.tile(predictions, [3, 1])
        repeated_predictions = repeated_predictions.transpose()
        mean_squared_error = np.mean((flattened_responses - repeated_predictions) ** 2, axis=1)
        fev = 1 - (mean_squared_error - noise_variance) / (total_variance - noise_variance)
        noise_variance_dict[data_key] = noise_variance
        total_variance_dict[data_key] = total_variance
        mse_dict[data_key] = mean_squared_error
        fev_dict[data_key] = fev
        repeated_predictions_dict[data_key] = repeated_predictions
    return noise_variance_dict, total_variance_dict, mse_dict, fev_dict, repeated_predictions_dict


def compute_oracle(responses, predictions=None):
    """
    computes oracle score for test responses
    :param responses: array of shape #cells x time x repetitions
    :param predictions: array of shape time x #cells
    :return: array of shape #cells containing oracle scores
    """
    if predictions is None:
        n_cells, _, n_reps = responses.shape
        oracle = np.zeros_like(responses)
        oracle_score = np.zeros(n_cells)
        for cell in range(n_cells):
            bool_mask = np.ones(n_reps, dtype=bool)
            for rep in range(n_reps):
                bool_mask[rep] = 0
                leave_one_out = responses[cell, :, bool_mask]
                oracle[cell, :, rep] = leave_one_out.mean(axis=0)
                bool_mask[rep] = 1
            # print(oracle[cell].shape)
            # print(responses[cell].shape)
            x = oracle[cell].reshape(-1, order="C")
            y = responses[cell].reshape(-1, order="C")
            oracle_score[cell] = corr(x, y)
        return oracle, oracle_score
    else:
        n_cells, _, n_reps = responses.shape
        oracle_score = np.zeros(n_cells)
        for cell in range(n_cells):
            x = np.tile(predictions[:, cell], n_reps)
            y = responses[cell].reshape(-1, order="F")
            oracle_score[cell] = corr(x, y)
        return x, oracle_score


def crop_responses(responses, predictions):
    """

    :param responses: array of responses, last axis is time
    :param predictions: array of predictions, first axis is time
    :return: responses cropped to same shape as predictions
    """

    lag = responses.shape[-1] - predictions.shape[0]
    return responses[..., lag:]
