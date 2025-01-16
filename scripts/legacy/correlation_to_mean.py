import numpy as np
import torch

from openretina.legacy.metrics import correlation_numpy


def get_mean_responses(loader, data_key):
    target, mean_resp = torch.empty(0), torch.empty(0)
    for _, responses in loader[data_key]:  # necessary for group assignments
        mean_resp = torch.cat(
            (mean_resp, responses.mean(axis=2).unsqueeze(2).repeat(1, 1, responses.shape[2]).detach().cpu()), dim=0
        )
        target = torch.cat((target, responses.detach().cpu()), dim=0)
    mean_resp = mean_resp.numpy()
    target = target.numpy()
    lag = target.shape[1] - mean_resp.shape[1]

    return target[:, lag:, ...], mean_resp


def corr_to_average(loader, avg: bool = True):
    n_neurons, correlations_sum = 0, 0
    if not avg:
        all_correlations = np.array([])

    for data_key in loader:
        # Usually used when we have a batch dim so we have to unsqueeze it
        target, mean_resp = get_mean_responses(loader, data_key)

        ret = correlation_numpy(target, mean_resp, axis=1)
        ret = ret.mean(axis=0)

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += mean_resp.shape[-1]
            correlations_sum += ret.sum()

    return correlations_sum / n_neurons if avg else all_correlations
