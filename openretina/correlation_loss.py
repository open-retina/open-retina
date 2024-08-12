import torch


def calculate_pairwise_correlations(traces: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """ traces.shape = (..., time_steps, neurons) """
    if unbiased:
        traces -= traces.mean(dim=-2, keepdim=True)
    norm_traces = torch.nn.functional.normalize(traces, dim=-2)
    # pairwise_correlations.shape = (..., neurons, neurons)
    pairwise_correlations = norm_traces.transpose(-2, -1) @ norm_traces
    return pairwise_correlations


def kl_divergence_loss(traces: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    pairwise_correlations = calculate_pairwise_correlations(traces, unbiased)
    # remove 1.0 correlations (those are present on the diagonal)
    pairwise_correlations = pairwise_correlations[pairwise_correlations >= 1.0]
    mean = pairwise_correlations.mean()
    loss = mean.abs()
    return loss
