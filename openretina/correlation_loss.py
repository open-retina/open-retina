import torch


def calculate_pairwise_correlations(traces: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """ traces.shape = (..., time_steps, neurons) """
    if unbiased:
        traces = traces - traces.mean(dim=-2, keepdim=True)
    norm_traces = torch.nn.functional.normalize(traces, dim=-2)
    # pairwise_correlations.shape = (..., neurons, neurons)
    norm_traces_t = norm_traces.transpose(-2, -1)
    pairwise_correlations = norm_traces_t @ norm_traces
    # remove diagonal values that are 1.0 
    one_correlation_mask = pairwise_correlations < 0.999
    pairwise_correlations_no_diag = pairwise_correlations[one_correlation_mask]
    return pairwise_correlations_no_diag


def simple_mean_variance_loss(traces: torch.Tensor, unbiased: bool = True) -> torch.Tensor | float:
    pairwise_correlations = calculate_pairwise_correlations(traces, unbiased)
    # remove 1.0 correlations (those are present on the diagonal)
    mean = pairwise_correlations.mean()
    var = pairwise_correlations.var()

    expected_mean = 0.2
    expected_var = 0.2
    mean_loss = torch.abs(mean - expected_mean)
    var_loss = torch.abs(var - expected_var)
    loss = mean_loss + var_loss
    return loss
