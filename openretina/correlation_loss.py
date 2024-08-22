import torch


def calculate_pairwise_correlations(traces: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """ traces.shape = (..., time_steps, neurons) """
    if unbiased:
        traces = traces - traces.mean(dim=-2, keepdim=True)
    norm_traces = torch.nn.functional.normalize(traces, dim=-2)
    # pairwise_correlations.shape = (..., neurons, neurons)
    norm_traces_t = norm_traces.transpose(-2, -1)
    pairwise_correlations = norm_traces_t @ norm_traces
    # Remove diagonal and duplicate values (pairwise correlations is a symmetrical matrix)
    upper_tri_indices = torch.triu_indices(pairwise_correlations.size(-2), pairwise_correlations.size(-1), offset=1)
    unique_correlations = pairwise_correlations[..., upper_tri_indices[0], upper_tri_indices[1]]
    return unique_correlations


def simple_mean_variance_loss(
    traces: torch.Tensor,
    unbiased: bool = True,
    expected_mean: float = 0.0,
    expected_var: float | None = None,
) -> torch.Tensor | float:
    pairwise_correlations = calculate_pairwise_correlations(traces, unbiased)
    # remove 1.0 correlations (those are present on the diagonal)
    mean = pairwise_correlations.mean()

    loss = torch.abs(mean - expected_mean)
    if expected_var is not None:
        var = pairwise_correlations.var()
        loss += torch.abs(var - expected_var)
    return loss
