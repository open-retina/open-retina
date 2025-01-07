import numpy as np
from einops import rearrange, repeat
from jaxtyping import Float

from openretina.eval.metrics import correlation_numpy


def oracle_corr_jackknife(
    repeated_responses: Float[np.ndarray, "frames repeats neurons"],
) -> tuple[Float[np.ndarray, " neurons"], Float[np.ndarray, " frames repeats neurons"]]:
    """
    Adapted from neuralpredictors.
    Compute the oracle correlations per neuron by averaging over repeated responses in a leave one out fashion.
    Note that oracle_corr_jackknife underestimates the true oracle correlation.

    Args:
        repeated_responses (array-like): numpy array with shape (images/time, repeats, neuron responses).

    Returns:
        tuple: A tuple containing:
            - oracle_score (array): Oracle correlation for each neuron
            - oracle (array): Oracle responses for each neuron
    """

    loo_oracles = []
    for response_t in repeated_responses:
        num_repeats = response_t.shape[0]
        # Compute the oracle by averaging over all repeats except the current one
        # (add all, subtract current, divide by num_repeats - 1)
        oracle_t = (response_t.sum(axis=0, keepdims=True) - response_t) / (num_repeats - 1)
        oracle_t = np.nan_to_num(oracle_t)
        loo_oracles.append(oracle_t)
    oracle = np.stack(loo_oracles)

    oracle_score = correlation_numpy(
        rearrange(repeated_responses, "t r n -> (t r) n"),
        rearrange(oracle, "t r n -> (t r) n", t=repeated_responses.shape[0]),
        axis=0,
    )

    return oracle_score, oracle


def global_mean_oracle(
    responses: Float[np.ndarray, "frames repeats neurons"] | Float[np.ndarray, "frames neurons"],
) -> Float[np.ndarray, " neurons"]:
    """
    Compute the oracle correlation between each neuron's response and the global mean response.

    The global mean oracle correlation represents how well each neuron's activity can be predicted
    by the average response across all neurons at each time point.

    Args:
        responses (np.ndarray): Neural responses array. Can be either:
            - 3D array of shape (frames, repeats, neurons)
            - 2D array of shape (frames, neurons) which will be treated as single repeat
        return_oracle (bool, optional): If True, returns both correlation values and oracle responses.
            Defaults to False.

    Returns:
        - 1D array of shape (neurons,) containing correlation values for each neuron

    Note:
        The function automatically handles single-repeat data by adding a singleton dimension.
    """
    if responses.ndim == 2:
        responses = responses[:, None, :]

    global_mean_response = repeat(responses.mean(axis=2, keepdims=True), "t r _ -> t r n", n=responses.shape[2])

    oracle_mean_corr = correlation_numpy(
        rearrange(responses, "t r n -> (t r) n"),
        rearrange(global_mean_response, "t r n -> (t r) n", t=responses.shape[0]),
        axis=0,
    )

    return oracle_mean_corr
