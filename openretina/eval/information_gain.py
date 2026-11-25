"""Poisson log-likelihood information gain of a model over a constant rate, for single-trial responses.

The gain is the test-set version of the training objective (``PoissonLoss3d``): per neuron, the mean over frames and
repeats of the Poisson log likelihood of the model's rate, minus that of the neuron's mean test response, in bits.

Most datasets here are continuous (calcium-derived spike-rate estimates, normalised), not counts. The formula
``lam^y exp(-lam) / Gamma(y + 1)`` then still gives the training objective, but over real ``y >= 0`` it integrates to
``Z_P(lam) < 1`` rather than 1, so the gain contains a model-dependent term

    B = mean_t log Z_P(lam_t) - log Z_P(lam_null)

next to the log-likelihood ratio of the normalised densities (:func:`poisson_normaliser_bias`). For well-fitting models
B is negative and can be a third of the gain; it does not cancel against the ``Gamma(y + 1)`` term. For a normalised
score against an upper reference use :mod:`openretina.eval.gold_standard`, whose likelihood is a proper density.
"""

from functools import lru_cache

import numpy as np
from jaxtyping import Float
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gammaln

LN2 = float(np.log(2))
RATE_FLOOR = 1e-12
_GRID_MIN, _GRID_MAX, _GRID_SIZE = 1e-3, 50.0, 600


def poisson_log_likelihood(
    responses: Float[np.ndarray, "time trials neurons"], rates: Float[np.ndarray, "time neurons"]
) -> Float[np.ndarray, " neurons"]:
    """Mean Poisson log likelihood per neuron (nats per sample), with log(y!) extended to ``gammaln(y + 1)``.

    The rate of frame t is used for every trial of frame t; rates are floored at ``RATE_FLOOR``.
    """
    lam = np.maximum(rates, RATE_FLOOR)[:, None, :]
    return (responses * np.log(lam) - lam - gammaln(responses + 1)).mean(axis=(0, 1))


def poisson_information_gain(
    responses: Float[np.ndarray, "time trials neurons"], predictions: Float[np.ndarray, "time neurons"]
) -> Float[np.ndarray, " neurons"]:
    """Poisson log-likelihood gain of ``predictions`` over each neuron's mean response, in bits per sample.

    The constant baseline is fitted on the same responses, so it is the best constant and the gain is conservative.
    Positive means the model predicts the single trials better than a flat rate.
    """
    null = np.broadcast_to(responses.mean(axis=(0, 1), keepdims=True)[0], predictions.shape)
    return (poisson_log_likelihood(responses, predictions) - poisson_log_likelihood(responses, null)) / LN2


def _poisson_normaliser(rate: float) -> float:
    """Z_P(rate) = integral over y >= 0 of rate^y exp(-rate) / Gamma(y + 1), by numerical quadrature."""

    def integrand(y: float) -> float:
        return float(np.exp(y * np.log(rate) - rate - gammaln(y + 1)))

    return float(quad(integrand, 0, np.inf, limit=500)[0])


@lru_cache(maxsize=1)
def _log_normaliser_spline() -> CubicSpline:
    log_grid = np.linspace(np.log(_GRID_MIN), np.log(_GRID_MAX), _GRID_SIZE)
    return CubicSpline(log_grid, np.log([_poisson_normaliser(float(np.exp(x))) for x in log_grid]))


def poisson_log_normaliser(rates: np.ndarray) -> np.ndarray:
    """log Z_P(rate), elementwise, by a cubic spline in log rate through a cached quadrature table (error < 1e-8).

    Z_P < 1 everywhere and approaches 1 for large rates (log Z_P(1) = -0.182, log Z_P(5) = -0.002); rates outside
    [1e-3, 50] are clipped to the table, where log Z_P is within 1e-12 of its limit at the top end.
    """
    return _log_normaliser_spline()(np.log(np.clip(rates, _GRID_MIN, _GRID_MAX)))


def poisson_normaliser_bias(
    responses: Float[np.ndarray, "time trials neurons"], predictions: Float[np.ndarray, "time neurons"]
) -> Float[np.ndarray, " neurons"]:
    """The part of :func:`poisson_information_gain` that comes from the normalisers, in bits per sample.

    ``poisson_information_gain - poisson_normaliser_bias`` is the log-likelihood ratio of the normalised continuous
    densities ``f_P / Z_P``.
    """
    null_rate = responses.mean(axis=(0, 1))
    bias = poisson_log_normaliser(predictions).mean(axis=0) - poisson_log_normaliser(null_rate)
    return bias / LN2
