import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

from openretina.eval.information_gain import (
    poisson_information_gain,
    poisson_log_likelihood,
    poisson_log_normaliser,
    poisson_normaliser_bias,
)


def responses(seed=0, n_frames=300, n_repeats=3, n_neurons=5):
    rng = np.random.default_rng(seed)
    rates = rng.gamma(2.0, 0.7, size=(n_frames, n_neurons))
    return rng.gamma(4.0, rates[:, None, :] / 4.0, size=(n_frames, n_repeats, n_neurons)), rates


def test_log_likelihood_matches_formula():
    y, rates = responses(n_frames=20, n_neurons=2)
    expected = (y * np.log(rates[:, None]) - rates[:, None] - gammaln(y + 1)).mean(axis=(0, 1))
    np.testing.assert_allclose(poisson_log_likelihood(y, rates), expected)


def test_gain_of_the_constant_rate_is_zero():
    y, _ = responses()
    np.testing.assert_allclose(poisson_information_gain(y, np.broadcast_to(y.mean(axis=(0, 1)), y.shape[::2])), 0)


def test_gain_of_the_true_rates_is_positive():
    y, rates = responses()
    assert np.all(poisson_information_gain(y, rates) > 0)


@pytest.mark.parametrize("rate,expected", [(0.3, -0.535), (1.0, -0.182), (2.0, -0.054), (5.0, -0.002)])
def test_log_normaliser(rate, expected):
    direct = np.log(quad(lambda y: np.exp(y * np.log(rate) - rate - gammaln(y + 1)), 0, np.inf, limit=500)[0])
    assert float(poisson_log_normaliser(np.array(rate))) == pytest.approx(direct, abs=1e-6)
    assert direct == pytest.approx(expected, abs=1e-3)


def test_bias_is_zero_for_the_constant_rate_and_explains_the_gap():
    y, rates = responses()
    null = np.broadcast_to(y.mean(axis=(0, 1)), rates.shape)
    np.testing.assert_allclose(poisson_normaliser_bias(y, null), 0, atol=1e-12)
    # gain minus bias = log-likelihood ratio of the normalised densities f_P / Z_P
    normalised = (
        poisson_log_likelihood(y, rates)
        - poisson_log_normaliser(rates).mean(axis=0)
        - poisson_log_likelihood(y, null)
        + poisson_log_normaliser(null).mean(axis=0)
    ) / np.log(2)
    np.testing.assert_allclose(poisson_information_gain(y, rates) - poisson_normaliser_bias(y, rates), normalised)
