import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

import openretina.eval.gold_standard as gs

EPS = gs.DEFAULT_EPS


def simulate(n_frames=200, n_repeats=3, n_neurons=4, p=1.3, phi=0.2, q=0.995, seed=0, means=None):
    """Responses from a log-normal prior over means and the zero-inflated Gamma likelihood; returns (y, means)."""
    rng = np.random.default_rng(seed)
    if means is None:
        means = np.exp(rng.normal(0.2, 0.5, size=(n_frames, n_neurons)))
    mu = np.repeat(means[:, None, :], n_repeats, axis=1)
    x = rng.gamma(mu ** (2 - p) / phi, phi * mu ** (p - 1))
    zero = rng.random(x.shape) > q
    return np.where(zero, rng.random(x.shape) * EPS, x + EPS), means


def true_mean_log_likelihood(y, means, p, phi):
    t = torch.as_tensor(y, dtype=gs.DTYPE)
    log_q, log_1mq = gs.zero_component_logs(t)
    noise = torch.full((y.shape[2],), math.log(phi), dtype=gs.DTYPE)
    mu = torch.as_tensor(means, dtype=gs.DTYPE)[:, None, :]
    return gs.zigp_log_prob(t, mu, noise, p, log_q, log_1mq).mean(dim=(0, 1)).numpy()


@pytest.mark.parametrize("family", ["zigp", "zil"])
@pytest.mark.parametrize("mu,noise,p", [(0.05, -1.0, 1.0), (1.0, -1.6, 1.5), (3.0, -0.5, 2.0)])
def test_likelihood_is_a_density(family, mu, noise, p):
    log_q, log_1mq = torch.tensor(math.log(0.97), dtype=gs.DTYPE), torch.tensor(math.log(0.03), dtype=gs.DTYPE)

    def density(y: float) -> float:
        args = (torch.tensor(y, dtype=gs.DTYPE), torch.tensor(mu, dtype=gs.DTYPE), torch.tensor(noise, dtype=gs.DTYPE))
        return float(gs.log_prob(family, *args, p, log_q, log_1mq).exp())

    total = quad(density, 0, EPS, limit=200)[0] + quad(density, EPS, np.inf, limit=400)[0]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_quadrature_matches_closed_form():
    """p = 2 (fixed shape k = 1/phi) with a Gamma(alpha0, beta0) prior on the rate k/mu has a closed-form predictive
    (the generalised beta prime of Lurz, Bashiri et al.'s Gamma_GS); the grid quadrature must reproduce it."""
    y_np, _ = simulate(n_frames=40, n_neurons=3, p=2.0, phi=0.25, q=1.0)
    y = torch.as_tensor(y_np, dtype=gs.DTYPE)
    log_q, log_1mq = gs.zero_component_logs(y)
    k = torch.full((3,), 4.0, dtype=gs.DTYPE)
    alpha0, beta0 = torch.full((3,), 3.0, dtype=gs.DTYPE), 2.0 * k
    log_mu, log_w = gs.mu_grid(y, n_grid=1024)
    log_rate = k[:, None].log() - log_mu
    log_prior_grid = (
        (alpha0 * beta0.log() - torch.lgamma(alpha0))[:, None]
        + (alpha0[:, None] - 1) * log_rate
        - beta0[:, None] * log_rate.exp()
        + k[:, None].log()
        - 2 * log_mu
    )
    quadrature = gs.loo_log_predictive(y, log_mu, log_w, log_prior_grid, "zigp", (1 / k).log(), 2.0, log_q, log_1mq)
    x = y - EPS
    a_post = alpha0 + (y.shape[1] - 1) * k
    b_post = beta0 + x.sum(dim=1, keepdim=True) - x
    closed = (
        (k - 1) * x.log()
        + torch.lgamma(a_post + k)
        + a_post * b_post.log()
        - torch.lgamma(k)
        - torch.lgamma(a_post)
        - (a_post + k) * (b_post + x).log()
        + log_q
    )
    assert (quadrature - closed).abs().max().item() < 1e-8


def test_gold_standard_between_null_and_true_means():
    p, phi = 1.3, 0.2
    y, means = simulate(n_frames=300, n_neurons=4, p=p, phi=phi)
    fit = gs.fit_exchangeable_gold_standard(y, prior="lognormal", p=p, n_grid=256)
    null = gs.null_log_likelihood(y, p=p)
    truth = true_mean_log_likelihood(y, means, p, phi)
    assert np.all(fit.log_likelihood > null)
    assert fit.log_likelihood.mean() < truth.mean()
    recovered = (fit.log_likelihood - null).mean() / (truth - null).mean()
    assert 0.3 < recovered < 1.0
    assert np.exp(fit.noise) == pytest.approx(np.full(4, phi), rel=0.25)


def test_shuffled_repeats_fall_to_the_null():
    """Each repeat's frames shuffled independently: the repeats share no stimulus, so the GS cannot beat the null."""
    y, _ = simulate(n_frames=300, n_neurons=3)
    rng = np.random.default_rng(1)
    for r in range(y.shape[1]):
        y[:, r] = y[rng.permutation(y.shape[0]), r]
    fit = gs.fit_exchangeable_gold_standard(y, n_grid=256)
    null = gs.null_log_likelihood(y)
    np.testing.assert_allclose(fit.log_likelihood, null, atol=0.01)


def test_null_does_not_depend_on_the_power():
    y, _ = simulate(n_frames=100, n_neurons=3)
    np.testing.assert_allclose(gs.null_log_likelihood(y, p=1.0), gs.null_log_likelihood(y, p=1.8), atol=1e-6)


def test_predictor_noise_recovers_the_power():
    y, means = simulate(n_frames=1500, n_repeats=1, n_neurons=20, p=1.5, phi=0.2, seed=3)
    noise = gs.fit_predictor_noise(y[:, 0], means, zero_fraction_from=y)
    assert noise.p == pytest.approx(1.5, abs=0.15)
    assert np.median(np.exp(noise.noise)) == pytest.approx(0.2, rel=0.15)


def test_true_means_score_like_the_simulation_and_nig():
    p, phi = 1.3, 0.2
    y, means = simulate(n_frames=400, n_neurons=4, p=p, phi=phi, seed=4)
    noise = gs.PredictorNoise(p=p, noise=np.full(4, math.log(phi)), log_likelihood_by_p={})
    np.testing.assert_allclose(gs.score_predictions(y, means, noise), true_mean_log_likelihood(y, means, p, phi))
    ll_null = gs.null_log_likelihood(y, p=p)
    ratio, per_neuron = gs.normalised_information_gain(ll_null + 0.5, ll_null, ll_null + 1.0)
    assert ratio == pytest.approx(0.5)
    np.testing.assert_allclose(per_neuron, 0.5)


def test_laplace_mode_is_stationary():
    y_np, _ = simulate(n_frames=60, n_neurons=2)
    y = torch.as_tensor(y_np, dtype=gs.DTYPE)
    log_q, log_1mq = gs.zero_component_logs(y)
    m = torch.full((2,), 0.2, dtype=gs.DTYPE)
    cov = gs.matern32_covariance(60, torch.full((2,), 4.0, dtype=gs.DTYPE), torch.full((2,), 0.25, dtype=gs.DTYPE))
    noise = torch.full((2,), math.log(0.2), dtype=gs.DTYPE)
    g, var = gs.laplace_posterior(y, [0, 1], m, cov, "zigp", noise, 1.3, log_q, log_1mq, kappa=2.0)
    g = g.clone().requires_grad_(True)
    obj = (
        gs.log_prob(
            "zigp",
            y[:, [0, 1]].permute(2, 0, 1),
            g.exp()[..., None],
            noise[:, None, None],
            1.3,
            log_q[:, None, None],
            log_1mq[:, None, None],
        ).sum()
        / 2.0
    )
    diff = g - m[:, None]
    obj = obj - 0.5 * torch.einsum("bt,bts,bs->", diff, torch.linalg.inv(cov), diff)
    (grad,) = torch.autograd.grad(obj, g)
    assert grad.abs().max().item() < 1e-4
    assert torch.all(var > 0)


def test_temporal_with_independent_frames_matches_exchangeable():
    """A near-zero length scale without tempering reduces the GP prior to the exchangeable log-normal prior; the
    Laplace and Gauss-Hermite approximations then agree with the exact quadrature to a few thousandths of a nat."""
    y, _ = simulate(n_frames=120, n_neurons=3, seed=5)
    exch = gs.fit_exchangeable_gold_standard(y, prior="lognormal", p=1.3, n_grid=256)
    temporal = gs.fit_temporal_gold_standard(y, exch, pairs=((0.05, 1.0),))
    np.testing.assert_allclose(temporal.log_likelihood, exch.log_likelihood, atol=5e-3)
