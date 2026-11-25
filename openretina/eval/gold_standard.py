"""Bayesian gold standard for log-likelihood scores, after Lurz, Bashiri et al. (ICLR 2023).

A model's log-likelihood gain over a constant (null) predictor has no ceiling by itself. The gold standard (GS) predicts
each held-out repeat ``j`` of frame ``t`` by its posterior predictive given the other repeats,

    p(y_tj | y_t,-j) = ∫ f(y_tj | mu) p(mu | y_t,-j) dmu,     p(mu | y_t,-j) ∝ prior(mu) prod_{i != j} f(y_ti | mu),

with the prior's hyperparameters fitted per neuron on all frames and repeats by maximising the summed leave-one-out
(LOO) log predictive. Unlike the plug-in oracle (the mean of the other repeats), it shrinks toward what is typical for
the neuron, so it stays a sensible reference with only a few repeats. The normalised information gain of a model is

    nIG = (LL_model - LL_null) / (LL_GS - LL_null).

Likelihood. All scores use one proper density on y >= 0, so that every difference is a log-likelihood ratio (the
Poisson formula on continuous responses does not integrate to 1; see :mod:`openretina.eval.information_gain`):

  - ``"zigp"``: zero-inflated Gamma with a power variance. y <= eps has a uniform density with probability 1 - q_n;
    above it, y - eps ~ Gamma with mean mu and variance phi_n * mu^p. With p = 2 this is the paper's ZIG.
  - ``"zil"``: zero-inflated log-normal (the paper's ZIL), log(y - eps) ~ N(log mu - sigma^2/2, sigma^2).

``q_n`` is the neuron's fraction of samples above eps, the same for every predictor, so it cancels in comparisons.
The noise parameter (``log phi`` for zigp, ``log sigma`` for zil) belongs to the predictor: for the GS it is the
trial-to-trial noise; a model or the null also absorbs its own errors in it, and fits its own (:func:`fit_noise`).

Priors on mu (exchangeable over frames, the paper's version): ``"gamma"`` (shape a, rate b) or ``"lognormal"``
(log mu ~ N(m, s^2)). The predictive is computed by quadrature on a per-neuron grid over mu, exact to ~1e-10.

Temporal prior (:func:`fit_temporal_gold_standard`): a Gaussian process on log mu over time, posterior by a Laplace
approximation. Noise within a trial is usually correlated across neighbouring frames (calcium dynamics, interpolated
imaging), which an independent-frame likelihood counts several times; the conditioning likelihood is therefore
tempered by 1/kappa, and (length scale, kappa) is chosen per neuron by the LOO objective. On hoefling_2024 this gives
no gain over the exchangeable prior, because that noise correlation leaves neighbouring frames with little new
information.

All functions take responses as ``(time, repeats, neurons)`` arrays and work in float64 on the given device.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from numpy.polynomial.hermite import hermgauss

Family = Literal["zigp", "zil"]
Prior = Literal["gamma", "lognormal"]

DTYPE = torch.float64
DEFAULT_EPS = 0.01
DEFAULT_N_GRID = 768
MEAN_FLOOR = 1e-3
_LOG_2PI = math.log(2 * math.pi)
# (length scale in frames, tempering kappa) candidates for the temporal prior; (0.5, 1) is close to independent frames
DEFAULT_TEMPORAL_PAIRS: tuple[tuple[float, float], ...] = (
    (0.5, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 4.0),
    (4.0, 4.0),
    (4.0, 8.0),
    (8.0, 8.0),
    (8.0, 16.0),
)


# --------------------------------------------------------------------------------------------------------- likelihood
def zero_component_logs(
    responses: torch.Tensor, eps: float = DEFAULT_EPS
) -> tuple[Float[torch.Tensor, " neurons"], Float[torch.Tensor, " neurons"]]:
    """log q_n and log(1 - q_n), with q_n the fraction of a neuron's samples above ``eps``; neurons on the last axis."""
    flat = responses.reshape(-1, responses.shape[-1])
    q = (flat > eps).to(DTYPE).mean(dim=0).clamp(1e-4, 1 - 1e-4)
    return q.log(), (-q).log1p()


def zigp_log_prob(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_phi: torch.Tensor,
    p: float,
    log_q: torch.Tensor,
    log_1mq: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Zero-inflated Gamma log density; the slab y - eps has mean mu and variance exp(log_phi) * mu^p. Broadcasts."""
    shape = mu.pow(2 - p) / log_phi.exp()
    log_scale = log_phi + (p - 1) * mu.log()
    x = (y - eps).clamp_min(1e-12)
    slab = -torch.lgamma(shape) - shape * log_scale + (shape - 1) * x.log() - x * torch.exp(-log_scale)
    return torch.where(y > eps, log_q + slab, log_1mq - math.log(eps))


def zil_log_prob(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    log_q: torch.Tensor,
    log_1mq: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Zero-inflated log-normal log density; the slab y - eps has mean mu. Broadcasts."""
    s2 = (2 * log_sigma).exp()
    x = (y - eps).clamp_min(1e-12)
    lx = x.log()
    slab = -0.5 * _LOG_2PI - log_sigma - (lx - mu.log() + s2 / 2) ** 2 / (2 * s2) - lx
    return torch.where(y > eps, log_q + slab, log_1mq - math.log(eps))


def log_prob(
    family: Family,
    y: torch.Tensor,
    mu: torch.Tensor,
    noise: torch.Tensor,
    p: float,
    log_q: torch.Tensor,
    log_1mq: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Log density of ``family``; ``noise`` is log phi (zigp) or log sigma (zil); ``p`` is ignored for zil."""
    if family == "zigp":
        return zigp_log_prob(y, mu, noise, p, log_q, log_1mq, eps)
    if family == "zil":
        return zil_log_prob(y, mu, noise, log_q, log_1mq, eps)
    raise ValueError(f"Unknown family {family!r}")


def _fisher_weight(family: Family, mu: torch.Tensor, noise: torch.Tensor, p: float) -> torch.Tensor:
    """Fisher information of log mu from one slab sample: mu^2 / Var for zigp, 1 / sigma^2 for zil."""
    if family == "zigp":
        return mu.pow(2 - p) / noise.exp()
    return torch.exp(-2 * noise) * torch.ones_like(mu)


# ------------------------------------------------------------------------------------------------- exchangeable prior
def log_prior(prior: Prior, log_mu: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Log prior density over mu at ``log_mu``. gamma: h1 = log a, h2 = log b; lognormal: h1 = m, h2 = log s."""
    if prior == "gamma":
        a, b = h1.exp(), h2.exp()
        return a * h2 - torch.lgamma(a) + (a - 1) * log_mu - b * log_mu.exp()
    if prior == "lognormal":
        return -log_mu - h2 - 0.5 * _LOG_2PI - (log_mu - h1) ** 2 / (2 * (2 * h2).exp())
    raise ValueError(f"Unknown prior {prior!r}")


def mu_grid(
    responses: Float[torch.Tensor, "time repeats neurons"], n_grid: int = DEFAULT_N_GRID
) -> tuple[Float[torch.Tensor, "neurons grid"], Float[torch.Tensor, "neurons grid"]]:
    """Per-neuron log-spaced grid over mu, 1e-3 to 20 * max response + 10, and log trapezoid weights for d mu.

    The wide upper end keeps heavy-tailed posteriors inside the grid; the trapezoid rule in log mu then converges
    exponentially for these smooth integrands.
    """
    hi = (20 * responses.amax(dim=(0, 1)) + 10).log()
    lo = torch.full_like(hi, math.log(MEAN_FLOOR))
    u = torch.linspace(0, 1, n_grid, dtype=DTYPE, device=responses.device)
    log_mu = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    step = (hi - lo) / (n_grid - 1)
    trapezoid = torch.ones(n_grid, dtype=DTYPE, device=responses.device)
    trapezoid[0] = trapezoid[-1] = 0.5
    return log_mu, (step[:, None] * trapezoid[None, :]).log() + log_mu  # d mu = mu d log mu


def loo_log_predictive(
    responses: Float[torch.Tensor, "time repeats neurons"],
    log_mu: Float[torch.Tensor, "neurons grid"],
    log_weights: Float[torch.Tensor, "neurons grid"],
    log_prior_grid: Float[torch.Tensor, "neurons grid"],
    family: Family,
    noise: Float[torch.Tensor, " neurons"],
    p: float,
    log_q: Float[torch.Tensor, " neurons"],
    log_1mq: Float[torch.Tensor, " neurons"],
    eps: float = DEFAULT_EPS,
) -> Float[torch.Tensor, "time repeats neurons"]:
    """Leave-one-repeat-out log posterior predictive of every sample, by quadrature over the mu grid."""
    ll = log_prob(
        family,
        responses[..., None],
        log_mu.exp()[None, None],
        noise[None, None, :, None],
        p,
        log_q[None, None, :, None],
        log_1mq[None, None, :, None],
        eps,
    )  # (time, repeats, neurons, grid)
    base = log_weights + log_prior_grid
    total = ll.sum(dim=1)
    numerator = torch.logsumexp(base[None] + total, dim=-1)
    denominator = torch.logsumexp(base[None, None] + total[:, None] - ll, dim=-1)
    return numerator[:, None] - denominator


def _initial_params(responses: torch.Tensor, family: Family, prior: Prior, p: float) -> list[torch.Tensor]:
    """Moment-based start: prior from the between-frame variance, noise from the within-frame variance."""
    frame_mean, frame_var = responses.mean(dim=1), responses.var(dim=1)
    grand_mean = frame_mean.mean(0)
    signal_var = torch.maximum(frame_mean.var(0) - frame_var.mean(0) / responses.shape[1], 0.05 * grand_mean**2)
    if prior == "gamma":
        h1, h2 = (grand_mean**2 / signal_var).log(), (grand_mean / signal_var).log()
    else:
        s2 = (1 + signal_var / grand_mean**2).log()
        h1, h2 = grand_mean.log() - s2 / 2, 0.5 * s2.log()
    m = frame_mean.clamp_min(0.05)
    if family == "zigp":
        noise = (frame_var.mean(0) / m.pow(p).mean(0)).clamp(1e-3, 10).log()
    else:
        noise = 0.5 * (1 + frame_var.mean(0) / (m**2).mean(0)).log()
    return [h1, h2, noise]


@dataclass
class ExchangeableGoldStandard:
    """Per-neuron hyperparameters and LOO log likelihood of the exchangeable-prior gold standard."""

    family: Family
    prior: Prior
    p: float
    eps: float
    h1: np.ndarray  # log a (gamma) or m (lognormal)
    h2: np.ndarray  # log b (gamma) or log s (lognormal)
    noise: np.ndarray  # log phi (zigp) or log sigma (zil): the trial-to-trial noise
    log_likelihood: np.ndarray  # mean LOO log predictive per neuron, nats per sample

    def log_mean_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Prior mean and variance of log mu, per neuron."""
        if self.prior == "lognormal":
            return self.h1, np.exp(2 * self.h2)
        a = torch.as_tensor(np.exp(self.h1))
        return torch.special.digamma(a).numpy() - self.h2, torch.special.polygamma(1, a).numpy()


def fit_exchangeable_gold_standard(
    responses: Float[np.ndarray, "time repeats neurons"],
    family: Family = "zigp",
    prior: Prior = "lognormal",
    p: float = 1.05,
    eps: float = DEFAULT_EPS,
    n_grid: int = DEFAULT_N_GRID,
    batch_size: int = 24,
    max_iter: int = 150,
    device: str | torch.device = "cpu",
) -> ExchangeableGoldStandard:
    """Fit the gold standard with an exchangeable prior, neuron by neuron (in batches).

    The hyperparameters and the noise maximise the summed LOO log predictive over all frames and repeats; ``p`` is
    fixed (choose it with :func:`select_power`). Memory grows with time * repeats * batch_size * n_grid.
    """
    out_params, out_ll = [], []
    for start in range(0, responses.shape[2], batch_size):
        y = torch.as_tensor(responses[:, :, start : start + batch_size], dtype=DTYPE, device=device)
        log_q, log_1mq = zero_component_logs(y, eps)
        log_mu, log_w = mu_grid(y, n_grid)
        params = [t.clone().requires_grad_(True) for t in _initial_params(y, family, prior, p)]
        optimiser = torch.optim.LBFGS(
            params,
            lr=1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            history_size=20,
        )

        def loo(params: list[torch.Tensor] = params) -> torch.Tensor:
            h1, h2, noise = params
            lp = log_prior(prior, log_mu, h1[:, None], h2[:, None])
            return loo_log_predictive(y, log_mu, log_w, lp, family, noise, p, log_q, log_1mq, eps)

        def closure(optimiser: torch.optim.LBFGS = optimiser, loo=loo) -> torch.Tensor:
            optimiser.zero_grad()
            loss = -loo().sum()
            loss.backward()
            return loss

        optimiser.step(closure)
        with torch.no_grad():
            out_ll.append(loo().mean(dim=(0, 1)).cpu().numpy())
        out_params.append(torch.stack([t.detach() for t in params]).cpu().numpy())
    params_np = np.concatenate(out_params, axis=1)
    return ExchangeableGoldStandard(
        family=family,
        prior=prior,
        p=p,
        eps=eps,
        h1=params_np[0],
        h2=params_np[1],
        noise=params_np[2],
        log_likelihood=np.concatenate(out_ll),
    )


def select_power(
    responses: Float[np.ndarray, "time repeats neurons"],
    p_values: tuple[float, ...] = tuple(np.round(np.arange(1.0, 2.01, 0.1), 2)),
    family: Family = "zigp",
    prior: Prior = "lognormal",
    **fit_kwargs,
) -> dict[float, float]:
    """Summed LOO log likelihood of the gold standard for each candidate power ``p`` (use a subset of neurons)."""
    return {
        float(p): float(
            fit_exchangeable_gold_standard(responses, family, prior, float(p), **fit_kwargs).log_likelihood.sum()
        )
        for p in p_values
    }


# ------------------------------------------------------------------------------------------------- null and models
def fit_noise(
    responses: torch.Tensor,
    means: torch.Tensor,
    family: Family,
    p: float,
    log_q: torch.Tensor,
    log_1mq: torch.Tensor,
    init: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Maximum-likelihood noise parameter per neuron for ``responses`` around ``means`` (neurons on the last axis)."""
    noise = init.clone().to(DTYPE).requires_grad_(True)
    optimiser = torch.optim.LBFGS([noise], lr=1, max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = -log_prob(family, responses, means, noise, p, log_q, log_1mq, eps).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return noise.detach()


def null_log_likelihood(
    responses: Float[np.ndarray, "time repeats neurons"],
    family: Family = "zigp",
    p: float = 1.05,
    eps: float = DEFAULT_EPS,
    device: str | torch.device = "cpu",
) -> Float[np.ndarray, " neurons"]:
    """Log likelihood (nats per sample) of the null: a constant mean and its own noise per neuron, both by ML.

    For zigp the result does not depend on ``p``: with a constant mean, phi * mu^p is one free variance per neuron.
    """
    y = torch.as_tensor(responses, dtype=DTYPE, device=device)
    log_q, log_1mq = zero_component_logs(y, eps)
    log_mean = y.mean(dim=(0, 1)).clamp_min(MEAN_FLOOR).log().clone().requires_grad_(True)
    noise = torch.zeros_like(log_mean).requires_grad_(True)
    optimiser = torch.optim.LBFGS(
        [log_mean, noise], lr=1, max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-9
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = -log_prob(family, y, log_mean.exp(), noise, p, log_q, log_1mq, eps).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        return log_prob(family, y, log_mean.exp(), noise, p, log_q, log_1mq, eps).mean(dim=(0, 1)).cpu().numpy()


@dataclass
class PredictorNoise:
    """A model's variance function, fitted on held-out (validation) data: shared power p and per-neuron noise."""

    p: float
    noise: np.ndarray
    log_likelihood_by_p: dict[float, float]


def fit_predictor_noise(
    responses: Float[np.ndarray, "time neurons"],
    predictions: Float[np.ndarray, "time neurons"],
    zero_fraction_from: Float[np.ndarray, "time repeats neurons"],
    family: Family = "zigp",
    p_values: tuple[float, ...] = tuple(np.round(np.arange(1.0, 2.01, 0.1), 2)),
    eps: float = DEFAULT_EPS,
    device: str | torch.device = "cpu",
) -> PredictorNoise:
    """Fit a model's own variance function on data it was not trained on (e.g. the validation split).

    A model's errors usually grow with the mean more steeply than trial-to-trial noise, so it gets its own ``p``
    (shared over neurons, from ``p_values``) and per-neuron noise. ``zero_fraction_from`` gives q_n (use the test
    responses, so that the zero component is the same for every predictor); it does not move p or the noise.
    """
    y = torch.as_tensor(responses, dtype=DTYPE, device=device)
    mu = torch.as_tensor(predictions, dtype=DTYPE, device=device).clamp_min(MEAN_FLOOR)
    log_q, log_1mq = zero_component_logs(torch.as_tensor(zero_fraction_from, dtype=DTYPE, device=device), eps)
    init = torch.zeros(y.shape[-1], dtype=DTYPE, device=device)
    best: tuple[float, torch.Tensor, float] | None = None
    scores: dict[float, float] = {}
    for p in p_values:
        noise = fit_noise(y, mu, family, float(p), log_q, log_1mq, init, eps)
        with torch.no_grad():
            total = float(log_prob(family, y, mu, noise, float(p), log_q, log_1mq, eps).sum())
        scores[float(p)] = total
        if best is None or total > best[2]:
            best = (float(p), noise, total)
    assert best is not None
    return PredictorNoise(p=best[0], noise=best[1].cpu().numpy(), log_likelihood_by_p=scores)


def score_predictions(
    responses: Float[np.ndarray, "time repeats neurons"],
    predictions: Float[np.ndarray, "time neurons"],
    noise_model: PredictorNoise,
    family: Family = "zigp",
    eps: float = DEFAULT_EPS,
    device: str | torch.device = "cpu",
) -> Float[np.ndarray, " neurons"]:
    """Log likelihood (nats per sample) of the test responses under a model's predictions and its fitted noise."""
    y = torch.as_tensor(responses, dtype=DTYPE, device=device)
    log_q, log_1mq = zero_component_logs(y, eps)
    mu = torch.as_tensor(predictions, dtype=DTYPE, device=device).clamp_min(MEAN_FLOOR)[:, None, :]
    noise = torch.as_tensor(noise_model.noise, dtype=DTYPE, device=device)
    with torch.no_grad():
        return log_prob(family, y, mu, noise, noise_model.p, log_q, log_1mq, eps).mean(dim=(0, 1)).cpu().numpy()


def normalised_information_gain(
    ll_model: np.ndarray, ll_null: np.ndarray, ll_gold_standard: np.ndarray
) -> tuple[float, np.ndarray]:
    """nIG as the ratio of mean gains over neurons, and per neuron (unstable where a neuron's GS gain is near 0)."""
    gain, gs_gain = ll_model - ll_null, ll_gold_standard - ll_null
    return float(gain.mean() / gs_gain.mean()), gain / gs_gain


# ---------------------------------------------------------------------------------------------------- temporal prior
def matern32_covariance(
    n_frames: int, length_scale: torch.Tensor, variance: torch.Tensor
) -> Float[torch.Tensor, "neurons time time"]:
    """Matérn-3/2 covariance over frames, per neuron, with a small jitter on the diagonal."""
    t = torch.arange(n_frames, dtype=DTYPE, device=length_scale.device)
    r = (t[:, None] - t[None, :]).abs()[None] * (math.sqrt(3) / length_scale[:, None, None])
    eye = torch.eye(n_frames, dtype=DTYPE, device=length_scale.device)
    return variance[:, None, None] * ((1 + r) * torch.exp(-r) + 1e-6 * eye)


def laplace_posterior(
    responses: Float[torch.Tensor, "time repeats neurons"],
    used_repeats: list[int],
    prior_mean: Float[torch.Tensor, " neurons"],
    covariance: Float[torch.Tensor, "neurons time time"],
    family: Family,
    noise: Float[torch.Tensor, " neurons"],
    p: float,
    log_q: Float[torch.Tensor, " neurons"],
    log_1mq: Float[torch.Tensor, " neurons"],
    kappa: float = 1.0,
    eps: float = DEFAULT_EPS,
    max_iter: int = 40,
) -> tuple[Float[torch.Tensor, "neurons time"], Float[torch.Tensor, "neurons time"]]:
    """Laplace approximation to p(g | y[:, used]) ∝ N(g; m, K) prod f(y | e^g)^(1/kappa), g = log mu over time.

    Newton iterations in the stable form of Rasmussen & Williams (2006, Alg. 3.1), with the Fisher weight (always
    positive) in place of the Hessian and the exact gradient, so the mode is exact; each step is damped per neuron
    until the objective improves. Returns the mode and the marginal posterior variances.
    """
    n_neurons, n_frames = covariance.shape[0], covariance.shape[1]
    y = responses[:, used_repeats].permute(2, 0, 1)  # (neurons, time, used)
    eye = torch.eye(n_frames, dtype=DTYPE, device=covariance.device)
    q = log_q.exp()[:, None]

    def log_lik_and_grad(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = g.detach().requires_grad_(True)
        with torch.enable_grad():
            ll = (
                log_prob(
                    family,
                    y,
                    g.exp()[..., None],
                    noise[:, None, None],
                    p,
                    log_q[:, None, None],
                    log_1mq[:, None, None],
                    eps,
                ).sum(dim=(1, 2))
                / kappa
            )
            (grad,) = torch.autograd.grad(ll.sum(), g)
        return ll.detach(), grad

    def factor(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = len(used_repeats) * q * _fisher_weight(family, g.exp(), noise[:, None], p) / kappa
        sw = w.sqrt()
        return w, sw, torch.linalg.cholesky(eye[None] + sw[:, :, None] * covariance * sw[:, None, :])

    f = torch.zeros(n_neurons, n_frames, dtype=DTYPE, device=covariance.device)
    a = torch.zeros_like(f)
    psi_old = torch.full((n_neurons,), -float("inf"), dtype=DTYPE, device=covariance.device)
    for _ in range(max_iter):
        g = f + prior_mean[:, None]
        _, grad = log_lik_and_grad(g)
        w, sw, chol = factor(g)
        b = w * f + grad
        v = torch.cholesky_solve((sw * (covariance @ b[..., None])[..., 0])[..., None], chol)[..., 0]
        a_new = b - sw * v
        f_new = (covariance @ a_new[..., None])[..., 0]
        step = torch.ones(n_neurons, 1, dtype=DTYPE, device=covariance.device)
        for _ in range(8):
            f_try, a_try = f + step * (f_new - f), a + step * (a_new - a)
            ll_try, _ = log_lik_and_grad(f_try + prior_mean[:, None])
            psi = ll_try - 0.5 * (a_try * f_try).sum(-1)
            worse = ~(psi >= psi_old - 1e-9)
            if not worse.any():
                break
            step = torch.where(worse[:, None], step / 2, step)
        f, a = f_try, a_try
        converged = (psi - psi_old).abs().max().item() < 1e-7
        psi_old = psi
        if converged:
            break
    g = f + prior_mean[:, None]
    _, sw, chol = factor(g)
    v = torch.linalg.solve_triangular(chol, sw[:, :, None] * covariance, upper=False)
    variance = (torch.diagonal(covariance, dim1=1, dim2=2) - (v**2).sum(dim=1)).clamp_min(1e-10)
    return g, variance


def gauss_hermite_log_predictive(
    y: Float[torch.Tensor, "neurons time"],
    g_mean: Float[torch.Tensor, "neurons time"],
    g_variance: Float[torch.Tensor, "neurons time"],
    family: Family,
    noise: Float[torch.Tensor, " neurons"],
    p: float,
    log_q: Float[torch.Tensor, " neurons"],
    log_1mq: Float[torch.Tensor, " neurons"],
    eps: float = DEFAULT_EPS,
    n_nodes: int = 32,
) -> Float[torch.Tensor, "neurons time"]:
    """log of the integral of f(y | e^g) N(g; mean, variance) dg, by Gauss-Hermite quadrature."""
    x, w = hermgauss(n_nodes)
    nodes = torch.as_tensor(x, dtype=DTYPE, device=g_mean.device)
    log_w = torch.as_tensor(np.log(w) - 0.5 * np.log(np.pi), dtype=DTYPE, device=g_mean.device)
    g = g_mean[..., None] + math.sqrt(2) * g_variance.sqrt()[..., None] * nodes
    ll = log_prob(
        family, y[..., None], g.exp(), noise[:, None, None], p, log_q[:, None, None], log_1mq[:, None, None], eps
    )
    return torch.logsumexp(ll + log_w, dim=-1)


@dataclass
class TemporalGoldStandard:
    """Per-neuron (length scale, kappa) and LOO log likelihood of the temporal-prior gold standard."""

    length_scale: np.ndarray
    kappa: np.ndarray
    log_likelihood: np.ndarray  # nats per sample


def fit_temporal_gold_standard(
    responses: Float[np.ndarray, "time repeats neurons"],
    exchangeable: ExchangeableGoldStandard,
    pairs: tuple[tuple[float, float], ...] = DEFAULT_TEMPORAL_PAIRS,
    batch_size: int = 48,
    device: str | torch.device = "cpu",
) -> TemporalGoldStandard:
    """Gold standard with a GP prior on log mu over time, built on a fitted exchangeable gold standard.

    The GP mean and variance are the exchangeable prior's moments of log mu, and the noise is its trial-to-trial
    noise. For each candidate (length scale, kappa) the LOO log predictive is computed with a Laplace posterior over
    the other repeats and Gauss-Hermite quadrature over each frame's marginal; per neuron the best pair is kept.
    """
    family, p, eps = exchangeable.family, exchangeable.p, exchangeable.eps
    prior_mean_np, prior_var_np = exchangeable.log_mean_moments()
    n_frames, n_repeats, n_neurons = responses.shape
    out_ell, out_kappa, out_ll = [], [], []
    for start in range(0, n_neurons, batch_size):
        sl = slice(start, start + batch_size)
        y = torch.as_tensor(responses[:, :, sl], dtype=DTYPE, device=device)
        b = y.shape[2]
        log_q, log_1mq = zero_component_logs(y, eps)
        m = torch.as_tensor(prior_mean_np[sl], dtype=DTYPE, device=device)
        s2 = torch.as_tensor(prior_var_np[sl], dtype=DTYPE, device=device)
        noise = torch.as_tensor(exchangeable.noise[sl], dtype=DTYPE, device=device)
        best = torch.full((b,), -float("inf"), dtype=DTYPE, device=device)
        ell, kap, ll_best = (torch.zeros(b, dtype=DTYPE, device=device) for _ in range(3))
        for length_scale, kappa in pairs:
            cov = matern32_covariance(n_frames, torch.full((b,), length_scale, dtype=DTYPE, device=device), s2)
            lpp = torch.zeros(n_frames, n_repeats, b, dtype=DTYPE, device=device)
            for j in range(n_repeats):
                others = [i for i in range(n_repeats) if i != j]
                g, var = laplace_posterior(y, others, m, cov, family, noise, p, log_q, log_1mq, kappa, eps)
                lpp[:, j] = gauss_hermite_log_predictive(y[:, j].T, g, var, family, noise, p, log_q, log_1mq, eps).T
            mean_ll = lpp.mean(dim=(0, 1))
            better = mean_ll > best
            best = torch.where(better, mean_ll, best)
            ell = torch.where(better, torch.full_like(ell, length_scale), ell)
            kap = torch.where(better, torch.full_like(kap, kappa), kap)
            ll_best = torch.where(better, mean_ll, ll_best)
        out_ell.append(ell.cpu().numpy())
        out_kappa.append(kap.cpu().numpy())
        out_ll.append(ll_best.cpu().numpy())
    return TemporalGoldStandard(np.concatenate(out_ell), np.concatenate(out_kappa), np.concatenate(out_ll))
