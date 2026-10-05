"""Free entropy, free Fisher information, transport distances, and the
functional inequalities of Section 4.

Definitions follow the paper:

*   Free entropy (logarithmic energy)

    .. math:: \\chi(\\mu) = \\iint \\log|x-y|\\, d\\mu(x)\\, d\\mu(y),

*   free Fisher information ``Phi(mu) = tau[xi_mu^2] = int xi_mu^2 d mu``,

*   free energy ``F(mu) = int x^2/2 d mu - chi(mu)``, minimised by the
    semicircular law ``gamma`` of unit variance,

*   relative quantities ``D(mu || gamma) = F(mu) - F(gamma)`` and
    ``I(mu || gamma) = int (xi_mu(x) - x)^2 d mu``.

The one-dimensional free Wasserstein distance coincides with the classical one
by Theorem 2.5 (Biane--Voiculescu), so ``W_2`` is computed from quantiles.

The logarithmic energy is singular on the diagonal.  For a discrete measure the
diagonal terms are excluded and the remaining sum is normalised by ``n(n-1)``,
which is the standard unbiased estimator of ``chi``; for a density on a grid the
double integral is computed with the diagonal handled by the limit of the
difference quotient, exactly as in the proof of Theorem 8.2.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "log_energy_samples",
    "log_energy_density",
    "free_entropy",
    "free_fisher",
    "free_energy",
    "relative_free_energy",
    "relative_fisher",
    "w1_samples",
    "w2_samples",
    "w1_density_vs_samples",
    "semicircle_density",
    "semicircle_free_energy",
    "inequality_ratios",
]


# ----------------------------------------------------------------------------
# logarithmic energy
# ----------------------------------------------------------------------------
def log_energy_samples(x):
    """Unbiased estimator of ``chi(mu)`` from a sample, diagonal excluded."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    d = np.abs(x[:, None] - x[None, :])
    iu = np.triu_indices(n, k=1)
    return 2.0 * np.sum(np.log(d[iu])) / (n * (n - 1))


def log_energy_density(x, psi):
    """``chi`` for a density given on a uniform grid, by trapezoidal quadrature.

    The diagonal singularity is integrable; the grid cell containing it is
    handled by the local average ``log(h/2) - 1``, the exact value of
    ``h^{-1} int_{-h/2}^{h/2} log|u| du``.
    """
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=float)
    h = x[1] - x[0]
    d = np.abs(x[:, None] - x[None, :])
    with np.errstate(divide="ignore"):
        K = np.log(d)
    np.fill_diagonal(K, np.log(h / 2.0) - 1.0)
    w = psi * h
    return float(w @ K @ w)


def free_entropy(x, psi=None):
    """``chi(mu)``; from samples if ``psi`` is None, otherwise from a density."""
    if psi is None:
        return log_energy_samples(x)
    return log_energy_density(x, psi)


def free_fisher(x, psi, xi):
    """``Phi(mu) = int xi^2 d mu`` on a uniform grid."""
    x = np.asarray(x, dtype=float)
    return float(np.trapezoid(np.asarray(xi) ** 2 * np.asarray(psi), x))


def free_energy(x, psi):
    """``F(mu) = int x^2/2 d mu - chi(mu)``."""
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=float)
    second = float(np.trapezoid(0.5 * x**2 * psi, x))
    return second - log_energy_density(x, psi)


def semicircle_density(x, v=1.0):
    """Density of ``gamma_v`` on ``[-2 sqrt v, 2 sqrt v]``."""
    x = np.asarray(x, dtype=float)
    r2 = 4.0 * v
    out = np.zeros_like(x)
    m = x**2 < r2
    out[m] = np.sqrt(r2 - x[m] ** 2) / (2.0 * np.pi * v)
    return out


def semicircle_free_energy(v=1.0):
    """``F(gamma_v)`` in closed form: ``v/2 - (log v)/2 - 1/4 ... ``.

    For the unit-variance semicircular law, ``chi(gamma_1) = -1/4`` and the
    second-moment term is ``1/2``, so ``F(gamma_1) = 3/4``.  The general case
    follows from ``chi(gamma_v) = (log v)/2 - 1/4``.
    """
    return 0.5 * v - (0.5 * np.log(v) - 0.25)


def relative_free_energy(x, psi, v=1.0):
    """``D(mu || gamma_v) = F(mu) - F(gamma_v) >= 0``."""
    return free_energy(x, psi) - semicircle_free_energy(v)


def relative_fisher(x, psi, xi):
    """``I(mu || gamma) = int (xi_mu(x) - x)^2 d mu``."""
    x = np.asarray(x, dtype=float)
    return float(np.trapezoid((np.asarray(xi) - x) ** 2 * np.asarray(psi), x))


# ----------------------------------------------------------------------------
# transport distances (one dimension: quantile coupling is optimal)
# ----------------------------------------------------------------------------
def w1_samples(a, b):
    """``W_1`` between two empirical measures of equal size, or by quantiles."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == b.size:
        return float(np.mean(np.abs(a - b)))
    q = np.linspace(0.0, 1.0, max(a.size, b.size, 2048))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def w2_samples(a, b):
    """``W_2`` between two samples via the quantile coupling."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == b.size:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    q = np.linspace(0.0, 1.0, max(a.size, b.size, 2048))
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def _quantiles_from_density(x, psi, q):
    x = np.asarray(x, dtype=float)
    psi = np.clip(np.asarray(psi, dtype=float), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (psi[1:] + psi[:-1]) * np.diff(x))])
    cdf = cdf / cdf[-1]
    return np.interp(q, cdf, x)


def w1_density_vs_samples(x, psi, samples, n_quantiles=4096):
    """``W_1`` between a density on a grid and an empirical sample."""
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    qa = _quantiles_from_density(x, psi, q)
    qb = np.quantile(np.asarray(samples, dtype=float).ravel(), q)
    return float(np.mean(np.abs(qa - qb)))


def w2_density_vs_density(x, psi, phi, n_quantiles=4096):
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    qa = _quantiles_from_density(x, psi, q)
    qb = _quantiles_from_density(x, phi, q)
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


# ----------------------------------------------------------------------------
# the three inequalities of Section 4
# ----------------------------------------------------------------------------
def inequality_ratios(x, psi, xi):
    """Ratios that Theorems 4.4--4.6 require to be at most ``1``.

    Returns a dict with keys ``lsi``, ``talagrand`` and ``hwi``:

    *   ``lsi``       ``D(mu||gamma) / (I(mu||gamma)/2)``,
    *   ``talagrand`` ``W_2(mu, gamma)^2 / (2 D(mu||gamma))``,
    *   ``hwi``       ``D / (W sqrt(I) - W^2/2)``.
    """
    D = relative_free_energy(x, psi)
    I = relative_fisher(x, psi, xi)
    gam = semicircle_density(x, 1.0)
    W = w2_density_vs_density(x, psi, gam)

    hwi_rhs = W * np.sqrt(I) - 0.5 * W**2
    return {
        "D": D,
        "I": I,
        "W2": W,
        "lsi": D / (0.5 * I) if I > 0 else np.nan,
        "talagrand": W**2 / (2.0 * D) if D > 0 else np.nan,
        "hwi": D / hwi_rhs if hwi_rhs > 0 else np.nan,
    }
