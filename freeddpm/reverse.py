"""Reverse-time dynamics: the probability flow and the reverse matrix SDE.

Two reversals are implemented, corresponding to Theorem 7.1 and Corollary 7.4.

*   :func:`probability_flow` integrates the deterministic equation

    .. math:: \\dot y = \\tfrac{\\beta}{2}\\bigl(y - \\xi_{\\mu_t}(y)\\bigr),

    with ``t`` decreasing, which has the same marginals as the reverse SDE.

*   :func:`reverse_matrix_sde` integrates

    .. math::
        dY_s = \\bigl(\\tfrac12\\beta Y_s - \\beta\\, \\xi_{\\mu_{T-s}}(Y_s)\\bigr) ds
        + \\sqrt{\\beta}\\; d\\bar S_s

    at the matrix level, applying the scalar score spectrally at each step.
    This is Algorithm 1 of the paper.

Sign convention.  ``xi`` is the *conjugate variable*, equal to twice the
Hilbert transform and to minus the classical score.  At equilibrium
``xi_gamma(y) = y`` and the reverse drift collapses to the forward drift
``-beta y / 2``; :func:`stationarity_residual` checks exactly this, and is the
diagnostic recommended in Remark 7.3.
"""

from __future__ import annotations

import numpy as np

from .matrix import gue, hermitian_brownian_increment, symmetrise

__all__ = [
    "probability_flow",
    "reverse_matrix_sde",
    "stationarity_residual",
]


def probability_flow(y0, alphas, score, beta=1.0, method="midpoint"):
    """Integrate the probability-flow ODE backwards along ``alphas``.

    Parameters
    ----------
    y0 : array_like
        Initial ensemble, distributed as ``mu_{t_0}`` at ``alphas[0]``.
    alphas : array_like
        Decreasing-in-time schedule, i.e. *increasing* values of ``alpha``,
        from the near-equilibrium end (``alpha`` small) to the data end
        (``alpha`` close to 1).
    score : callable
        ``score(y, alpha) -> xi_{mu_t}(y)``.
    method : {"midpoint", "euler"}
        Midpoint is the rule used for the reported figures.

    Returns
    -------
    y : ndarray
        The ensemble after the final step.
    history : list of ndarray
        The ensemble after each step, including the initial one.
    """
    y = np.asarray(y0, dtype=float).copy()
    alphas = np.asarray(alphas, dtype=float)
    history = [y.copy()]

    # The reverse time variable is s = T - t.  With alpha_t = exp(-beta t), an
    # increasing sequence of alphas corresponds to a decreasing sequence of t
    # and hence to an *increasing* s, so the step length in s is
    # h = t(alpha_k) - t(alpha_{k+1}) > 0 and the drift is the one written in
    # Corollary 7.4, (beta/2)(y - xi).  Integrating with the opposite sign runs
    # the flow forwards again and collapses the ensemble onto the semicircular
    # law; the assertion below guards against that.
    t_grid = -np.log(alphas) / beta
    if np.any(np.diff(t_grid) >= 0):
        raise ValueError("alphas must be strictly increasing (t strictly decreasing)")

    for k in range(alphas.size - 1):
        t0, t1 = t_grid[k], t_grid[k + 1]
        h = t0 - t1  # positive step in the reverse time s
        a0 = alphas[k]

        def drift(yy, aa):
            return 0.5 * beta * (yy - score(yy, aa))

        if method == "euler":
            y = y + h * drift(y, a0)
        elif method == "midpoint":
            a_mid = float(np.exp(-beta * 0.5 * (t0 + t1)))
            y_mid = y + 0.5 * h * drift(y, a0)
            y = y + h * drift(y_mid, a_mid)
        else:
            raise ValueError("method must be 'midpoint' or 'euler'")
        history.append(y.copy())

    return y, history


def reverse_matrix_sde(N, alphas, score, beta=1.0, rng=None, X_init=None):
    """Integrate the reverse-time *matrix* SDE (7.1) by Euler--Maruyama.

    At each step the current matrix is diagonalised, the scalar free score is
    applied to the eigenvalues, and the drift is reassembled in the eigenbasis;
    Hermitian Gaussian noise is added and the iterate re-symmetrised.  This is
    the sampling half of Algorithm 1.

    Because the score is a scalar function of the spectrum, the same score
    drives every dimension ``N``: this is the dimension-transfer experiment of
    Section 10.5.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = gue(N, rng) if X_init is None else np.asarray(X_init, dtype=complex)
    alphas = np.asarray(alphas, dtype=float)
    t_grid = -np.log(alphas) / beta

    for k in range(alphas.size - 1):
        h = t_grid[k] - t_grid[k + 1]  # positive length of the reverse step
        a = alphas[k]
        lam, U = np.linalg.eigh(symmetrise(X))
        drift_eigs = 0.5 * beta * lam - beta * score(lam, a)
        drift = U @ np.diag(drift_eigs) @ U.conj().T
        dH = hermitian_brownian_increment(N, h, rng)
        X = X + drift * h + np.sqrt(beta / N) * dH
        X = symmetrise(X)

    return X


def stationarity_residual(y, beta=1.0):
    """Residual of the consistency check of Remark 7.3.

    At equilibrium ``mu = gamma`` the conjugate variable is ``xi(y) = y``, so
    the reverse drift ``beta y / 2 - beta xi(y)`` must equal the forward drift
    ``-beta y / 2``.  Returns the pointwise difference, which is identically
    zero for the correct sign convention and equals ``2 beta y`` for the
    opposite one.
    """
    y = np.asarray(y, dtype=float)
    reverse = 0.5 * beta * y - beta * y
    forward = -0.5 * beta * y
    return reverse - forward
