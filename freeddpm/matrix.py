"""Finite-``N`` Hermitian matrix models.

Normalisation.  Throughout, ``gue(N)`` returns a Hermitian matrix whose
empirical spectral distribution converges to the semicircular law of unit
variance, supported on ``[-2, 2]``.  Concretely, the off-diagonal entries have
variance ``1/N`` and the diagonal entries variance ``1/N``.  With this
convention the matrix model matching the free flow is equation (3.10) of the
paper,

.. math::
    dX^N_t = -\\tfrac12 \\beta(t) X^N_t\\, dt + \\sqrt{\\beta(t)/N}\\; dH_t,

with ``H`` a Hermitian Brownian motion, and the induced eigenvalue system is
the Dyson system (3.11).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gue",
    "hermitian_brownian_increment",
    "ou_matrix_path",
    "dyson_path",
    "designed_dyson_path",
    "designed_matrix_path",
    "symmetrise",
    "spectrum",
]


def symmetrise(A):
    """Hermitian part ``(A + A*)/2``."""
    return 0.5 * (A + A.conj().T)


def gue(N: int, rng: np.random.Generator):
    """A GUE matrix normalised to the semicircular law of unit variance."""
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return (A + A.conj().T) / (2.0 * np.sqrt(N))


def hermitian_brownian_increment(N: int, dt: float, rng: np.random.Generator):
    """Increment ``dH`` of a Hermitian Brownian motion, unnormalised in ``N``.

    Diagonal entries are real with variance ``dt``; off-diagonal entries are
    standard complex with variance ``dt``.  The ``1/sqrt(N)`` of (3.10) is
    applied by the caller.
    """
    A = rng.normal(scale=np.sqrt(dt), size=(N, N)) + 1j * rng.normal(
        scale=np.sqrt(dt), size=(N, N)
    )
    H = (A + A.conj().T) / 2.0
    # the construction above gives diagonal variance dt/2; rescale the diagonal
    idx = np.arange(N)
    H[idx, idx] = rng.normal(scale=np.sqrt(dt), size=N)
    return H


def spectrum(X):
    """Sorted eigenvalues of a Hermitian matrix."""
    return np.linalg.eigvalsh(symmetrise(X))


def ou_matrix_path(X0, times, beta=1.0, rng=None, exact=True):
    """Sample the Hermitian OU diffusion (3.10) at the given ``times``.

    With ``exact=True`` the marginals are sampled exactly using the closed-form
    solution ``X_t = sqrt(alpha_t) X_0 + sqrt(1-alpha_t) S_N`` with ``S_N`` a
    GUE matrix; this is the law of the OU process at time ``t`` and is what the
    marginal experiments require.  With ``exact=False`` the SDE is integrated
    by Euler--Maruyama, which is needed when the *path* (not only the marginal)
    matters, as for the eigenvalue trajectories of Figure 10.

    Returns an array of shape ``(len(times), N, N)``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X0 = np.asarray(X0, dtype=complex)
    N = X0.shape[0]
    times = np.asarray(times, dtype=float)

    if exact:
        out = np.empty((times.size, N, N), dtype=complex)
        for k, t in enumerate(times):
            alpha = np.exp(-beta * t)
            out[k] = np.sqrt(alpha) * X0 + np.sqrt(1.0 - alpha) * gue(N, rng)
        return out

    out = np.empty((times.size, N, N), dtype=complex)
    X = X0.copy()
    t_prev = 0.0
    for k, t in enumerate(times):
        n_sub = max(1, int(np.ceil((t - t_prev) / 1e-3)))
        dt = (t - t_prev) / n_sub
        for _ in range(n_sub):
            dH = hermitian_brownian_increment(N, dt, rng)
            X = X - 0.5 * beta * X * dt + np.sqrt(beta / N) * dH
            X = symmetrise(X)
        out[k] = X
        t_prev = t
    return out


def dyson_path(lam0, times, beta=1.0, rng=None, dt=1e-4):
    """Integrate the Dyson system (3.11) for the OU-confined eigenvalues.

    .. math::
        d\\lambda_i = -\\tfrac12 \\beta \\lambda_i\\, dt
        + \\frac{\\beta}{N} \\sum_{j \\ne i} \\frac{dt}{\\lambda_i - \\lambda_j}
        + \\sqrt{\\beta/N}\\, dB_i .

    Returns an array of shape ``(len(times), N)`` of sorted eigenvalues.  The
    ordering is preserved by the dynamics (the paths do not collide), which the
    test suite checks.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lam = np.sort(np.asarray(lam0, dtype=float)).copy()
    N = lam.size
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, N))

    t = 0.0
    for k, t_target in enumerate(times):
        while t < t_target - 1e-15:
            drift = -0.5 * beta * lam + (beta / N) * _repulsion(lam)
            h = _safe_step(lam, drift, dt, t_target - t)
            lam = lam + drift * h + np.sqrt(beta * h / N) * rng.normal(size=N)
            lam = np.sort(lam)
            t += h
        out[k] = lam
    return out


def designed_dyson_path(lam0, times, dV, rng=None, dt=1e-5, floor=None):
    """Eigenvalue system for the designed diffusion ``dX = -V'(X)/2 dt + N^{-1/2} dH``.

    .. math::
        d\\lambda_i = -\\tfrac12 V'(\\lambda_i)\\, dt
        + \\frac{1}{N} \\sum_{j \\ne i} \\frac{dt}{\\lambda_i - \\lambda_j}
        + \\frac{1}{\\sqrt N}\\, dB_i .

    This is the finite-``N`` model whose hydrodynamic limit is the continuity
    equation of Theorem 8.2 with ``f = 1``, and whose equilibrium is the law
    designed by Theorem 8.11.  Used for the Marchenko--Pastur experiment of
    Section 10.6, where ``V`` is defined only on the positive half line; the
    optional ``floor`` keeps the iterates there, which is the numerical
    counterpart of the caveat recorded in Example 8.15.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lam = np.sort(np.asarray(lam0, dtype=float)).copy()
    N = lam.size
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, N))

    t = 0.0
    for k, t_target in enumerate(times):
        while t < t_target - 1e-15:
            drift = -0.5 * dV(lam) + (1.0 / N) * _repulsion(lam)
            h = _safe_step(lam, drift, dt, t_target - t)
            lam = lam + drift * h + np.sqrt(h / N) * rng.normal(size=N)
            if floor is not None:
                lam = np.maximum(lam, floor)
            lam = np.sort(lam)
            t += h
        out[k] = lam
    return out


def _safe_step(lam, drift, dt, remaining, safety=0.15):
    """Adaptive step: no eigenvalue may move more than a fraction of the local gap.

    The Dyson system is stiff wherever two eigenvalues are close, since the
    repulsion term blows up like the reciprocal gap.  An explicit step longer
    than the gap divided by the drift can push eigenvalues through one another,
    which destroys the ordering and, for confining potentials singular at an
    endpoint, can send an eigenvalue off to infinity.  Capping the displacement
    at ``safety`` times the smallest gap keeps the explicit scheme stable at a
    cost of a few extra substeps near collisions.
    """
    gaps = np.diff(lam)
    gmin = gaps.min() if gaps.size else np.inf
    dmax = np.max(np.abs(drift))
    h = dt
    if dmax > 0 and np.isfinite(gmin):
        h = min(h, safety * gmin / dmax)
    return float(max(min(h, remaining), 1e-12))


def designed_matrix_path(X0, times, dV, rng=None, dt=1e-3, floor=None):
    """Integrate ``dX = -V'(X)/2 dt + N^{-1/2} dH`` at the matrix level.

    This is the model of Proposition 8.3 with ``f = 1``, and it is the
    integrator to prefer over :func:`designed_dyson_path` for confining
    potentials that are singular at an endpoint.  The eigenvalue system is
    stiff for such potentials -- the repulsion grows like the reciprocal gap
    while ``V'`` blows up at the endpoint -- so an explicit eigenvalue scheme is
    forced onto steps of order ``1e-8``.  At the matrix level the repulsion is
    never formed explicitly: it is generated by the noise, and a step of
    ``1e-3`` is stable.  The cost is one Hermitian eigendecomposition per step,
    which is ``O(N^3)`` but done a few thousand times rather than a few billion.

    ``V'`` is applied spectrally.  ``floor`` clips the eigenvalues away from the
    singularity of ``V'`` before the drift is evaluated; for Marchenko--Pastur
    any value well below the lower edge ``(1-sqrt L)^2`` and away from the
    origin (0.02 say) works, and this is the numerical counterpart of the
    caveat recorded in Example 8.15.

    Returns an array of shape ``(len(times), N)`` of sorted eigenvalues.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X0, dtype=complex)
    if X.ndim == 1:
        X = np.diag(X).astype(complex)
    N = X.shape[0]
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, N))

    t = 0.0
    for k, t_target in enumerate(times):
        while t < t_target - 1e-12:
            h = min(dt, t_target - t)
            lam, U = np.linalg.eigh(symmetrise(X))
            if floor is not None:
                lam = np.maximum(lam, floor)
            drift = U @ np.diag(-0.5 * dV(lam)) @ U.conj().T
            X = symmetrise(
                X + drift * h + np.sqrt(1.0 / N) * hermitian_brownian_increment(N, h, rng)
            )
            t += h
        out[k] = np.linalg.eigvalsh(symmetrise(X))
    return out


def _repulsion(lam):
    """``sum_{j != i} 1/(lam_i - lam_j)`` computed pairwise."""
    diff = lam[:, None] - lam[None, :]
    np.fill_diagonal(diff, np.inf)
    return np.sum(1.0 / diff, axis=1)
