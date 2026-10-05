"""Prescribing the equilibrium: Theorem 8.11 and the Marchenko--Pastur example.

Theorem 8.11 states that for a compactly supported ``mu_*`` with Hoelder
continuous density and a positive continuous ``f``, the drift

.. math::
    b_*(x) = -f(x)^2\\, H(f^2 \\mu_*)(x),
    \\qquad H(f^2\\mu_*)(x) = \\mathrm{p.v.}\\!\\int \\frac{f(y)^2 d\\mu_*(y)}{x-y},

makes ``mu_*`` stationary for ``dX = b_*(X) dt + f(X) dS f(X)``.  The principal
value converges at interior points of the support; boundedness up to a hard
edge requires the density to vanish there at a Hoelder rate, and this module
provides :func:`edge_diagnostic` to check that hypothesis numerically, together
with the counterexample of the proof (a flat density on ``[-1,1]``, for which
``H mu_*`` diverges logarithmically at both edges).

For ``f = 1`` the drift reduces to ``b_* = -H mu_*``, and the corresponding
diffusion is ``dX = -V'(X)/2 dt + dS`` with ``V' = 2 H mu_*``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hilbert_transform_grid",
    "design_drift",
    "mp_density",
    "mp_potential_derivative",
    "mp_support",
    "flat_law_hilbert",
    "edge_diagnostic",
]


def hilbert_transform_grid(x, w, x_eval=None):
    """Principal-value Hilbert transform ``H nu(x) = p.v. int d nu(y)/(x-y)``.

    ``w`` are the masses carried by the grid points ``x`` (for a density
    ``psi`` on a uniform grid, ``w = psi * h``).  The diagonal term is omitted,
    which is the symmetric-difference regularisation of the principal value and
    is second-order accurate on a uniform grid.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    xe = x if x_eval is None else np.asarray(x_eval, dtype=float)
    out = np.empty(xe.size)
    # chunked so that the n x n kernel is never materialised: the grids used in
    # the experiments have tens of thousands of points.
    block = max(1, int(4e6 // max(x.size, 1)))
    for start in range(0, xe.size, block):
        stop = min(start + block, xe.size)
        d = xe[start:stop, None] - x[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            K = 1.0 / d
        K[~np.isfinite(K)] = 0.0
        out[start:stop] = K @ w
    return out


def design_drift(x, psi, f=None):
    """The drift ``b_* = -f^2 H(f^2 mu_*)`` of Theorem 8.11 on a uniform grid."""
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=float)
    h = x[1] - x[0]
    fv = np.ones_like(x) if f is None else np.asarray(f, dtype=float)
    w = fv**2 * psi * h
    return -(fv**2) * hilbert_transform_grid(x, w)


# ----------------------------------------------------------------------------
# Marchenko--Pastur (Example 8.15)
# ----------------------------------------------------------------------------
def mp_support(L: float):
    """``[(1-sqrt L)^2, (1+sqrt L)^2]``."""
    s = np.sqrt(L)
    return (1.0 - s) ** 2, (1.0 + s) ** 2


def mp_density(x, L: float):
    """Marchenko--Pastur density with ratio ``L`` and unit scale."""
    x = np.asarray(x, dtype=float)
    lo, hi = mp_support(L)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2.0 * np.pi * L * x[m])
    return out


def mp_potential_derivative(x, L: float):
    """``V'(x) = 1/L - (1-L)/(L x)`` for ``V(x) = (x - (1-L) log x)/L``.

    ``V`` is strictly convex on ``(0, oo)`` with ``V'' = (1-L)/(L x^2) > 0`` for
    ``L < 1``, which is what places the Marchenko--Pastur diffusion inside
    Corollary 8.16.  The potential is defined only on the positive half line;
    see Example 8.15 for the caveat this entails at the level of the SDE.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / L - (1.0 - L) / (L * x)


# ----------------------------------------------------------------------------
# the edge counterexample from the proof of Theorem 8.11
# ----------------------------------------------------------------------------
def flat_law_hilbert(x):
    """``H mu_*(x) = log|(x+1)/(x-1)| / 2`` for the flat density on ``[-1,1]``.

    This is the counterexample in the proof of Theorem 8.11: the density is
    Hoelder continuous on its support (it is constant) yet ``H mu_*`` diverges
    logarithmically at both edges, so the design drift is unbounded there and
    Hoelder continuity up to the edge fails.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * np.log(np.abs((x + 1.0) / (x - 1.0)))


def edge_diagnostic(x, psi, edge, kappa_grid=None, window=0.1):
    """Estimate the exponent ``kappa`` in ``psi(x) ~ dist(x, edge)^kappa``.

    Theorem 8.11 gives a bounded, Hoelder continuous drift on the whole support
    when ``kappa > 0``; a soft (square-root) edge has ``kappa = 1/2`` and a hard
    edge ``kappa = 0``.  The estimate is a least-squares fit of ``log psi``
    against ``log dist`` over the points within ``window`` of the edge.
    """
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=float)
    d = np.abs(x - edge)
    m = (d > 0) & (d < window) & (psi > 0)
    if m.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(d[m]), np.log(psi[m]), 1)[0]
    return float(slope)
