"""Cauchy transforms, subordination, and free additive convolution.

Conventions used throughout, matching the paper:

*   The Cauchy (Stieltjes) transform of a probability measure ``mu`` is

    .. math:: G_\\mu(z) = \\int \\frac{d\\mu(y)}{z-y}, \\qquad z \\in \\mathbb{C}^+,

    so that ``G`` maps the upper half plane into the *lower* half plane and
    ``G(z) ~ 1/z`` as ``|z| -> oo``.

*   The Hilbert transform and the free score (conjugate variable) are recovered
    from the boundary values by equation (9.6) of the paper,

    .. math::
        \\psi(x) = -\\tfrac{1}{\\pi}\\operatorname{Im} G(x+i0),
        \\qquad
        \\xi_\\mu(x) = 2\\operatorname{Re} G(x+i0) = 2 H\\mu(x).

    Note the sign convention for ``xi``: it is *minus* the classical score, see
    Remark 2.3 of the paper.  With this convention the semicircular law of unit
    variance has ``xi_gamma(x) = x``.

*   ``gamma_v`` denotes the semicircular law of variance ``v``, supported on
    ``[-2 sqrt(v), 2 sqrt(v)]``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "g_semicircle",
    "g_atomic",
    "g_two_atom_cubic",
    "subordination_g",
    "free_convolve_atomic",
    "density_from_g",
    "score_from_g",
]

_EPS_DEFAULT = 1e-6


# ----------------------------------------------------------------------------
# closed-form Cauchy transforms
# ----------------------------------------------------------------------------
def g_semicircle(z, v=1.0):
    """Cauchy transform of the semicircular law of variance ``v``.

    ``G(z) = (z - sqrt(z^2 - 4v)) / (2v)`` with the branch fixed by
    ``Im G < 0`` on the upper half plane.
    """
    z = np.asarray(z, dtype=complex)
    if v <= 0:
        return 1.0 / z
    root = np.sqrt(z * z - 4.0 * v)
    g_minus = (z - root) / (2.0 * v)
    g_plus = (z + root) / (2.0 * v)
    # select the branch with Im G < 0 on the upper half plane, equivalently the
    # one asymptotic to 1/z; on the real axis fall back to the same choice by
    # continuity from above.
    take_minus = np.imag(g_minus) <= np.imag(g_plus)
    return np.where(take_minus, g_minus, g_plus)


def g_atomic(z, atoms, weights=None):
    """Cauchy transform of a finitely supported measure.

    Parameters
    ----------
    z : array_like of complex
    atoms : array_like of float
        Support points.
    weights : array_like of float, optional
        Masses; defaults to the uniform law on ``atoms``.
    """
    z = np.asarray(z, dtype=complex)
    atoms = np.asarray(atoms, dtype=float)
    if weights is None:
        weights = np.full(atoms.shape, 1.0 / atoms.size)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.sum(weights / (z[..., None] - atoms), axis=-1)


def g_two_atom_cubic(z, a, v):
    """Cauchy transform of ``(delta_{-a} + delta_{a})/2 boxplus gamma_v``.

    Solves the cubic (9.5) of the paper,

    .. math:: v^2 g^3 - 2 v z g^2 + (z^2 - a^2 + v) g - z = 0,

    and selects the root with ``Im g < 0`` on the upper half plane which is
    asymptotic to ``1/z``.  This is the exactly solvable benchmark used in
    Sections 9 and 10.

    The root selection is done by continuity in ``z``: among the roots with the
    correct half-plane, the one closest to the semicircular/atomic
    interpolation is chosen.  For ``Im z > 0`` the admissible root is unique,
    so the rule is unambiguous away from the real axis.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    if v <= 0:
        out = g_atomic(z, [-a, a])
        return out

    shape = z.shape
    zf = z.ravel()
    # numpy.roots is per-polynomial; vectorise via companion eigenvalues.
    c3 = v * v
    c2 = -2.0 * v * zf
    c1 = zf * zf - a * a + v
    c0 = -zf

    roots = _cubic_roots(c3, c2, c1, c0)

    # admissible: Im g < 0 strictly (boundary values handled by eps > 0)
    imag = np.imag(roots)
    bad = imag >= 0
    # tie-break with the large-|z| asymptote 1/z, which is the correct root for
    # every z in the open upper half plane
    ref = (1.0 / zf)[:, None]
    dist = np.abs(roots - ref)
    dist = np.where(bad, np.inf, dist)
    idx = np.argmin(dist, axis=1)
    out = roots[np.arange(roots.shape[0]), idx]

    # fall back to the nearest root overall if no admissible one was found
    fallback = ~np.isfinite(dist.min(axis=1))
    if np.any(fallback):
        idx2 = np.argmin(np.abs(roots - ref), axis=1)
        out[fallback] = roots[np.arange(roots.shape[0]), idx2][fallback]
    return out.reshape(shape)


def _cubic_roots(c3, c2, c1, c0):
    """Roots of ``c3 x^3 + c2 x^2 + c1 x + c0`` for arrays of coefficients."""
    c3 = np.asarray(c3, dtype=complex)
    c2 = np.asarray(c2, dtype=complex)
    c1 = np.asarray(c1, dtype=complex)
    c0 = np.asarray(c0, dtype=complex)
    c3, c2, c1, c0 = np.broadcast_arrays(c3, c2, c1, c0)
    n = c0.size
    comp = np.zeros((n, 3, 3), dtype=complex)
    b = (c2 / c3).ravel()
    c = (c1 / c3).ravel()
    d = (c0 / c3).ravel()
    comp[:, 0, 0] = -b
    comp[:, 0, 1] = -c
    comp[:, 0, 2] = -d
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    return np.linalg.eigvals(comp)


# ----------------------------------------------------------------------------
# subordination for a general (discretised) initial law
# ----------------------------------------------------------------------------
def subordination_g(z, atoms, weights=None, v=1.0, tol=1e-12, max_iter=20000,
                    damping=0.5, omega0=None, return_omega=False):
    """Cauchy transform of ``rho boxplus gamma_v`` by subordination.

    Solves the fixed point (10.1) of the paper,

    .. math:: \\omega(z) = z - v\\, G_\\rho(\\omega(z)),

    by damped iteration, and returns ``G(z) = G_rho(omega(z))``.  Because
    ``omega`` maps the upper half plane into itself, the iteration is started
    from ``z`` and the imaginary part is kept bounded below by ``Im z``, which
    is what makes the damped iteration stable near the real axis.

    Parameters
    ----------
    z : array_like of complex
        Evaluation points, ordinarily ``x + i*eps`` with ``eps > 0``.
    atoms, weights : array_like
        A finitely supported approximation of ``rho``.
    v : float
        Variance of the semicircular summand.
    damping : float
        Relaxation parameter in ``(0, 1]``; ``omega <- (1-d) omega + d * F(omega)``.
    omega0 : array_like of complex, optional
        Starting point for the iteration.  When the transform is needed at a
        sequence of nearby noise levels, passing the previous level's ``omega``
        (obtained with ``return_omega=True``) cuts the iteration count by an
        order of magnitude, which is what makes the tabulation in the spiked
        experiment feasible.
    return_omega : bool
        Also return the converged subordination function.
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    if v <= 0:
        g = g_atomic(z, atoms, weights)
        return (g, z.copy()) if return_omega else g

    atoms = np.asarray(atoms, dtype=float)
    if weights is None:
        weights = np.full(atoms.shape, 1.0 / atoms.size)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    im_floor = np.maximum(np.imag(z), 1e-14)
    omega = z.copy() if omega0 is None else np.asarray(omega0, dtype=complex).copy()

    # Active-set iteration.  The fixed point converges at very different speeds
    # across the grid: points well away from the support settle in a few dozen
    # steps, while points near an outlier or close to the real axis can need
    # thousands.  Iterating the entire grid until the slowest point converges
    # therefore wastes almost all of the work, so points are retired from the
    # active set as they converge.  The result is identical to the naive loop
    # up to the tolerance; only the cost differs.
    shape = omega.shape
    out = omega.ravel().copy()
    idx = np.arange(out.size)
    z_a = z.ravel().copy()
    om_a = out.copy()
    fl_a = np.broadcast_to(im_floor, shape).ravel().copy()

    for _ in range(max_iter):
        g_a = g_atomic(om_a, atoms, weights)
        target = z_a - v * g_a
        # keep the iterate inside the upper half plane
        target = np.real(target) + 1j * np.maximum(np.imag(target), fl_a)
        new = (1.0 - damping) * om_a + damping * target
        delta = np.abs(new - om_a)
        om_a = new
        done = delta < tol
        if done.any():
            out[idx[done]] = om_a[done]
            keep = ~done
            idx, z_a, om_a, fl_a = idx[keep], z_a[keep], om_a[keep], fl_a[keep]
            if idx.size == 0:
                break

    if idx.size:  # max_iter reached for the stragglers
        out[idx] = om_a

    omega = out.reshape(shape)
    g = g_atomic(omega, atoms, weights)
    return (g, omega) if return_omega else g


def free_convolve_atomic(x, atoms, weights=None, v=1.0, eps=_EPS_DEFAULT, **kw):
    """Density and free score of ``rho boxplus gamma_v`` on a grid ``x``.

    Returns ``(psi, xi)`` where ``psi`` is the density and ``xi = 2 Re G`` the
    free score, both evaluated at ``x + i*eps``.
    """
    z = np.asarray(x, dtype=float) + 1j * eps
    g = subordination_g(z, atoms, weights, v=v, **kw)
    return density_from_g(g), score_from_g(g)


# ----------------------------------------------------------------------------
# boundary values
# ----------------------------------------------------------------------------
def density_from_g(g):
    """``psi(x) = -Im G(x + i0) / pi``."""
    return -np.imag(g) / np.pi


def score_from_g(g):
    """``xi(x) = 2 Re G(x + i0)``; the free score, equal to ``2 H mu(x)``."""
    return 2.0 * np.real(g)
