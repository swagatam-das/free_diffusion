"""The forward free Ornstein--Uhlenbeck flow and its spectral marginals.

The forward process of the paper is

.. math::
    dX_t = -\\tfrac12 \\beta(t) X_t\\, dt + \\sqrt{\\beta(t)}\\, dS_t,

whose spectral marginals are, by Proposition 3.5,

.. math::
    \\mu_t = (D_{\\sqrt{\\alpha_t}} \\mu_0) \\boxplus \\gamma_{v_t},
    \\qquad \\alpha_t = e^{-\\Lambda(t)},\\quad v_t = 1-\\alpha_t,

with ``Lambda(t) = int_0^t beta``.  All experiments use the constant schedule
``beta = 1``, so ``Lambda(t) = t`` and ``alpha_t = e^{-t}``; the module is
written in terms of ``alpha`` throughout, so a different schedule only changes
the map ``t -> alpha``.
"""

from __future__ import annotations

import numpy as np

from .cauchy import (
    density_from_g,
    g_two_atom_cubic,
    score_from_g,
    subordination_g,
)

__all__ = [
    "Schedule",
    "TwoAtomLaw",
    "EmpiricalLaw",
    "critical_variance",
    "transition_alpha",
]


class Schedule:
    """Constant-rate noise schedule ``beta(t) = beta``.

    ``alpha(t) = exp(-beta t)`` and ``v(t) = 1 - alpha(t)``.
    """

    def __init__(self, beta: float = 1.0):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)

    def Lambda(self, t):
        return self.beta * np.asarray(t, dtype=float)

    def alpha(self, t):
        return np.exp(-self.Lambda(t))

    def v(self, t):
        return 1.0 - self.alpha(t)

    def t_of_alpha(self, alpha):
        return -np.log(np.asarray(alpha, dtype=float)) / self.beta


class TwoAtomLaw:
    """``mu_0 = (delta_{-a0} + delta_{a0})/2`` with exact marginals.

    The marginal at level ``alpha`` is
    ``(delta_{-a} + delta_{a})/2 boxplus gamma_v`` with ``a = sqrt(alpha) a0``
    and ``v = 1 - alpha``, whose Cauchy transform solves the cubic (9.5).
    """

    def __init__(self, a0: float = 1.6):
        self.a0 = float(a0)

    # -- exact marginals -----------------------------------------------------
    def g(self, x, alpha, eps=1e-6):
        a = np.sqrt(alpha) * self.a0
        v = 1.0 - alpha
        z = np.asarray(x, dtype=float) + 1j * eps
        return g_two_atom_cubic(z, a, v)

    def density(self, x, alpha, eps=1e-6):
        return density_from_g(self.g(x, alpha, eps))

    def score(self, x, alpha, eps=1e-6):
        """Free score ``xi_{mu_t}(x) = 2 Re G``."""
        return score_from_g(self.g(x, alpha, eps))

    def sample_matrix(self, N, alpha, rng):
        """A Hermitian sample whose spectrum is the finite-``N`` marginal.

        ``X = sqrt(alpha) X_0 + sqrt(1-alpha) S_N`` with ``X_0`` diagonal with
        entries ``+-a0`` and ``S_N`` a GUE matrix normalised so that its
        spectral law converges to ``gamma_1``.
        """
        from .matrix import gue

        d = np.empty(N)
        d[: N // 2] = -self.a0
        d[N // 2 :] = self.a0
        X0 = np.diag(d).astype(complex)
        return np.sqrt(alpha) * X0 + np.sqrt(1.0 - alpha) * gue(N, rng)

    # -- support ------------------------------------------------------------
    def is_bimodal(self, alpha):
        """Theorem 9.3: bimodal iff ``v < a^2``, i.e. ``alpha > 1/(1+a0^2)``."""
        return alpha > transition_alpha(self.a0)


class EmpiricalLaw:
    """A general initial law given by atoms and weights, handled by subordination.

    Used for the spiked covariance model of Section 10.7, for which no closed
    form is available.
    """

    def __init__(self, atoms, weights=None):
        atoms = np.asarray(atoms, dtype=float)
        if weights is None:
            weights = np.full(atoms.shape, 1.0 / atoms.size)
        weights = np.asarray(weights, dtype=float)
        self.atoms = atoms
        self.weights = weights / weights.sum()

    def g(self, x, alpha, eps=1e-4, **kw):
        z = np.asarray(x, dtype=float) + 1j * eps
        scaled = np.sqrt(alpha) * self.atoms
        return subordination_g(z, scaled, self.weights, v=1.0 - alpha, **kw)

    def density(self, x, alpha, eps=1e-4, **kw):
        return density_from_g(self.g(x, alpha, eps, **kw))

    def score(self, x, alpha, eps=1e-4, **kw):
        return score_from_g(self.g(x, alpha, eps, **kw))

    def sample_matrix(self, N, alpha, rng):
        from .matrix import gue

        idx = rng.choice(self.atoms.size, size=N, p=self.weights)
        X0 = np.diag(self.atoms[idx]).astype(complex)
        return np.sqrt(alpha) * X0 + np.sqrt(1.0 - alpha) * gue(N, rng)


# ----------------------------------------------------------------------------
# the support transition (Theorem 9.3)
# ----------------------------------------------------------------------------
def critical_variance(a: float) -> float:
    """``v* = a^2``: the support of ``(delta_{-a}+delta_a)/2 boxplus gamma_v``
    is connected iff ``v >= v*``."""
    return float(a) ** 2


def transition_alpha(a0: float) -> float:
    """``alpha* = 1/(1+a0^2)``, the level at which the gap closes along the flow.

    Equivalently ``Lambda* = log(1 + a0^2)``.
    """
    return 1.0 / (1.0 + float(a0) ** 2)
