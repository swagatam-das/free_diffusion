"""Tests for the identities the paper proves.

These are not smoke tests: each one checks a statement from the paper against
an independent computation.  Run with ``pytest -q``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freeddpm.cauchy import (  # noqa: E402
    density_from_g,
    g_atomic,
    g_semicircle,
    g_two_atom_cubic,
    score_from_g,
    subordination_g,
)
from freeddpm.design import (  # noqa: E402
    edge_diagnostic,
    flat_law_hilbert,
    hilbert_transform_grid,
    mp_density,
    mp_potential_derivative,
    mp_support,
)
from freeddpm.forward import TwoAtomLaw, critical_variance, transition_alpha  # noqa: E402
from freeddpm.functionals import (  # noqa: E402
    free_energy,
    log_energy_density,
    relative_free_energy,
    semicircle_density,
    semicircle_free_energy,
)
from freeddpm.learn import check_gradients  # noqa: E402
from freeddpm.matrix import dyson_path, gue  # noqa: E402
from freeddpm.reverse import stationarity_residual  # noqa: E402

GRID = np.linspace(-8.0, 8.0, 8001)


# ---------------------------------------------------------------------------
# Cauchy transforms
# ---------------------------------------------------------------------------
def test_semicircle_transform_solves_its_quadratic():
    """``v G^2 - z G + 1 = 0`` for the semicircular law of variance ``v``."""
    z = np.array([0.3 + 0.7j, -1.2 + 0.4j, 2.5 + 2.0j])
    for v in (0.5, 1.0, 2.3):
        g = g_semicircle(z, v)
        assert np.allclose(v * g**2 - z * g + 1.0, 0.0, atol=1e-12)
        assert np.all(np.imag(g) < 0)


def test_semicircle_density_and_moments():
    x = GRID
    psi = density_from_g(g_semicircle(x + 1e-9j, 1.0))
    assert np.trapezoid(psi, x) == pytest.approx(1.0, abs=2e-4)
    assert np.trapezoid(x**2 * psi, x) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(x**4 * psi, x) == pytest.approx(2.0, abs=1e-2)  # Catalan C_2


def test_semicircle_score_is_the_identity():
    """``xi_gamma(x) = x`` on the support: the free analogue of the Gaussian score."""
    x = np.linspace(-1.9, 1.9, 400)
    xi = score_from_g(g_semicircle(x + 1e-9j, 1.0))
    assert np.allclose(xi, x, atol=1e-6)


def test_two_atom_cubic_root_satisfies_the_cubic():
    a, v = 1.6, 1.0
    z = GRID + 1e-6j
    g = g_two_atom_cubic(z, a, v)
    res = v**2 * g**3 - 2 * v * z * g**2 + (z**2 - a**2 + v) * g - z
    assert np.max(np.abs(res)) < 1e-8


def test_two_atom_moments_match_free_convolution():
    """``m_2 = a^2 + v`` and ``m_4 = a^4 - 2a^4 + 2(a^2+v)^2`` (Remark 3.2)."""
    a, v = 1.6, 1.0
    x = GRID
    psi = density_from_g(g_two_atom_cubic(x + 1e-7j, a, v))
    m2 = np.trapezoid(x**2 * psi, x)
    m4 = np.trapezoid(x**4 * psi, x)
    assert m2 == pytest.approx(a**2 + v, abs=2e-3)
    assert m4 == pytest.approx(a**4 - 2 * a**4 + 2 * (a**2 + v) ** 2, abs=1e-2)


def test_subordination_reproduces_the_cubic():
    """The general subordination solver agrees with the closed-form cubic."""
    a, v = 1.6, 0.7
    alpha = 1.0 - v
    x = np.linspace(-5.0, 5.0, 401)
    z = x + 3e-3j
    g_exact = g_two_atom_cubic(z, a, v)
    g_sub = subordination_g(z, np.array([-a, a]), None, v=v, damping=0.4,
                            max_iter=20000, tol=1e-13)
    assert np.max(np.abs(g_exact - g_sub)) < 1e-6


def test_subordination_recovers_the_semicircle():
    """``delta_0 boxplus gamma_v = gamma_v``."""
    z = np.linspace(-4, 4, 201) + 1e-3j
    g = subordination_g(z, np.array([0.0]), None, v=1.3, damping=0.4, tol=1e-13)
    assert np.max(np.abs(g - g_semicircle(z, 1.3))) < 1e-6


# ---------------------------------------------------------------------------
# the support transition (Theorem 9.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a", [0.8, 1.2, 1.6, 2.0])
def test_gap_open_below_and_closed_above_critical_variance(a):
    vstar = critical_variance(a)
    below = density_from_g(g_two_atom_cubic(np.array([1e-8j]), a, 0.85 * vstar))[0]
    above = density_from_g(g_two_atom_cubic(np.array([1e-8j]), a, 1.15 * vstar))[0]
    assert below < 1e-5
    assert above > 1e-2


def test_transition_alpha_matches_the_flow():
    a0 = 1.6
    law = TwoAtomLaw(a0)
    astar = transition_alpha(a0)
    assert law.is_bimodal(astar * 1.2)
    assert not law.is_bimodal(astar * 0.8)
    # and the density at the origin agrees with that classification
    assert law.density(np.array([0.0]), astar * 1.2, eps=1e-8)[0] < 1e-4
    assert law.density(np.array([0.0]), astar * 0.8, eps=1e-8)[0] > 1e-2


# ---------------------------------------------------------------------------
# free entropy and the free energy
# ---------------------------------------------------------------------------
def test_log_energy_of_the_semicircle():
    """``chi(gamma_1) = -1/4``."""
    x = np.linspace(-3.0, 3.0, 6001)
    psi = semicircle_density(x, 1.0)
    assert log_energy_density(x, psi) == pytest.approx(-0.25, abs=2e-3)


def test_free_energy_minimised_by_the_semicircle():
    """``F(gamma) = 3/4`` and ``D(mu||gamma) >= 0`` with equality at gamma."""
    x = np.linspace(-6.0, 6.0, 6001)
    assert semicircle_free_energy(1.0) == pytest.approx(0.75)
    assert free_energy(x, semicircle_density(x, 1.0)) == pytest.approx(0.75, abs=2e-3)
    assert abs(relative_free_energy(x, semicircle_density(x, 1.0))) < 2e-3

    law = TwoAtomLaw(1.6)
    for alpha in (0.2, 0.5, 0.8):
        g = law.g(x, alpha, eps=1e-7)
        psi = np.clip(-np.imag(g) / np.pi, 0.0, None)
        psi /= np.trapezoid(psi, x)
        assert relative_free_energy(x, psi) > 0


# ---------------------------------------------------------------------------
# designed equilibria (Theorem 8.11)
# ---------------------------------------------------------------------------
def test_marchenko_pastur_is_stationary_for_the_designed_drift():
    """``V' = 2 H mu_*`` on the interior of the support, with ``f = 1``."""
    L = 0.5
    lo, hi = mp_support(L)
    x = np.linspace(lo, hi, 6000)
    psi = mp_density(x, L)
    h = x[1] - x[0]
    Hmu = hilbert_transform_grid(x, psi * h)
    Vp = mp_potential_derivative(x, L)
    m = (x > lo + 0.1 * (hi - lo)) & (x < hi - 0.1 * (hi - lo))
    rel = np.max(np.abs(Vp[m] - 2 * Hmu[m])) / np.max(np.abs(Vp[m]))
    assert rel < 0.02


def test_marchenko_pastur_has_soft_edges():
    """Square-root vanishing, which is the hypothesis under which Theorem 8.11
    gives a bounded drift up to the edge."""
    L = 0.5
    lo, hi = mp_support(L)
    x = np.linspace(lo, hi, 12000)
    psi = mp_density(x, L)
    assert edge_diagnostic(x, psi, lo, window=0.02) == pytest.approx(0.5, abs=0.1)
    assert edge_diagnostic(x, psi, hi, window=0.25) == pytest.approx(0.5, abs=0.1)


def test_flat_density_has_an_unbounded_design_drift():
    """The counterexample in the proof of Theorem 8.11.

    A flat density on ``[-1,1]`` is Hoelder continuous on its support, yet
    ``H mu_*`` diverges logarithmically at both edges, so the design drift is
    not bounded there.  This is why the theorem is stated on compact subsets of
    the interior unless the density vanishes at the edges.
    """
    x = np.array([0.0, 0.5, 0.9, 0.99, 0.999, 0.9999])
    H = flat_law_hilbert(x)
    assert H[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(H) > 0)
    assert H[-1] > 4.0  # growing without bound
    # and the numerical principal value on a grid agrees away from the edges
    xs = np.linspace(-1.0, 1.0, 12001)
    psi = np.full_like(xs, 0.5)
    h = xs[1] - xs[0]
    Hnum = hilbert_transform_grid(xs, psi * h)
    m = np.abs(xs) < 0.9
    assert np.max(np.abs(Hnum[m] - flat_law_hilbert(xs)[m])) < 5e-3


# ---------------------------------------------------------------------------
# the matrix model
# ---------------------------------------------------------------------------
def test_gue_spectrum_is_semicircular():
    rng = np.random.default_rng(3)
    ev = np.linalg.eigvalsh(gue(1500, rng))
    assert np.mean(ev**2) == pytest.approx(1.0, abs=0.05)
    assert ev.max() == pytest.approx(2.0, abs=0.1)
    assert ev.min() == pytest.approx(-2.0, abs=0.1)


def test_dyson_eigenvalues_do_not_collide():
    """The noncollision property underlying the Dyson system (3.11)."""
    rng = np.random.default_rng(5)
    N = 25
    lam0 = np.linspace(-1.5, 1.5, N)
    paths = dyson_path(lam0, np.linspace(0.05, 1.0, 20), beta=1.0, rng=rng, dt=5e-4)
    gaps = np.diff(paths, axis=1)
    assert np.all(gaps > 0)


# ---------------------------------------------------------------------------
# the sign convention (Remark 7.3)
# ---------------------------------------------------------------------------
def test_reverse_drift_reduces_to_the_forward_drift_at_equilibrium():
    y = np.linspace(-3, 3, 101)
    assert np.max(np.abs(stationarity_residual(y, beta=1.0))) == 0.0
    # the opposite sign convention would give 2 beta y, which is not zero
    wrong = 0.5 * y + y - (-0.5 * y)
    assert np.max(np.abs(wrong)) > 1.0


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------
def test_analytic_gradients_match_finite_differences():
    assert check_gradients() < 1e-6


def test_denoiser_learns_a_known_function():
    """A sanity check on the optimiser: fit ``h(l,a) = a * tanh(l)``."""
    from freeddpm.learn import train_denoiser

    rng = np.random.default_rng(7)
    X = np.stack([rng.uniform(-3, 3, 4000), rng.uniform(0.1, 0.9, 4000)], axis=1)
    y = X[:, 1] * np.tanh(X[:, 0])
    net, hist = train_denoiser(X, y, widths=(2, 32, 32, 1), epochs=120,
                               batch_size=256, lr=5e-3, rng=rng)
    pred = net.forward(X)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.02
    assert hist[-1] < hist[0] / 10
