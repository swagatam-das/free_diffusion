"""State-dependent (sandwich) volatility: the integrator and the structural identities it must satisfy."""

import numpy as np

from freeddpm.design import design_drift, mp_density, mp_potential_derivative
from freeddpm.matrix import designed_matrix_path, sandwich_path


def _start(N=40):
    return np.linspace(0.35, 1.9, N)


def test_sandwich_with_unit_volatility_is_the_designed_matrix_model():
    """f = 1 and b = -V'/2 must reproduce designed_matrix_path with the same random numbers."""
    L = 0.5
    dV = lambda z: mp_potential_derivative(z, L)
    a = designed_matrix_path(_start(), [0.05], dV, rng=np.random.default_rng(3), dt=5e-3, floor=0.02)
    b = sandwich_path(_start(), [0.05], lambda z: -0.5 * dV(z), lambda z: np.ones_like(z),
                      rng=np.random.default_rng(3), dt=5e-3, floor=0.02)
    assert np.max(np.abs(a - b)) < 1e-8


def test_constant_volatility_is_a_change_of_clock():
    """f = c, b = c^4 beta is exactly the f = 1 flow with drift beta run at clock c^4:
    the spectral velocity of (10) is c^4 (H mu_t - H mu_*) (Theorem 8.1 with f constant)."""
    c = np.sqrt(2.0)                          # c^4 = 4
    beta = lambda z: -0.3 * z
    h, k = 0.01, 6
    a = sandwich_path(_start(), [k * h], lambda z: c**4 * beta(z), lambda z: c * np.ones_like(z),
                      rng=np.random.default_rng(7), dt=h)
    b = designed_matrix_path(_start(), [k * h * c**4], lambda z: -2.0 * beta(z),
                             rng=np.random.default_rng(7), dt=h * c**4)
    assert np.max(np.abs(a - b)) < 1e-8


def test_design_drift_with_state_dependent_f_cancels_the_flow_velocity():
    """b_* + f^2 H(f^2 mu_*) = 0 on the support for a non-constant f (Theorem 8.11)."""
    from freeddpm.design import hilbert_transform_grid
    L = 0.5; x = np.linspace(0.02, 3.8, 1500); h = x[1] - x[0]; psi = mp_density(x, L)
    f = np.sqrt(0.5 + 1.5 * x / (1 + x))
    b = design_drift(x, psi, f=f)
    V = b + f**2 * hilbert_transform_grid(x, f**2 * psi * h)
    assert np.max(np.abs(V)) < 1e-10


def test_state_dependent_design_keeps_marchenko_pastur_stationary():
    """Started at (a quantile discretisation of) mu_*, the designed non-constant-f flow stays near mu_*."""
    from freeddpm.functionals import w1_density_vs_samples
    L = 0.5; N = 80
    x = np.linspace(0.02, 3.8, 1500); h = x[1] - x[0]; psi = mp_density(x, L)
    f2 = lambda z: 0.5 + 1.5 * z / (1 + z)
    bgrid = design_drift(x, psi, f=np.sqrt(f2(x)))
    cdf = np.cumsum(psi) * h; cdf /= cdf[-1]
    start = np.interp((np.arange(N) + 0.5) / N, cdf, x)
    out = sandwich_path(start, [0.6], lambda z: np.interp(z, x, bgrid), lambda z: np.sqrt(f2(z)),
                        rng=np.random.default_rng(1), dt=4e-3, floor=0.02, ceil=3.8)
    xg = np.linspace(0.02, 3.5, 1500)
    assert w1_density_vs_samples(xg, mp_density(xg, L), out[-1]) < 0.06
