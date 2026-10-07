"""The deterministic particle discretisation of the free flow and its probability-flow reversal."""

import numpy as np

from freeddpm.design import design_drift, mp_density
from freeddpm.functionals import w1_samples
from freeddpm.particles import forward_flow, reverse_probability_flow

ONE = lambda z: np.ones_like(z)


def _semicircle_quantiles(v, M):
    x = np.linspace(-2 * np.sqrt(v), 2 * np.sqrt(v), 40001)
    rho = np.sqrt(np.maximum(4 * v - x**2, 0)) / (2 * np.pi * v)
    cdf = np.cumsum(rho) * (x[1] - x[0]); cdf /= cdf[-1]
    return np.interp((np.arange(M) + 0.5) / M, cdf, x)


def test_particles_follow_the_exact_free_ou_marginal():
    """From a semicircle of variance v0 the free OU flow stays semicircular with variance a*v0 + 1 - a, a = e^{-t}."""
    v0, M, t = 0.25, 300, 1.0
    _, snaps, _ = forward_flow(_semicircle_quantiles(v0, M), lambda z: -0.5 * z, ONE, T=t, snap_dt=0.1)
    a = np.exp(-t)
    assert w1_samples(snaps[-1], _semicircle_quantiles(a * v0 + 1 - a, M)) < 0.01


def test_probability_flow_reverses_a_state_dependent_flow():
    """Reversing the noise-free flow with velocity -V_{T-s} recovers the initial particles (non-constant f)."""
    M = 300
    f2 = lambda z: 0.5 + 1.5 * z / (1 + z)
    x = np.linspace(0.02, 3.8, 1500); psi = mp_density(x, 0.5)
    bgrid = design_drift(x, psi, f=np.sqrt(f2(x)))
    b = lambda z: np.interp(z, x, bgrid)
    Q0 = np.sort(np.concatenate([0.7 + 0.25 * _semicircle_quantiles(1.0, M // 2) / 2, 2.1 + 0.25 * _semicircle_quantiles(1.0, M // 2) / 2]))
    _, snaps, _ = forward_flow(Q0, b, f2, T=0.6, snap_dt=5e-3, lo=0.02, hi=3.8)
    y0 = reverse_probability_flow(snaps[-1].copy(), snaps, b, f2, snap_dt=5e-3, lo=0.02, hi=3.8)
    assert w1_samples(y0, Q0) < 0.02
    assert w1_samples(snaps[-1], Q0) > 0.05            # the forward flow genuinely moved the law


def test_designed_state_dependent_flow_keeps_marchenko_pastur_stationary_in_the_particle_system():
    M = 300
    f2 = lambda z: 0.5 + 1.5 * z / (1 + z)
    x = np.linspace(0.02, 3.8, 1500); h = x[1] - x[0]; psi = mp_density(x, 0.5)
    bgrid = design_drift(x, psi, f=np.sqrt(f2(x)))
    cdf = np.cumsum(psi) * h; cdf /= cdf[-1]
    Q0 = np.interp((np.arange(M) + 0.5) / M, cdf, x)
    _, snaps, _ = forward_flow(Q0, lambda z: np.interp(z, x, bgrid), f2, T=0.3, snap_dt=0.02, lo=0.02, hi=3.8)
    assert w1_samples(snaps[-1], Q0) < 0.02
