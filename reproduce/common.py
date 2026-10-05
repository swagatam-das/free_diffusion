"""Shared numerical utilities for the experiments of
'Beyond the Semicircle: Free Diffusion Models with Prescribed Equilibria'."""
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.special import ndtr

trapz = getattr(np, "trapezoid", None) or np.trapz


def gue(N, rng):
    """GUE normalised so that its spectrum is the standard semicircle on [-2, 2]
    (E|H_ij|^2 = 1/N)."""
    Z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    return (Z + Z.conj().T) / np.sqrt(2 * N)


def w1_to_cdf(samples, grid, cdf):
    """W1 between the empirical measure of `samples` and a law with CDF `cdf`
    evaluated on `grid`:  W1 = int |F_emp - F| dx."""
    s = np.sort(np.asarray(samples))
    Femp = np.searchsorted(s, grid, side="right") / len(s)
    return trapz(np.abs(Femp - cdf), grid)


def free_conv_semicircle_density(x, atoms, weights, v, eps=1e-7):
    """Density of  sum_i w_i delta_{a_i}  [free-convolved with]  semicircle(variance v).

    Solves the Pastur equation  G = sum_i w_i / (z - v G - a_i),  z = x + i*eps,
    as a polynomial in G and keeps the root in the lower half-plane
    (the unique Stieltjes solution).  Returns -Im G / pi at each point of x."""
    atoms = np.asarray(atoms, float)
    weights = np.asarray(weights, float)
    out = np.zeros(len(x))
    m = len(atoms)
    for k, xx in enumerate(x):
        z = xx + 1j * eps
        facs = [np.array([z - a, -v]) for a in atoms]          # (z - a_i) - v G
        Ptot = np.array([1.0 + 0j])
        for f in facs:
            Ptot = P.polymul(Ptot, f)
        Q = np.zeros(1, dtype=complex)
        for i in range(m):
            term = np.array([weights[i] + 0j])
            for j in range(m):
                if j != i:
                    term = P.polymul(term, facs[j])
            Q = P.polyadd(Q, term)
        poly = P.polysub(P.polymulx(Ptot), Q)                   # G*P(G) - Q(G) = 0
        r = P.polyroots(poly)
        G = r[np.argmin(r.imag)]
        out[k] = max(-G.imag / np.pi, 0.0)
    return out


def cdf_from_density(grid, dens):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return c


def classical_conv_cdf(grid, atoms, weights, v):
    """CDF of  sum_i w_i delta_{a_i} * N(0, v)  (classical convolution)."""
    return sum(w * ndtr((grid - a) / np.sqrt(v)) for a, w in zip(atoms, weights))
