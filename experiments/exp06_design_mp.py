"""Section 10.6 -- a designed equilibrium: Marchenko--Pastur.

Theorem 8.11 and Example 8.15 predict that the diffusion

    dX = -V'(X)/2 dt + dS,   V'(x) = 1/L - (1-L)/(L x),

relaxes to the Marchenko--Pastur law with ratio L.  We integrate the
corresponding eigenvalue system at N = 400, L = 1/2, from an initial spectrum
far from the equilibrium, and compare the terminal empirical spectral
distribution with the theoretical density and support.

Two further checks are run, both of them consequences of the correction to
Theorem 8.11 discussed in the paper:

*   the stationarity identity V' = 2 H mu_* is verified directly on a grid,
    which is the content of Theorem 8.11 with f = 1;
*   the edge exponent of the Marchenko--Pastur density is estimated, confirming
    the square-root (soft-edge) vanishing under which Theorem 8.11 gives a
    bounded, Hoelder continuous drift up to the edge.

The process lives on the positive half line and V' is unbounded at the origin,
so the integrator keeps the iterates positive; this is the numerical
counterpart of the caveat in Example 8.15, and it is a caveat about the SDE,
not about the continuity equation.
"""

from __future__ import annotations

import numpy as np

from _common import cfg, parse, report, rng
from freeddpm.design import (
    edge_diagnostic,
    hilbert_transform_grid,
    mp_density,
    mp_potential_derivative,
    mp_support,
)
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.matrix import designed_matrix_path
from freeddpm.plotting import C_DATA, C_FREE, save, use_style

import matplotlib.pyplot as plt


def main():
    args = parse(__doc__)
    r = rng(args)
    L = 0.5
    lo, hi = mp_support(L)

    # --- stationarity identity V' = 2 H mu_* -------------------------------
    n_grid = cfg(args, 4000, 1200)
    x = np.linspace(lo, hi, n_grid)
    psi = mp_density(x, L)
    h = x[1] - x[0]
    Hmu = hilbert_transform_grid(x, psi * h)
    Vp = mp_potential_derivative(x, L)
    # compare on the interior, away from the edges where the grid quadrature of
    # a square-root singularity is least accurate
    m = (x > lo + 0.08 * (hi - lo)) & (x < hi - 0.08 * (hi - lo))
    stat_err = float(np.max(np.abs(Vp[m] - 2.0 * Hmu[m])) / np.max(np.abs(Vp[m])))

    edge_lo = edge_diagnostic(x, psi, lo, window=0.02)
    edge_hi = edge_diagnostic(x, psi, hi, window=0.25)

    # --- relaxation of the matrix model ------------------------------------
    N = cfg(args, 400, 120)
    T = cfg(args, 8.0, 3.0)
    n_out = cfg(args, 5, 4)
    times = np.linspace(T / n_out, T, n_out)
    lam0 = np.linspace(0.35, 1.9, N)  # far from mu_*: a flat band
    paths = designed_matrix_path(
        lam0, times,
        dV=lambda z: mp_potential_derivative(z, L),
        rng=r, dt=cfg(args, 1e-3, 4e-3), floor=0.02,
    )
    final = paths[-1]

    xg = np.linspace(max(1e-3, lo - 0.4), hi + 0.4, 2000)
    psi_g = mp_density(xg, L)
    w1 = w1_density_vs_samples(xg, psi_g, final)

    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.4))
    ax1.hist(final, bins=45, density=True, color=C_DATA, alpha=0.6,
             label=f"eigenvalues, $N={N}$")
    ax1.plot(xg, psi_g, color=C_FREE, label="Marchenko--Pastur")
    ax1.axvline(lo, color="k", ls=":", lw=1.0)
    ax1.axvline(hi, color="k", ls=":", lw=1.0)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("density")
    ax1.set_title(rf"terminal spectrum, $W_1={w1:.3f}$")
    ax1.legend(frameon=False)

    ax2.plot(x[m], Vp[m], color=C_FREE, label=r"$V'(x)$")
    ax2.plot(x[m], 2.0 * Hmu[m], "--", color="#c0392b", label=r"$2H\mu_{*}(x)$")
    ax2.set_xlabel("$x$")
    ax2.set_title("stationarity identity of Theorem 8.11")
    ax2.legend(frameon=False)
    save(fig, "fig14_design_mp.png")

    report("exp06_design_mp", {
        "L": L,
        "support_theory": [lo, hi],
        "support_empirical": [float(final.min()), float(final.max())],
        "W1_terminal": w1,
        "stationarity_relative_error_interior": stat_err,
        "edge_exponent_lower": edge_lo,
        "edge_exponent_upper": edge_hi,
        "edge_exponent_expected": 0.5,
    })


if __name__ == "__main__":
    main()
