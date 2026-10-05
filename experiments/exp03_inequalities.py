"""Section 10.3 -- entropy dissipation and the functional inequalities.

Along the forward flow started from the two-atom law, three ratios are formed
from Theorems 4.4--4.6, each of which the theory requires to be at most 1:

    LSI        D(mu_t || gamma) / (I(mu_t || gamma)/2)
    Talagrand  W_2(mu_t, gamma)^2 / (2 D(mu_t || gamma))
    HWI        D / (W sqrt(I) - W^2/2)

The middle panel checks the exponential decay D(mu_t) <= e^{-Lambda(t)} D(mu_0)
of Theorem 4.3, and the right panel verifies the free de Bruijn identity
-dD/dLambda = I/2 pointwise, by comparing a centred numerical derivative of the
free energy with I/2.  The de Bruijn check is a direct test of the constant
beta/2, which is the point of Remark 4.2.

Produces fig13_inequalities.png.

Numerical note.  The logarithmic energy is evaluated on a uniform grid with the
diagonal cell replaced by the exact local average log(h/2) - 1; see
freeddpm.functionals.log_energy_density.  The grid must cover the support of
mu_t with margin, and the relative quantities are differences of two O(1)
numbers, so a fine grid is needed for the de Bruijn derivative.
"""

from __future__ import annotations

import numpy as np

from _common import A0, BETA, cfg, parse, report, rng
from freeddpm.forward import Schedule, TwoAtomLaw
from freeddpm.functionals import (
    inequality_ratios,
    semicircle_density,
    relative_fisher,
    relative_free_energy,
)
from freeddpm.plotting import C_CLASSICAL, C_FREE, save, use_style

import matplotlib.pyplot as plt


def main():
    args = parse(__doc__)
    rng(args)
    sched = Schedule(BETA)
    law = TwoAtomLaw(A0)

    n_grid = cfg(args, 6001, 1601)
    x = np.linspace(-6.0, 6.0, n_grid)
    n_t = cfg(args, 60, 24)
    Lam = np.linspace(0.05, 3.0, n_t)
    alphas = np.exp(-Lam)

    # Quadrature floor.  D and I are differences of O(1) logarithmic energies,
    # so once mu_t is within the quadrature error of gamma the ratios are
    # meaningless.  Evaluating D for the exact semicircular law on the same
    # grid, where the true value is zero, gives the floor directly.
    floor = abs(relative_free_energy(x, semicircle_density(x, 1.0)))

    rows = []
    for a in alphas:
        g = law.g(x, a, eps=cfg(args, 1e-7, 1e-6))
        psi = np.clip(-np.imag(g) / np.pi, 0.0, None)
        psi = psi / np.trapezoid(psi, x)
        xi = 2.0 * np.real(g)
        rows.append(inequality_ratios(x, psi, xi))

    D = np.array([r["D"] for r in rows])
    I = np.array([r["I"] for r in rows])
    lsi = np.array([r["lsi"] for r in rows])
    tal = np.array([r["talagrand"] for r in rows])
    hwi = np.array([r["hwi"] for r in rows])

    trust = D > 20.0 * floor

    # de Bruijn: -dD/dLambda should equal I/2
    dD = np.gradient(D, Lam)
    debruijn_lhs = -dD
    debruijn_rhs = 0.5 * I
    m = np.isfinite(debruijn_lhs) & (debruijn_rhs > 0) & (D > 20.0 * abs(
        relative_free_energy(x, semicircle_density(x, 1.0))))
    rel = np.abs(debruijn_lhs[m] - debruijn_rhs[m]) / debruijn_rhs[m]

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

    axes[0].plot(Lam[trust], lsi[trust], color=C_FREE, label="free LSI")
    axes[0].plot(Lam[trust], tal[trust], color=C_CLASSICAL, label="Talagrand")
    axes[0].plot(Lam[trust], hwi[trust], color="#1a8a5a", label="HWI")
    axes[0].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xlabel(r"$\Lambda(t)$")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("inequalities as ratios (must stay below 1)")
    axes[0].legend(frameon=False)

    axes[1].semilogy(Lam, D, color=C_FREE, label=r"$D(\mu_t\|\gamma)$")
    axes[1].semilogy(Lam, D[0] * np.exp(-(Lam - Lam[0])), "k--", lw=1.0,
                     label=r"$e^{-\Lambda(t)}D(\mu_0\|\gamma)$")
    axes[1].set_xlabel(r"$\Lambda(t)$")
    axes[1].set_title("exponential relaxation")
    axes[1].legend(frameon=False)

    axes[2].loglog(debruijn_rhs[m], debruijn_lhs[m], "o", ms=3.5, color=C_FREE)
    lim = [min(debruijn_rhs[m].min(), debruijn_lhs[m].min()),
           max(debruijn_rhs[m].max(), debruijn_lhs[m].max())]
    axes[2].plot(lim, lim, "k--", lw=1.0)
    axes[2].set_xlabel(r"$\frac{1}{2} I(\mu_t\|\gamma)$")
    axes[2].set_ylabel(r"$-\,dD/d\Lambda$")
    axes[2].set_title("free de Bruijn identity")
    save(fig, "fig13_inequalities.png")

    max_lsi = float(np.nanmax(lsi[trust]))
    max_tal = float(np.nanmax(tal[trust]))
    max_hwi = float(np.nanmax(hwi[trust]))

    report("exp03_inequalities", {
        "quadrature_floor_on_D": float(floor),
        "points_kept": int(trust.sum()),
        "points_dropped_below_floor": int((~trust).sum()),
        "max_lsi_ratio": max_lsi,
        "max_talagrand_ratio": max_tal,
        "max_hwi_ratio": max_hwi,
        "all_ratios_below_one": bool(max(max_lsi, max_tal, max_hwi) < 1.0),
        "debruijn_median_relative_error": float(np.median(rel)),
        "debruijn_max_relative_error": float(rel.max()),
        "decay_ratio_max": float(np.max(D / (D[0] * np.exp(-(Lam - Lam[0]))))),
    })


if __name__ == "__main__":
    main()
