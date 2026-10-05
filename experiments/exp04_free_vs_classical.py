"""Section 10.4 -- free versus coordinatewise corruption (failure mode F1).

Corrupting a Hermitian matrix with Hermitian noise convolves its spectrum
freely; treating the eigenvalues as independent coordinates and adding
independent Gaussian noise convolves it classically.  Proposition 3.2 says the
two operations never coincide, and Remark 3.2 quantifies the discrepancy at
fourth order,

    m_4(mu boxplus gamma_v) - m_4(mu * N(0,v)) = -v (2 m_2 + v) < 0.

This script measures both sides.  For the two-atom law with a = 1.6 and v = 1
the predicted gap is -6.12, with free and classical fourth moments 18.7936 and
24.9136 respectively.

Produces fig5_free_vs_classical.png.
"""

from __future__ import annotations

import numpy as np

from _common import cfg, parse, report, rng
from freeddpm.cauchy import density_from_g, g_two_atom_cubic
from freeddpm.matrix import gue
from freeddpm.plotting import C_CLASSICAL, C_DATA, C_FREE, save, use_style

import matplotlib.pyplot as plt


def classical_density(x, a, v):
    """Density of ``(delta_{-a}+delta_a)/2 * N(0,v)``."""
    x = np.asarray(x, dtype=float)
    norm = 1.0 / np.sqrt(2.0 * np.pi * v)
    return 0.5 * norm * (np.exp(-((x - a) ** 2) / (2 * v)) + np.exp(-((x + a) ** 2) / (2 * v)))


def main():
    args = parse(__doc__)
    r = rng(args)
    a = 1.6
    m2 = a**2
    N = cfg(args, 2500, 500)
    N_moment = cfg(args, 3000, 600)
    x = np.linspace(-8.0, 8.0, cfg(args, 4001, 1601))

    variances = [0.25, 1.00, 2.56]
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2), sharey=False)
    table = {}

    for ax, v in zip(axes, variances):
        X0 = np.diag(np.where(np.arange(N) < N // 2, -a, a)).astype(complex)
        X = X0 + np.sqrt(v) * gue(N, r)
        ev = np.linalg.eigvalsh(X)

        psi_free = density_from_g(g_two_atom_cubic(x + 1e-6j, a, v))
        psi_cl = classical_density(x, a, v)

        ax.hist(ev, bins=70, density=True, color=C_DATA, alpha=0.55, label="eigenvalues")
        ax.plot(x, psi_free, color=C_FREE, label=r"$\mu_0\boxplus\gamma_v$")
        ax.plot(x, psi_cl, color=C_CLASSICAL, ls="--", label=r"$\mu_0 * N(0,v)$")
        ax.set_xlim(-6.5, 6.5)
        ax.set_title(f"$v={v}$")
        ax.set_xlabel("$x$")

        # W_1 of the empirical spectrum against each prediction
        from freeddpm.functionals import w1_density_vs_samples

        table[f"W1_free_v_{v}"] = w1_density_vs_samples(x, psi_free, ev)
        table[f"W1_classical_v_{v}"] = w1_density_vs_samples(x, psi_cl, ev)

    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    save(fig, "fig5_free_vs_classical.png")

    # ---- fourth moments at v = 1 -----------------------------------------
    v = 1.0
    psi_free = density_from_g(g_two_atom_cubic(x + 1e-7j, a, v))
    psi_free = psi_free / np.trapezoid(psi_free, x)
    m4_free_num = float(np.trapezoid(x**4 * psi_free, x))
    m4_free_exact = a**4 - 2 * m2**2 + 2 * (m2 + v) ** 2
    m4_cl_exact = a**4 - 3 * m2**2 + 3 * (m2 + v) ** 2

    X0 = np.diag(np.where(np.arange(N_moment) < N_moment // 2, -a, a)).astype(complex)
    ev = np.linalg.eigvalsh(X0 + np.sqrt(v) * gue(N_moment, r))
    m4_empirical = float(np.mean(ev**4))

    report("exp04_free_vs_classical", {
        **table,
        "m4_free_exact": m4_free_exact,
        "m4_free_quadrature": m4_free_num,
        "m4_classical_exact": m4_cl_exact,
        "m4_empirical_matrix": m4_empirical,
        "predicted_gap": -v * (2 * m2 + v),
        "measured_gap": m4_free_exact - m4_cl_exact,
    })


if __name__ == "__main__":
    main()
