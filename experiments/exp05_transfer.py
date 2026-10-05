"""Section 10.5 -- one score for every dimension (failure mode F2).

The reverse-time matrix SDE of Theorem 7.1 is run at several dimensions, driven
in every case by the *same* scalar free score, computed once from the cubic
(9.5).  At each step the current matrix is diagonalised, the score is applied
spectrally, an Euler--Maruyama step is taken with Hermitian Gaussian noise, and
the iterate is re-symmetrised; this is Algorithm 1.

An entrywise score would be an N^2-dimensional object tied to the dimension it
was estimated at.  The free score is a single scalar function, so the same
object drives every N, and the reconstruction error decreases with N at the
rate of the empirical spectral fluctuation.

Produces fig6_dimension_transfer.png.
"""

from __future__ import annotations

import numpy as np

from _common import A0, BETA, cfg, parse, report, rng
from freeddpm.forward import TwoAtomLaw
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.plotting import C_DATA, C_FREE, save, use_style
from freeddpm.reverse import reverse_matrix_sde

import matplotlib.pyplot as plt


def main():
    args = parse(__doc__)
    r = rng(args)
    law = TwoAtomLaw(A0)

    x = np.linspace(-5.0, 5.0, cfg(args, 2001, 801))
    alpha_lo, alpha_hi = 0.02, 0.90
    n_steps = cfg(args, 400, 120)
    alphas = np.exp(np.linspace(np.log(alpha_lo), np.log(alpha_hi), n_steps))

    def score(lam, alpha):
        return law.score(lam, alpha, eps=cfg(args, 1e-5, 1e-4))

    Ns = cfg(args, [50, 200, 600], [40, 100, 200])
    psi_target = law.density(x, alpha_hi)

    errs = {}
    use_style()
    fig, axes = plt.subplots(1, len(Ns) + 1, figsize=(3.3 * (len(Ns) + 1), 3.0))
    for ax, N in zip(axes, Ns):
        Y = reverse_matrix_sde(N, alphas, score, beta=BETA, rng=r)
        ev = np.linalg.eigvalsh(Y)
        errs[N] = w1_density_vs_samples(x, psi_target, ev)
        ax.hist(ev, bins=45, density=True, color=C_DATA, alpha=0.6, label="generated")
        ax.plot(x, psi_target, color=C_FREE, label="target")
        ax.set_title(rf"$N={N}$, $W_1={errs[N]:.3f}$")
        ax.set_xlim(-4.2, 4.2)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)

    axN = axes[-1]
    Nv = np.array(Ns, float)
    ev_arr = np.array([errs[N] for N in Ns])
    axN.loglog(Nv, ev_arr, "o-", color=C_FREE, label="reconstruction $W_1$")
    axN.loglog(Nv, ev_arr[0] * (Nv / Nv[0]) ** -0.5, "k--", lw=1.0, label=r"$N^{-1/2}$")
    axN.set_xlabel("$N$")
    axN.set_title("error against dimension")
    axN.legend(frameon=False)
    save(fig, "fig6_dimension_transfer.png")

    report("exp05_transfer", {
        **{f"W1_N_{N}": errs[N] for N in Ns},
        "slope": float(np.polyfit(np.log(Nv), np.log(ev_arr), 1)[0]),
        "n_reverse_steps": n_steps,
        "alpha_range": [alpha_lo, alpha_hi],
    })


if __name__ == "__main__":
    main()
