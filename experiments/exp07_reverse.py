"""Section 10.7 -- reverse-time reconstruction.

The deterministic probability-flow form of Corollary 7.4,

    y' = (beta/2) (y - xi_{mu_t}(y)),   t decreasing,

is integrated from the near-equilibrium law back to the two-atom law, using the
exact free score from the cubic (9.5).  The flow has the same marginals as the
reverse SDE of Theorem 7.1 but is deterministic, so the reconstruction error is
a pure discretisation error and does not average out.

Also reported is the sign check of Remark 7.3: at equilibrium xi_gamma(y) = y,
so the reverse drift must collapse to the forward drift -beta y / 2.  With the
opposite sign convention for xi it would equal +3 beta y / 2, and the flow would
diverge.  This is the diagnostic the paper recommends for any variant of the
construction.

Produces fig4_reverse_reconstruction.png.
"""

from __future__ import annotations

import numpy as np

from _common import A0, BETA, cfg, parse, report, rng
from freeddpm.forward import Schedule, TwoAtomLaw
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.plotting import C_DATA, C_FREE, save, use_style
from freeddpm.reverse import probability_flow, stationarity_residual

import matplotlib.pyplot as plt


def main():
    args = parse(__doc__)
    r = rng(args)
    law = TwoAtomLaw(A0)
    x = np.linspace(-5.0, 5.0, cfg(args, 2001, 801))

    # start from mu_t with v = 0.985, i.e. alpha = 0.015
    alpha_start = 1.0 - 0.985
    alpha_end = 0.90
    n_steps = cfg(args, 600, 200)
    alphas = np.exp(np.linspace(np.log(alpha_start), np.log(alpha_end), n_steps))

    # sample the initial ensemble from mu_{t_0} by inverse transform
    n_particles = cfg(args, 60000, 8000)
    psi0 = np.clip(law.density(x, alpha_start), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (psi0[1:] + psi0[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    y0 = np.interp(r.random(n_particles), cdf, x)

    # Tabulate the score on the spatial grid at each schedule level and
    # interpolate, rather than solving the cubic at all 6e4 particle positions
    # every step.  The cubic is solved once per grid point per level; linear
    # interpolation between grid points is far below the discretisation error of
    # the flow itself, and the same device is used in the spiked experiment.
    eps_s = cfg(args, 1e-5, 1e-4)
    levels = np.unique(np.concatenate([alphas, np.exp(
        -0.5 * (np.log(alphas[:-1]) + np.log(alphas[1:])) * -1.0)]))
    tables = {float(a): law.score(x, a, eps=eps_s) for a in levels}
    keys = np.array(sorted(tables))
    stack = np.array([tables[float(k)] for k in keys])

    def score(y, alpha):
        a = float(alpha)
        j = int(np.clip(np.searchsorted(keys, a), 1, keys.size - 1))
        w = (a - keys[j - 1]) / (keys[j] - keys[j - 1])
        xi = (1.0 - w) * stack[j - 1] + w * stack[j]
        return np.interp(y, x, xi)

    y, history = probability_flow(y0, alphas, score, beta=BETA, method="midpoint")

    # snapshots
    show = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    use_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=False)
    errs = {}
    for ax, k in zip(axes, show):
        a = alphas[k]
        psi = law.density(x, a)
        ens = history[k]
        ax.hist(ens, bins=70, density=True, color=C_DATA, alpha=0.6, label="flow")
        ax.plot(x, psi, color=C_FREE, label=r"target $\psi_t$")
        ax.set_title(rf"$\alpha_t={a:.3f}$")
        ax.set_xlim(-4.2, 4.2)
        ax.set_xlabel("$x$")
        errs[f"W1_alpha_{a:.3f}"] = w1_density_vs_samples(x, psi, ens)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    save(fig, "fig4_reverse_reconstruction.png")

    resid = float(np.max(np.abs(stationarity_residual(np.linspace(-3, 3, 101), BETA))))

    report("exp07_reverse", {
        **errs,
        "W1_terminal": w1_density_vs_samples(x, law.density(x, alpha_end), y),
        "n_particles": n_particles,
        "n_steps": n_steps,
        "sign_check_residual": resid,
    })


if __name__ == "__main__":
    main()
