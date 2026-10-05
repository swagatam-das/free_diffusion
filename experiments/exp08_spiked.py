"""Section 10.8 -- a spiked covariance model.

The initial law is the spectral law of a centred sample covariance matrix from
a factor model: N = 600, aspect ratio n/N = 2, population covariance
diag(6, 4, 3, 1, ..., 1).  So mu_0 has a Marchenko--Pastur-type bulk together
with three outliers, and no closed form is available.

The marginals are computed by the subordination fixed point (10.1),

    omega(z) = z - (1 - alpha) G_{mu_t}(z),
    G_{mu_t}(z) = G_{D_sqrt(alpha) mu_0}(omega(z)),

by damped iteration on z = x + i eps, using that omega maps the upper half
plane into itself.

Two figures: fig7_spiked_forward.png and fig8_spiked_reverse.png.

The outliers carry mass 3/N, which vanishes in the limit.  They are therefore a
finite-rank effect of Baik--Ben Arous--Peche type, invisible to mu_0 in the
limit, and Theorem 9.5 -- which concerns gaps with positive mass on both sides
-- does not govern them.  No absorption time is reported for that reason: near
an outlier of mass O(1/N) the density is comparable to the regularisation eps
used to solve the fixed point, so any threshold-based detection of gap closure
measures the threshold rather than the flow.
"""

from __future__ import annotations

import numpy as np

from _common import BETA, cfg, parse, report, rng
from freeddpm.forward import EmpiricalLaw, Schedule
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.plotting import C_DATA, C_FREE, save, use_style
from freeddpm.reverse import probability_flow

import matplotlib.pyplot as plt


def spiked_spectrum(N, ratio, spikes, rng):
    """Eigenvalues of a centred sample covariance matrix with planted spikes."""
    n = int(ratio * N)
    pop = np.ones(N)
    pop[: len(spikes)] = spikes
    Z = rng.normal(size=(N, n))
    X = (np.sqrt(pop)[:, None] * Z) / np.sqrt(n)
    S = X @ X.conj().T
    ev = np.linalg.eigvalsh(S)
    # centre and scale so that the bulk sits on an O(1) scale comparable to the
    # semicircular reference used by the diffusion
    ev = ev - ev.mean()
    return ev / ev.std()


def main():
    args = parse(__doc__)
    r = rng(args)
    N = cfg(args, 600, 200)
    law_atoms = spiked_spectrum(N, 2.0, [6.0, 4.0, 3.0], r)
    law = EmpiricalLaw(law_atoms)

    x = np.linspace(law_atoms.min() - 2.5, law_atoms.max() + 2.5, cfg(args, 1200, 500))
    eps = cfg(args, 2e-3, 5e-3)
    sub_kw = dict(damping=cfg(args, 0.7, 0.7), max_iter=cfg(args, 2500, 1500), tol=1e-10)

    # ---------------- forward ---------------------------------------------
    alphas_fwd = [0.95, 0.79, 0.50, 0.25]
    use_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))
    fwd_err = {}
    for ax, a in zip(axes, alphas_fwd):
        psi = np.clip(law.density(x, a, eps=eps, **sub_kw), 0.0, None)
        X = law.sample_matrix(N, a, r)
        ev = np.linalg.eigvalsh(X)
        ax.hist(ev, bins=60, density=True, color=C_DATA, alpha=0.6, label="eigenvalues")
        ax.plot(x, psi, color=C_FREE, label="free theory")
        ax.set_title(rf"$\alpha_t={a:.2f}$")
        ax.set_xlabel("$x$")
        fwd_err[f"W1_forward_alpha_{a}"] = w1_density_vs_samples(x, psi, ev)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    save(fig, "fig7_spiked_forward.png")

    # ---------------- reverse ---------------------------------------------
    alpha_start, alpha_end = 0.015, 0.79
    n_steps = cfg(args, 300, 90)
    alphas = np.exp(np.linspace(np.log(alpha_start), np.log(alpha_end), n_steps))
    n_particles = cfg(args, 60000, 6000)

    psi0 = np.clip(law.density(x, alpha_start, eps=eps, **sub_kw), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (psi0[1:] + psi0[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    y0 = np.interp(r.random(n_particles), cdf, x)

    # Pre-tabulate the score on the spatial grid and interpolate, in two respects.
    #
    #   In space: evaluating the subordination fixed point at each of the 6e4
    #   particle positions on every step would repeat the same solve 6e4 times.
    #
    #   In the noise level: the score depends smoothly on alpha, so it is tabulated
    #   on a coarse geometric grid of levels and interpolated linearly onto the finer
    #   grid the flow is integrated on.  Each level is warm-started from the
    #   subordination function of the previous one.  The accuracy of the second
    #   approximation is checked below against a direct solve at a level that is not
    #   in the table, and the relative error is reported with the results.
    from freeddpm.cauchy import score_from_g, subordination_g

    n_levels = cfg(args, n_steps, 12)
    levels = np.exp(np.linspace(np.log(alphas[0]), np.log(alphas[-1]), n_levels))

    tables = np.empty((n_levels, x.size))
    omega = None
    zgrid = x + 1j * eps
    for k, a in enumerate(levels):
        g, omega = subordination_g(zgrid, np.sqrt(a) * law.atoms, law.weights,
                                   v=1.0 - a, omega0=omega, return_omega=True,
                                   **sub_kw)
        tables[k] = score_from_g(g)

    def score(y, alpha):
        a = float(alpha)
        j = int(np.clip(np.searchsorted(levels, a), 1, n_levels - 1))
        w = (a - levels[j - 1]) / (levels[j] - levels[j - 1])
        xi = (1.0 - w) * tables[j - 1] + w * tables[j]
        return np.interp(y, x, xi)

    k_mid = n_levels // 2
    a_chk = float(np.sqrt(levels[k_mid] * levels[k_mid + 1]))
    xi_exact = score_from_g(subordination_g(zgrid, np.sqrt(a_chk) * law.atoms,
                                            law.weights, v=1.0 - a_chk, **sub_kw))
    interp_err = float(np.max(np.abs(xi_exact - score(x, a_chk)))
                       / max(np.max(np.abs(xi_exact)), 1e-12))

    y, history = probability_flow(y0, alphas, score, beta=BETA, method="midpoint")

    show = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))
    rev_err = {}
    for ax, k in zip(axes, show):
        a = alphas[k]
        psi = np.clip(law.density(x, a, eps=eps, **sub_kw), 0.0, None)
        ens = history[k]
        ax.hist(ens, bins=60, density=True, color=C_DATA, alpha=0.6, label="flow")
        ax.plot(x, psi, color=C_FREE, label="target")
        ax.set_title(rf"$\alpha_t={a:.3f}$")
        ax.set_xlabel("$x$")
        rev_err[f"W1_reverse_alpha_{a:.3f}"] = w1_density_vs_samples(x, psi, ens)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    save(fig, "fig8_spiked_reverse.png")

    report("exp08_spiked", {
        "N": N,
        "n_spikes": 3,
        "outlier_mass": 3.0 / N,
        "regularisation_eps": eps,
        "score_table_levels": n_levels,
        "alpha_interpolation_relative_error": interp_err,
        **fwd_err,
        **rev_err,
    })


if __name__ == "__main__":
    main()
