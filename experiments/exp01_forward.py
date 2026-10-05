"""Section 10.1 -- forward dynamics and the hydrodynamic limit.

Produces three figures:

fig1_forward_convergence.png   empirical spectrum at N=1200 against the free
                               marginal, at four values of alpha
fig10_spacetime.png            the density over the (Lambda, x) plane with the
                               support and gap edges, and eigenvalue paths at
                               N=40 with the free edges superimposed
fig11_rate.png                 W_1 between the empirical spectral distribution
                               and the free marginal against N, log-log

and reports the fitted convergence exponents, which the paper quotes as
-0.535 at alpha = 0.3 and -0.511 at alpha = 0.6.
"""

from __future__ import annotations

import numpy as np

from _common import A0, BETA, cfg, parse, report, rng
from freeddpm.forward import Schedule, TwoAtomLaw, transition_alpha
from freeddpm.functionals import w1_density_vs_samples, w1_samples
from freeddpm.matrix import dyson_path
from freeddpm.plotting import C_DATA, C_FREE, save, use_style

import matplotlib.pyplot as plt


def support_edges(x, psi, threshold=1e-3):
    """Outer edges and inner gap edges of the numerically computed support."""
    m = psi > threshold
    if not m.any():
        return None
    idx = np.flatnonzero(m)
    lo, hi = x[idx[0]], x[idx[-1]]
    # a gap is a maximal run of False strictly inside [lo, hi]
    inner = ~m[idx[0] : idx[-1] + 1]
    if not inner.any():
        return lo, hi, None, None
    j = np.flatnonzero(inner)
    return lo, hi, x[idx[0] + j[0]], x[idx[0] + j[-1]]


def main():
    args = parse(__doc__)
    r = rng(args)
    sched = Schedule(BETA)
    law = TwoAtomLaw(A0)

    N_big = cfg(args, 1200, 300)
    n_grid = cfg(args, 2001, 801)
    x = np.linspace(-5.0, 5.0, n_grid)

    # ---------------- figure 1: four snapshots ----------------------------
    alphas = [0.90, 0.60, 0.30, 0.08]
    use_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.9), sharey=True)
    w1_snap = {}
    for ax, a in zip(axes, alphas):
        X = law.sample_matrix(N_big, a, r)
        ev = np.linalg.eigvalsh(X)
        psi = law.density(x, a)
        ax.hist(ev, bins=70, density=True, color=C_DATA, alpha=0.55,
                label=f"eigenvalues, $N={N_big}$")
        ax.plot(x, psi, color=C_FREE, label="free marginal")
        ax.set_title(rf"$\alpha_t={a:.2f}$")
        ax.set_xlim(-4.2, 4.2)
        ax.set_xlabel("$x$")
        w1_snap[f"W1_alpha_{a}"] = w1_density_vs_samples(x, psi, ev)
    axes[0].set_ylabel(r"$\psi_t$")
    axes[0].legend(loc="upper left", frameon=False)
    save(fig, "fig1_forward_convergence.png")

    # ---------------- figure 10: spacetime --------------------------------
    n_t = cfg(args, 260, 90)
    Lam = np.linspace(0.02, 3.2, n_t)
    al = np.exp(-Lam)
    dens = np.empty((n_t, x.size))
    edges = np.full((n_t, 4), np.nan)
    for i, a in enumerate(al):
        psi = law.density(x, a)
        dens[i] = psi
        e = support_edges(x, psi)
        if e is not None:
            for j, val in enumerate(e):
                if val is not None:
                    edges[i, j] = val

    N_small = cfg(args, 40, 20)
    lam0 = np.where(np.arange(N_small) < N_small // 2, -A0, A0).astype(float)
    lam0 = lam0 + 1e-3 * np.arange(N_small)  # break the exact degeneracy
    t_grid = Lam / BETA
    paths = dyson_path(lam0, t_grid, beta=BETA, rng=r, dt=cfg(args, 2e-4, 1e-3))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.6))
    im = axL.pcolormesh(Lam, x, dens.T, shading="auto", cmap="magma")
    axL.plot(Lam, edges[:, 0], color="w", lw=1.0)
    axL.plot(Lam, edges[:, 1], color="w", lw=1.0)
    axL.plot(Lam, edges[:, 2], color="w", lw=1.0, ls="--")
    axL.plot(Lam, edges[:, 3], color="w", lw=1.0, ls="--")
    Lstar = np.log(1.0 + A0**2)
    axL.axvline(Lstar, color="c", ls=":", lw=1.3)
    axL.text(Lstar, 4.3, r"$\Lambda^{*}$", color="c", ha="center")
    axL.set_xlabel(r"$\Lambda(t)$")
    axL.set_ylabel("$x$")
    axL.set_ylim(-4.6, 4.9)
    axL.set_title("spectral density with free support edges")
    fig.colorbar(im, ax=axL, pad=0.02)

    axR.plot(Lam, paths, color="k", lw=0.4, alpha=0.75)
    axR.plot(Lam, edges[:, 0], color=C_FREE, lw=1.4)
    axR.plot(Lam, edges[:, 1], color=C_FREE, lw=1.4, label="free support edges")
    axR.set_xlabel(r"$\Lambda(t)$")
    axR.set_title(rf"eigenvalue paths, $N={N_small}$")
    axR.legend(frameon=False)
    save(fig, "fig10_spacetime.png")

    n_cross = int(np.sum(np.diff(np.sort(paths, axis=1), axis=1) <= 0))
    frac_out = float(
        np.mean((paths < edges[:, [0]] - 0.05) | (paths > edges[:, [1]] + 0.05))
    )

    # ---------------- figure 11: convergence rate -------------------------
    # Two conventions are reported, and they differ.
    #
    #   "exact"      W_1 between the empirical spectral distribution and the
    #                *exact* free marginal mu_t.  This is the quantity
    #                Proposition 3.9 is about, and it decays like N^{-1}:
    #                eigenvalues of a deformed GUE are rigid, so their
    #                empirical measure is far closer to its limit than an iid
    #                sample of the same size would be.
    #
    #   "mc"         W_1 between the empirical spectral distribution and an
    #                equal-size Monte Carlo sample drawn from mu_t.  This decays
    #                like N^{-1/2}, but the exponent measured is that of the
    #                *reference* sample, not of the matrix model.
    #
    # The exponents -0.535 and -0.511 quoted in the paper correspond to "mc".
    # See README.md, "Known discrepancies".
    Ns = cfg(args, [25, 50, 100, 200, 400, 800], [25, 50, 100, 200])
    reps = cfg(args, 12, 4)
    slopes = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    for mode, ax in (("exact", ax1), ("mc", ax2)):
        for a, colour in zip((0.3, 0.6), (C_FREE, "#c0392b")):
            psi = np.clip(law.density(x, a), 0.0, None)
            cdf = np.concatenate(
                [[0.0], np.cumsum(0.5 * (psi[1:] + psi[:-1]) * np.diff(x))]
            )
            cdf /= cdf[-1]
            errs = []
            for N in Ns:
                vals = []
                for _ in range(reps):
                    ev = np.linalg.eigvalsh(law.sample_matrix(N, a, r))
                    if mode == "exact":
                        vals.append(w1_density_vs_samples(x, psi, ev))
                    else:
                        ref_sample = np.interp(r.random(N), cdf, x)
                        vals.append(w1_samples(ev, ref_sample))
                errs.append(np.mean(vals))
            errs = np.array(errs)
            slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
            slopes[f"slope_{mode}_alpha_{a}"] = slope
            ax.loglog(Ns, errs, "o-", color=colour,
                      label=rf"$\alpha_t={a}$, slope ${slope:.3f}$")
        guide = -1.0 if mode == "exact" else -0.5
        ref = errs[0] * (np.array(Ns, float) / Ns[0]) ** guide
        ax.loglog(Ns, ref, "k--", lw=1.0, label=rf"$N^{{{guide:g}}}$")
        ax.set_xlabel("$N$")
        ax.legend(frameon=False)
    ax1.set_ylabel(r"$W_1(L_N(t),\mu_t)$")
    ax1.set_title("against the exact free marginal")
    ax2.set_title("against an equal-size sample from $\\mu_t$")
    save(fig, "fig11_rate.png")

    report("exp01_forward", {
        **w1_snap,
        **slopes,
        "Lambda_star": Lstar,
        "alpha_star": transition_alpha(A0),
        "eigenvalue_crossings": n_cross,
        "fraction_outside_support": frac_out,
    })


if __name__ == "__main__":
    main()
