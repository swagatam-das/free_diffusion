"""Targets of increasing complexity (Section 6 of the paper).

The designed flow with f = 1 (b = -H mu_*) is run as a noise-free particle system (``freeddpm.particles``) from a narrow flat
band to five targets: Marchenko--Pastur (one cut, strictly convex potential), a symmetric and an asymmetric two-cut law, a three-cut
law, and Marchenko--Pastur with two isolated bumps (the structure of the S&P 500 spectrum).  Theorem 4.2 makes every target
stationary; the experiment measures whether and how fast each is reached.  For every gap in a target the gap-closing time of
Theorem 4.8, ``log(1 + 1/T)`` with ``T = min_gap int dmu(y)/(u-y)^2``, is reported alongside.  Particles cannot cross, so the
number of particles in each cut is fixed once the gaps have opened; the final counts are compared with the target masses.
"""

import numpy as np

from _common import cache_load, cache_save, cfg, out_of_time, parse, report
from freeddpm.design import design_drift, mp_density
from freeddpm.forward import EmpiricalLaw
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.particles import forward_flow
from freeddpm.plotting import save, use_style

import matplotlib.pyplot as plt

ONE = lambda z: np.ones_like(z)


def atoms_density(x, atoms, weights, v):
    a = 1.0 - v                                       # density(x, a): law of D_{sqrt a}(atoms') boxplus gamma_{1-a}
    law = EmpiricalLaw(np.array(atoms) / np.sqrt(a), np.array(weights))
    return np.clip(law.density(x, a, eps=1e-4), 0.0, None)


def bump(x, c, r, mass):
    return mass * np.sqrt(np.clip(r**2 - (x - c) ** 2, 0, None)) * 2 / (np.pi * r**2)


def targets():
    """name -> (grid lower end, grid upper end, density function of x)."""
    mp = lambda x: mp_density(x, 0.5)
    return {
        "Marchenko--Pastur": (-0.2, 3.4, mp),
        "two cuts, symmetric": (-3.6, 3.6, lambda x: atoms_density(x, (-1.6, 1.6), (0.5, 0.5), 0.3)),
        "two cuts, 0.3/0.7": (-3.6, 3.6, lambda x: atoms_density(x, (-1.6, 1.6), (0.3, 0.7), 0.3)),
        "three cuts": (-3.6, 3.4, lambda x: atoms_density(x, (-2.3, 0.4, 1.9), (0.5, 0.2, 0.3), 0.1)),
        "MP + two bumps": (-0.2, 5.4, lambda x: 0.97 * mp(x) + bump(x, 3.6, 0.15, 0.015) + bump(x, 4.6, 0.15, 0.015)),
    }


def components(x, psi, thr=1e-3, min_gap=0.08):
    """Cuts of the support (index ranges) and the midpoints of the gaps between them."""
    on = psi > thr * psi.max()
    idx = np.where(on)[0]
    cuts, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if x[b] - x[a] > min_gap:
            cuts.append((start, a)); start = b
    cuts.append((start, idx[-1]))
    return cuts, [0.5 * (x[c1[1]] + x[c2[0]]) for c1, c2 in zip(cuts[:-1], cuts[1:])]


def first_below(t, w, thr):
    i = np.where(w < thr)[0]
    if not i.size:
        return None
    i = i[0]
    return float(t[0]) if i == 0 else float(t[i - 1] + (np.log(thr) - np.log(w[i - 1])) / (np.log(w[i]) - np.log(w[i - 1])) * (t[i] - t[i - 1]))


def main():
    args = parse(__doc__, horizon=dict(type=float, default=0.0, help="final time T (default 8; 3 with --quick)"), snap=dict(type=float, default=0.0, help="snapshot interval"),
                 budget=dict(type=float, default=0.0, help="stop after this many seconds (resume by re-running)"))
    M, T, snap_dt = cfg(args, 400, 150), (args.horizon or cfg(args, 8.0, 3.0)), (args.snap or cfg(args, 0.02, 0.05))
    npts = cfg(args, 8000, 3000)
    res, curves, finals = {}, {}, {}
    cfgkey = f"M{M}_T{T}_sd{snap_dt}_n{npts}"
    units = {k: v for k, v in cache_load("exp13").items() if k[0] == cfgkey}
    for name, (xlo, xhi, dens) in targets().items():
        x = np.linspace(xlo, xhi, npts); h = x[1] - x[0]
        if (cfgkey, name) in units:
            res[name], curves[name], finals[name] = units[(cfgkey, name)]
            continue
        if out_of_time(args):
            cache_save("exp13", units); print("budget reached; re-run to continue"); raise SystemExit(3)
        psi = dens(x); psi = psi / (psi.sum() * h)
        cdf = np.cumsum(psi) * h; cdf /= cdf[-1]
        b_grid = design_drift(x, psi); b = lambda z, bg=b_grid: np.interp(z, x, bg)
        cuts, seps = components(x, psi)
        mass = [float(psi[a:c + 1].sum() * h) for a, c in cuts]
        A, B = x[cuts[0][0]], x[cuts[-1][1]]
        c0, r0 = 0.5 * (A + B), 0.5 * (B - A)
        Q0 = c0 + 0.5 * r0 * np.linspace(-1, 1, M)           # flat band, half the width of the support
        t, snaps, n = forward_flow(Q0, b, ONE, T, snap_dt, lo=x[0], hi=x[-1])
        w1 = np.array([w1_density_vs_samples(x, psi, q) for q in snaps])
        qstat = np.interp((np.arange(M) + 0.5) / M, cdf, x)
        _, st, _ = forward_flow(qstat, b, ONE, 1.0, 0.1, lo=x[0], hi=x[-1])
        counts = np.histogram(snaps[-1], bins=[-np.inf] + seps + [np.inf])[0] / M
        # Theorem 4.8 for every gap of the target
        lam = []
        for (c1, c2) in zip(cuts[:-1], cuts[1:]):
            u = x[c1[1] + 1:c2[0]]
            Th = np.array([np.sum(psi[np.abs(x - uu) > 1e-9] * h / (uu - x[np.abs(x - uu) > 1e-9]) ** 2) for uu in u])
            lam.append(float(np.log(1 + 1 / Th.min())))
        res[name] = dict(n_cuts=len(cuts), target_mass_per_cut=mass, final_fraction_per_cut=[float(c) for c in counts],
                         W1_final=float(w1[-1]), W1_min=float(w1.min()), time_to_W1_below_0p05=first_below(t, w1, 0.05), W1_at_times={str(tt): float(w1[int(np.argmin(np.abs(t - tt)))]) for tt in (2.0, 4.0, 8.0, 16.0, 24.0, 40.0) if tt <= T + 1e-9},
                         stationarity_W1_from_target=float(w1_density_vs_samples(x, psi, st[-1])), gap_closing_Lambda_star=lam, steps=int(n))
        curves[name] = (t, w1); finals[name] = (x, psi, snaps[-1], cuts)
        units[(cfgkey, name)] = (res[name], curves[name], finals[name]); cache_save("exp13", units)
        print(f"  {name:22s} cuts {len(cuts)}  W1(T) {w1[-1]:.4f}  min {w1.min():.4f}  mass target {np.round(mass,3)} final {np.round(counts,3)}  stationarity {res[name]['stationarity_W1_from_target']:.4f}", flush=True)

    use_style()
    fig, ax = plt.subplots(1, 6, figsize=(15.5, 2.7)); fig.subplots_adjust(wspace=0.35)
    for name, (t, w1) in curves.items():
        ax[0].semilogy(t, np.maximum(w1, 1e-4), label=name)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$W_1(\mu_t,\mu_*)$"); ax[0].legend(frameon=False, fontsize=6)
    for a, (name, (xx, psi, q, cuts)) in zip(ax[1:], finals.items()):
        lo, hi = xx[cuts[0][0]] - 0.4, xx[cuts[-1][1]] + 0.4
        a.hist(q, bins=40, range=(lo, hi), density=True, color="0.8"); a.plot(xx, psi, "k", lw=1); a.set_xlim(lo, hi)
        a.set_title(name, fontsize=8); a.set_xlabel("$x$")
    save(fig, "fig18_target_complexity.png")
    out = {"M_particles": M, "T": T, "snapshot_interval": snap_dt, "flow": "designed, f = 1"}
    for k, v in res.items():
        for kk, vv in v.items(): out[f"{k} | {kk}"] = vv
    report("exp13_target_complexity", out)


if __name__ == "__main__":
    main()
