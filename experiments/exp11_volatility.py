"""Section 10.10: what state-dependent volatility adds at a fixed equilibrium.

Target: Marchenko--Pastur (L = 1/2).  Four model classes are run from the same far-from-equilibrium start:

  (a) free OU: linear drift, f = 1                         -- equilibrium is the semicircle, whatever the target;
  (b) designed, f = 1: b = -H mu_* = -V'/2                 -- free Langevin dynamics, the Wasserstein gradient flow of F_V;
  (c) designed, f^2 = 2 (constant): b = -f^4 H mu_*        -- only a change of clock, c^4 = 4 times faster than (b);
  (d) designed, f^2(x) = 1/2 + 3x/(2(1+x)) non-constant     -- b = -f^2 H(f^2 mu_*), the new dynamics of Theorem 8.11.

(b), (c), (d) all have mu_* as stationary law; for a fixed target the f = 1 flow is unique, and a non-constant f is the
remaining design freedom.  The experiment measures W1(mu_t, mu_*) along each flow, checks that mu_* stays stationary under
(d), and reports the classical (coordinatewise) Langevin baseline, whose equilibrium exp(-V) is Gamma(2, rate 2).
"""

import numpy as np

from _common import cache_load, cache_save, cfg, out_of_time, parse, report, rng
from freeddpm.design import design_drift, mp_density, mp_potential_derivative, mp_support
from freeddpm.functionals import w1_density_vs_samples
from freeddpm.matrix import sandwich_path
from freeddpm.plotting import C_CLASSICAL, C_DATA, C_FREE, C_LEARNED, save, use_style

import matplotlib.pyplot as plt

L = 0.5
f2 = lambda z: 0.5 + 1.5 * z / (1.0 + z)


def main():
    args = parse(__doc__, budget=dict(type=float, default=0.0, help="stop after this many seconds (resume by re-running)"))
    N = cfg(args, 150, 70)
    T = cfg(args, 6.0, 3.0)
    dt = cfg(args, 2e-3, 4e-3)
    n_snap = cfg(args, 40, 16)
    n_grid = cfg(args, 3000, 900)
    lo, hi = mp_support(L)

    x = np.linspace(0.02, 3.8, n_grid)
    psi = mp_density(x, L)
    b1 = design_drift(x, psi)                       # -H mu_*  (= -V'/2 on the support)
    bf = design_drift(x, psi, f=np.sqrt(f2(x)))     # -f^2 H(f^2 mu_*)
    one = lambda z: np.ones_like(z)
    flows = {
        "free OU (a)": dict(key="a", b=lambda z: -0.5 * z, f=one, kw=dict()),
        "designed f=1 (b)": dict(key="b", b=lambda z: -0.5 * mp_potential_derivative(z, L), f=one, kw=dict(floor=0.02, ceil=3.8)),
        "designed f^2=2 (c)": dict(key="c", b=lambda z: -2.0 * mp_potential_derivative(z, L), f=lambda z: np.sqrt(2.0) * one(z), kw=dict(floor=0.02, ceil=3.8)),
        "designed f(x) (d)": dict(key="d", b=lambda z: np.interp(z, x, bf), f=lambda z: np.sqrt(f2(z)), kw=dict(floor=0.02, ceil=3.8)),
    }
    times = np.geomspace(0.02, T, n_snap)          # fine early resolution: the c^4-faster flow arrives within ~0.1
    xg = np.linspace(-3.0, 4.5, 6000)
    psi_g = mp_density(xg, L)

    cfgkey = f"N{N}_T{T}_dt{dt}_s{args.seed}" + ("_q" if args.quick else "")
    units = cache_load("exp11")
    units = {k: v for k, v in units.items() if k[0] == cfgkey}

    start = np.linspace(0.35, 1.9, N)               # a flat band, far from mu_*
    cdf = np.cumsum(psi) * (x[1] - x[0]); cdf /= cdf[-1]
    mp_quant = np.interp((np.arange(N) + 0.5) / N, cdf, x)     # a discretisation of mu_* itself

    # ---- relaxation from the flat band, and stationarity from mu_* itself ---------------------------
    for name, fl in flows.items():
        for task, X0, TT in (("relax", start, T), ("stat", mp_quant, cfg(args, 1.5, 0.8))):
            if task == "stat" and name.startswith("free OU"):
                continue
            key = (cfgkey, name, task)
            if key in units:
                continue
            if out_of_time(args):
                cache_save("exp11", units)
                print(f"budget reached; {len(units)} units cached; re-run to continue")
                raise SystemExit(3)
            ts = times if task == "relax" else np.array([TT])
            r = np.random.default_rng(args.seed + (0 if task == "relax" else 1))
            paths = sandwich_path(X0, ts, fl["b"], fl["f"], rng=r, dt=dt, **fl["kw"])
            units[key] = dict(times=ts, paths=paths, w1=[w1_density_vs_samples(xg, psi_g, p) for p in paths])
            print(f"  {name:22s} {task:5s} done: final W1 = {units[key]['w1'][-1]:.4f}", flush=True)
    cache_save("exp11", units)

    w1 = {n: np.array(units[(cfgkey, n, "relax")]["w1"]) for n in flows}
    stat = {n: float(units[(cfgkey, n, "stat")]["w1"][-1]) for n in flows if not n.startswith("free OU")}
    start_w1 = w1_density_vs_samples(xg, psi_g, mp_quant)

    def first_below(w, thr):
        """First time W1 falls below thr, interpolated linearly in log W1 between snapshots."""
        idx = np.where(w < thr)[0]
        if not idx.size:
            return None
        i = idx[0]
        if i == 0:
            return float(times[0])
        a, b_ = np.log(w[i - 1]), np.log(w[i])
        return float(times[i - 1] + (np.log(thr) - a) / (b_ - a) * (times[i] - times[i - 1]))

    # Edge arrival.  Under a change of clock (constant f) every time scale of the flow is multiplied by the
    # same factor, so the RATIO of the arrival times of the two spectral edges is invariant; a non-constant f
    # can change it.  Progress of a quantile q towards its Marchenko--Pastur value, started from the flat band.
    qlo, qhi = 0.05, 0.95
    tgt_lo, tgt_hi = float(np.interp(qlo, cdf, x)), float(np.interp(qhi, cdf, x))
    s_lo, s_hi = float(np.quantile(start, qlo)), float(np.quantile(start, qhi))

    def progress(paths, side):
        q = np.quantile(paths, qlo if side == "lo" else qhi, axis=1)
        return (s_lo - q) / (s_lo - tgt_lo) if side == "lo" else (q - s_hi) / (tgt_hi - s_hi)

    def arrival(paths, side, frac=0.8):
        prog = progress(paths, side)
        idx = np.where(prog >= frac)[0]
        if not idx.size:
            return None
        i = idx[0]
        if i == 0:
            return float(times[0])
        return float(times[i - 1] + (frac - prog[i - 1]) / (prog[i] - prog[i - 1]) * (times[i] - times[i - 1]))

    t_thr = {n: {"0.1": first_below(w1[n], 0.1), "0.05": first_below(w1[n], 0.05)} for n in flows}
    arr = {n: {"lo": arrival(units[(cfgkey, n, "relax")]["paths"], "lo"), "hi": arrival(units[(cfgkey, n, "relax")]["paths"], "hi")}
           for n in flows if not n.startswith("free OU")}
    ratio = {n: (a["lo"] / a["hi"] if a["lo"] and a["hi"] else None) for n, a in arr.items()}

    # ---- classical (coordinatewise) Langevin baseline: stationary law exp(-V) = Gamma(2, rate 2) --------------
    from scipy.stats import gamma
    u = (np.arange(40000) + 0.5) / 40000
    qmp = np.interp(u, cdf, x)
    w1_gamma = float(np.mean(np.abs(gamma.ppf(u, a=2.0, scale=0.5) - qmp)))

    # ---- figure -----------------------------------------------------------------------------------------
    use_style()
    plt.rcParams.update({"font.size": 6.8, "axes.titlesize": 7.2, "axes.labelsize": 6.8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8})
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 1.75))
    fig.subplots_adjust(wspace=0.62, left=0.075, right=0.985, bottom=0.2, top=0.88)
    cols = {"free OU (a)": C_CLASSICAL, "designed f=1 (b)": C_FREE, "designed f^2=2 (c)": C_DATA, "designed f(x) (d)": C_LEARNED}
    lab = {"free OU (a)": "(a) free OU", "designed f=1 (b)": r"(b) $f\equiv1$", "designed f^2=2 (c)": r"(c) $f^2\equiv2$", "designed f(x) (d)": r"(d) non-constant $f$"}
    for n in flows:
        ax[0].semilogy(times, w1[n], marker="o", ms=1.6, lw=0.9, color=cols[n], label=lab[n])
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$W_1(\mu_t,\mu_*)$", labelpad=1); ax[0].set_title("relaxation")
    ax[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.04, 0.52), handlelength=1.2, borderpad=0.1, labelspacing=0.2)
    m = (x > lo + 0.1) & (x < hi - 0.1)
    ax[1].plot(x[m], b1[m], color=C_FREE, lw=1.0, label=r"$b_*$, $f\equiv1$"); ax[1].plot(x[m], bf[m], color=C_LEARNED, lw=1.0, label=r"$b_*$, non-const. $f$")
    ax2 = ax[1].twinx(); ax2.plot(x[m], f2(x[m]), color="k", ls=":", lw=0.9); ax2.set_ylabel("$f^2$", labelpad=1); ax2.grid(False); ax2.tick_params(labelsize=6)
    ax[1].set_xlabel("$x$"); ax[1].set_title("designed drifts"); ax[1].legend(frameon=False, loc="lower left", handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    for n, c_, tt, tag in (("designed f=1 (b)", C_FREE, times, r"(b)"), ("designed f^2=2 (c)", C_DATA, 4.0 * times, r"(c), time $\times4$"), ("designed f(x) (d)", C_LEARNED, times, r"(d)")):
        P_ = units[(cfgkey, n, "relax")]["paths"]
        ax[2].plot(tt, progress(P_, "hi"), color=c_, lw=1.0, label=tag)
        ax[2].plot(tt, progress(P_, "lo"), color=c_, lw=0.8, ls="--")
    ax[2].set_xscale("log"); ax[2].set_xlim(0.03, T); ax[2].set_ylim(-0.1, 1.15); ax[2].axhline(0.8, color="0.75", lw=0.5)
    ax[2].set_xlabel("$t$ (rescaled for (c))"); ax[2].set_ylabel("edge progress", labelpad=1); ax[2].set_title("collapse test")
    ax[2].legend(frameon=False, loc="lower right", handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    save(fig, "fig15_volatility.png")

    out = {"N": N, "T": T, "dt": dt, "L": L, "f2": "1/2 + 3x/(2(1+x))", "n_grid": n_grid,
           "W1_start_quantile_discretisation": start_w1,
           "W1_classical_langevin_gamma_2_2": w1_gamma}
    for n in flows:
        k = flows[n]["key"]
        out[f"W1_final_{k}"] = float(w1[n][-1])
        out[f"t_below_0.05_{k}"] = t_thr[n]["0.05"]
        out[f"t_below_0.1_{k}"] = t_thr[n]["0.1"]
    for n, v in stat.items():
        out[f"stationarity_W1_{flows[n]['key']}"] = v
    for n, a in arr.items():
        k = flows[n]["key"]
        out[f"edge_arrival_lower_{k}"] = a["lo"]; out[f"edge_arrival_upper_{k}"] = a["hi"]; out[f"edge_arrival_ratio_lower_over_upper_{k}"] = ratio[n]
    out["times"] = times.tolist()
    for n in flows:
        out["W1_trajectory_" + flows[n]["key"]] = w1[n].tolist()
    report("exp11_volatility", out)


if __name__ == "__main__":
    main()
