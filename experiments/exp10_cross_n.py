"""Section 10.11: a learned score transferred across matrix sizes.

The free score is a scalar function of (lambda, alpha), so a denoiser trained at one matrix size N_train can in principle drive
the reverse sampler at any other size.  Here the data law is the SAME at every N (two N-independent atomic laws, so a finite-N
spectrum is just the atoms with multiplicities proportional to the weights), and each has an exact oracle score; any change
in W1 with N is therefore a transfer effect, not a change of target.

For each law, denoisers are trained by free denoising score matching at N_train in {100, 400} and used, without retraining, to
generate spectra at N in {50, 100, 200, 400, 800}.  Reported: W1 between the generated spectrum and the exact free marginal at
the stopping time (the sampler stops at alpha = 0.9, so the target is mu_{alpha_end}, not the atoms), for the learned scores
and for the exact (oracle) score.  All scores use the same random numbers at a given N (common random numbers).
"""

import numpy as np

from _common import cache_load, cache_save, cfg, out_of_time, parse, report
from freeddpm.forward import EmpiricalLaw, TwoAtomLaw
from freeddpm.functionals import w1_density_vs_samples, w1_samples
from freeddpm.learn import MLP, learned_score, make_training_pairs, train_denoiser
from freeddpm.plotting import C_FREE, C_LEARNED, save, use_style
from freeddpm.reverse import reverse_matrix_sde

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

A_LO, A_HI = 0.02, 0.9
LAWS = {
    "two-atom": dict(atoms=np.array([-1.6, 1.6]), weights=np.array([0.5, 0.5])),
    "three-atom": dict(atoms=np.array([-2.3, 0.4, 1.9]), weights=np.array([0.5, 0.2, 0.3])),
}


def spectrum(law, N):
    counts = np.round(law["weights"] * N).astype(int)
    assert counts.sum() == N, "N must be compatible with the weights"
    return np.repeat(law["atoms"], counts)


def exact_law(name):
    law = LAWS[name]
    if name == "two-atom":
        return TwoAtomLaw(float(law["atoms"][1])), {}
    return EmpiricalLaw(law["atoms"], law["weights"]), dict(eps=2e-3)


def main():
    args = parse(__doc__,
                 budget=dict(type=float, default=0.0, help="stop after this many seconds (resume by re-running)"),
                 sizes=dict(type=str, default=None, help="comma-separated test sizes"),
                 trains=dict(type=str, default=None, help="comma-separated training sizes"),
                 laws=dict(type=str, default="two-atom,three-atom"),
                 report_only=dict(action="store_true", help="build the report from the cache without computing anything new"),
                 metrics=dict(action="store_true", help="also record KS distance, moments and support endpoints for every draw (same seeds)"),
                 extra_draws=dict(type=int, default=0, help="additional noise draws for sizes >= --extra-from"),
                 extra_from=dict(type=int, default=400))
    sizes = [int(s) for s in (args.sizes or cfg(args, "50,100,200,400,800", "50,100")).split(",")]
    trains = [int(s) for s in (args.trains or cfg(args, "100,400", "50,100")).split(",")]
    laws = args.laws.split(",")
    n_pairs, epochs = cfg(args, 60000, 8000), cfg(args, 100, 30)
    K = cfg(args, 300, 100)
    alphas = np.exp(np.linspace(np.log(A_LO), np.log(A_HI), K))

    reps_for = lambda N: (1 if args.quick else {50: 8, 100: 8, 200: 4}.get(N, 1))     # more noise draws where runs are cheap
    cfgkey = f"K{K}_p{n_pairs}_e{epochs}_s{args.seed}" + ("_q" if args.quick else "")
    units = {k: v for k, v in cache_load("exp10").items() if k[0] == cfgkey}

    def checkpoint():
        if out_of_time(args):
            cache_save("exp10", units)
            print(f"budget reached; {len(units)} units cached; re-run to continue", flush=True)
            raise SystemExit(3)

    # ---- exact oracles: score tables on a grid, and the exact marginal at the stopping time -----------------------
    oracle, marg, exact_stats = {}, {}, {}
    for name in ([] if args.report_only else laws):
        law = LAWS[name]; ex, kw = exact_law(name)
        x = np.linspace(law["atoms"].min() - 3.0, law["atoms"].max() + 3.0, 1200)
        tables = {float(a): ex.score(x, a, **kw) for a in alphas}
        def xi(lam, alpha, x=x, tables=tables):
            key = float(alpha)
            if key not in tables:
                j = min(max(int(np.searchsorted(alphas, alpha)), 1), alphas.size - 1)
                w = (alpha - alphas[j - 1]) / (alphas[j] - alphas[j - 1])
                return np.interp(lam, x, (1 - w) * tables[float(alphas[j - 1])] + w * tables[float(alphas[j])])
            return np.interp(lam, x, tables[key])
        oracle[name] = xi
        marg[name] = (x, np.clip(ex.density(x, A_HI, **kw), 0.0, None))
        dens = marg[name][1]; dx = x[1] - x[0]; w = dens * dx; w = w / w.sum()
        edge = np.where(dens > 1e-2)[0]
        exact_stats[name] = dict(cdf=np.cumsum(w), x=x, mom=[float(np.sum(w * x**k)) for k in (1, 2, 4)], lo=float(x[edge[0]]), hi=float(x[edge[-1]]))

    # ---- training ------------------------------------------------------------------------------------------------
    nets = {}
    for li, name in enumerate([] if args.report_only else laws):
        for Ntr in trains:
            key = (cfgkey, "net", name, Ntr)
            if key not in units:
                checkpoint()
                r = np.random.default_rng(args.seed + 100 * li + Ntr)
                ins, tgt = make_training_pairs(spectrum(LAWS[name], Ntr), n_pairs, Ntr, A_LO, A_HI, rng=r)
                net, hist = train_denoiser(ins, tgt, widths=(2, 64, 64, 1), epochs=epochs, batch_size=512, lr=3e-3, rng=r)
                units[key] = dict(theta=net.get_flat(), x_mean=net.x_mean, x_std=net.x_std, loss=float(hist[-1]))
                print(f"  trained {name} at N={Ntr}: final loss {hist[-1]:.5f}", flush=True)
                cache_save("exp10", units)
            u = units[key]
            net = MLP((2, 64, 64, 1), x_mean=u["x_mean"], x_std=u["x_std"]); net.set_flat(u["theta"])
            nets[(name, Ntr)] = learned_score(net)

    # ---- generation ----------------------------------------------------------------------------------------------
    sources = ["oracle"] + [f"learned@{t}" for t in trains]
    for N in ([] if args.report_only else sizes):
        for name in laws:
            for src in sources:
                key = (cfgkey, "gen", name, src, N)
                if key in units:
                    if args.extra_draws and N >= args.extra_from and units[key]["reps"] < 1 + args.extra_draws:
                        u = units[key]
                        vals = u.setdefault("W1_values", [u["W1_marginal"]]); dvals = u.setdefault("W1_data_values", [u["W1_data"]])
                        sc = oracle[name] if src == "oracle" else nets[(name, int(src.split("@")[1]))]
                        xg, pg = marg[name]
                        while len(vals) < 1 + args.extra_draws:
                            checkpoint()
                            Y = reverse_matrix_sde(N, alphas, sc, beta=1.0, rng=np.random.default_rng(args.seed + 1000 + N + 17 * len(vals)))
                            ev = np.linalg.eigvalsh(Y)
                            vals.append(w1_density_vs_samples(xg, pg, ev)); dvals.append(w1_samples(ev, spectrum(LAWS[name], N)))
                            u.update(W1_marginal=float(np.mean(vals)), W1_marginal_std=float(np.std(vals)), W1_data=float(np.mean(dvals)), reps=len(vals))
                            print(f"  {name:10s} {src:12s} N={N:4d}: draw {len(vals)}: mean W1 {u['W1_marginal']:.4f} (+-{u['W1_marginal_std']:.4f})", flush=True)
                            cache_save("exp10", units)
                    continue
                checkpoint()
                sc = oracle[name] if src == "oracle" else nets[(name, int(src.split("@")[1]))]
                xg, pg = marg[name]; w1m, w1d = [], []
                for rep_ in range(reps_for(N)):                      # common random numbers across the three scores
                    Y = reverse_matrix_sde(N, alphas, sc, beta=1.0, rng=np.random.default_rng(args.seed + 1000 + N + 17 * rep_))
                    ev = np.linalg.eigvalsh(Y)
                    w1m.append(w1_density_vs_samples(xg, pg, ev)); w1d.append(w1_samples(ev, spectrum(LAWS[name], N)))
                units[key] = dict(W1_marginal=float(np.mean(w1m)), W1_marginal_std=float(np.std(w1m)), W1_data=float(np.mean(w1d)), reps=len(w1m))
                print(f"  {name:10s} {src:12s} N={N:4d}: W1 to exact marginal {units[key]['W1_marginal']:.4f} (+-{units[key]['W1_marginal_std']:.4f}, {len(w1m)} draws)", flush=True)
                cache_save("exp10", units)
    cache_save("exp10", units)

    def spectrum_metrics(ev, st):
        ev = np.sort(ev); n = ev.size
        F = np.interp(ev, st["x"], st["cdf"]); i = np.arange(1, n + 1)
        ks = float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
        m1, m2, m4 = st["mom"]; var_ex = m2 - m1 ** 2
        return dict(ks=ks, mean=abs(float(ev.mean()) - m1), var=abs(float(ev.var()) - var_ex) / var_ex,
                    m4=abs(float(np.mean(ev ** 4)) - m4) / m4, lo=abs(float(ev[0]) - st["lo"]), hi=abs(float(ev[-1]) - st["hi"]))

    if args.metrics and not args.report_only:
        for N in sizes:
            for name in laws:
                for src in sources:
                    gkey = (cfgkey, "gen", name, src, N); mkey = (cfgkey, "met", name, src, N)
                    if mkey in units or gkey not in units:
                        continue
                    checkpoint()
                    sc = oracle[name] if src == "oracle" else nets[(name, int(src.split("@")[1]))]
                    xg, pg = marg[name]; rows_ = []; w1s = []
                    for rep_ in range(units[gkey]["reps"]):
                        Y = reverse_matrix_sde(N, alphas, sc, beta=1.0, rng=np.random.default_rng(args.seed + 1000 + N + 17 * rep_))
                        ev = np.linalg.eigvalsh(Y)
                        rows_.append(spectrum_metrics(ev, exact_stats[name])); w1s.append(w1_density_vs_samples(xg, pg, ev))
                    units[mkey] = dict(rows=rows_, w1_check=float(np.mean(w1s)))
                    print(f"  metrics {name:10s} {src:12s} N={N:4d}: KS {np.mean([r['ks'] for r in rows_]):.4f}  (W1 recomputed {np.mean(w1s):.4f} vs cached {units[gkey]['W1_marginal']:.4f})", flush=True)
                    cache_save("exp10", units)
        cache_save("exp10", units)

    # ---- report and figure ---------------------------------------------------------------------------------------
    res = {name: {src: {N: units[(cfgkey, "gen", name, src, N)]["W1_marginal"] for N in sizes if (cfgkey, "gen", name, src, N) in units} for src in sources} for name in laws}
    ratio = {name: {src: {N: res[name][src][N] / res[name]["oracle"][N] for N in res[name][src] if N in res[name]["oracle"]} for src in sources if src != "oracle"} for name in laws}
    use_style()
    fig, ax = plt.subplots(1, len(laws), figsize=(5.2 * len(laws), 3.2), squeeze=False)
    sty = {"oracle": ("o-", "#2ca02c", "exact (oracle) score")}
    pal = ["#9467bd", "#8c564b", "#e377c2"]
    for i, t in enumerate(trains):
        sty[f"learned@{t}"] = ("s--", pal[i % 3], f"learned, trained at $N={t}$")
    for a, name in zip(ax[0], laws):
        for src in sources:
            Ns = sorted(res[name][src])
            if not Ns:
                continue
            m, c, lb = sty[src]
            a.loglog(Ns, [res[name][src][N] for N in Ns], m, color=c, label=lb, ms=4)
            if src.startswith("learned"):
                t = int(src.split("@")[1])
                if t in res[name][src]:
                    a.loglog([t], [res[name][src][t]], "o", ms=10, mfc="none", mec=c)       # ring = tested at its training size
        a.set_xlabel("test size $N$"); a.set_title(f"{name} law"); present = sorted({N for src in sources for N in res[name][src]}); a.set_xticks(present); a.set_xticklabels([str(n) for n in present]); a.xaxis.set_minor_formatter(NullFormatter())
    ax[0][0].set_ylabel(r"$W_1$ to the exact marginal"); ax[0][0].legend(frameon=False, fontsize=7)
    save(fig, "fig16_cross_n.png")

    out = {"sizes": sizes, "trains": trains, "alpha_range": [A_LO, A_HI], "steps": K, "n_training_pairs": n_pairs, "epochs": epochs, "seed": args.seed}
    for name in laws:
        out[f"final_loss_{name}"] = {str(t): units[(cfgkey, "net", name, t)]["loss"] for t in trains if (cfgkey, "net", name, t) in units}
        out[f"W1_std_{name}"] = {src: {str(N): units[(cfgkey, "gen", name, src, N)]["W1_marginal_std"] for N in sizes if (cfgkey, "gen", name, src, N) in units} for src in sources}
        out[f"noise_draws_{name}"] = {str(N): units[(cfgkey, "gen", name, "oracle", N)]["reps"] for N in sizes if (cfgkey, "gen", name, "oracle", N) in units}
        out[f"W1_{name}"] = {s: {str(N): v for N, v in d.items()} for s, d in res[name].items()}
        out[f"ratio_to_oracle_{name}"] = {s: {str(N): v for N, v in d.items()} for s, d in ratio[name].items()}
        allr = [v for d in ratio[name].values() for v in d.values()]
        out[f"max_ratio_to_oracle_{name}"] = max(allr) if allr else None
    met = {}
    for name in laws:
        for src in sources:
            for N in sizes:
                mk = (cfgkey, "met", name, src, N)
                if mk in units:
                    rws = units[mk]["rows"]
                    edge = np.array([max(r["lo"], r["hi"]) for r in rws])
                    met.setdefault(name, {}).setdefault(src, {})[str(N)] = dict(
                        draws=len(rws), ks=float(np.mean([r["ks"] for r in rws])), mean_err=float(np.median([r["mean"] for r in rws])),
                        var_err=float(np.median([r["var"] for r in rws])), m4_err=float(np.median([r["m4"] for r in rws])),
                        edge_err=float(np.median(edge)), stray_draws=int((edge > 1.0).sum()),
                        edge_err_max=float(edge.max()))
    if met:
        out["extra_metrics_mean_over_draws"] = met
        out["metrics_w1_reproduces_cached"] = bool(all(abs(units[k]["w1_check"] - units[(k[0], "gen") + k[2:]]["W1_marginal"]) < 1e-9 for k in units if len(k) > 1 and k[1] == "met"))
    report("exp10_cross_n", out)


if __name__ == "__main__":
    main()
