"""Reverse sampling of a designed flow (Section 6 of the paper).

Prior: Marchenko--Pastur (L = 1/2), the stationary law of the designed flow.  Data: the spectrum of
diag(0.7, 2.1) + sqrt(0.03) * GUE, two separated bumps.  The designed flow carries the data law to the prior; it is run
forward as a noise-free particle system (``freeddpm.particles``), which gives the spectral law ``mu_t`` at every time.  A fresh
sample of the prior is then moved back to time 0 by non-interacting particles with velocity ``-V_{T-s}``, where
``V_t = b + f^2 H(f^2 mu_t)`` is the exact velocity of the forward flow.  Two flows with the same stationary law are compared:
``f = 1`` and ``f^2(x) = 1/2 + 3x/(2(1+x))``.  The velocity is computed from the forward solution, not learned, and the
reversal is deterministic (a statement about laws, not a reverse-time SDE).
"""

import numpy as np

from _common import cache_load, cache_save, cfg, out_of_time, parse, report
from freeddpm.design import design_drift, mp_density, mp_potential_derivative
from freeddpm.forward import EmpiricalLaw
from freeddpm.functionals import w1_density_vs_samples, w1_samples
from freeddpm.matrix import hermitian_brownian_increment
from freeddpm.particles import forward_flow, reverse_probability_flow
from freeddpm.plotting import C_CLASSICAL, C_DATA, C_FREE, C_LEARNED, save, use_style

import matplotlib.pyplot as plt

L = 0.5
ATOMS, V0 = (0.7, 2.1), 0.03
LO, HI = 0.02, 3.8
f2_var = lambda z: 0.5 + 1.5 * z / (1.0 + z)
f2_one = lambda z: np.ones_like(z)


def data_spectrum(n, rng):
    X = np.diag(np.repeat(ATOMS, n // 2)).astype(complex) + np.sqrt(V0) * hermitian_brownian_increment(n, 1.0, rng) / np.sqrt(n)
    return np.linalg.eigvalsh((X + X.conj().T) / 2)


def data_quantiles(M):
    """Equal-mass quantile points of the exact data law (atoms boxplus semicircle).  Evenly spread in mass, unlike a random
    eigenvalue sample, whose near-coincident pairs make the noise-free particle system stiff."""
    a = 1.0 - V0                                    # density(x, a) is the law of D_{sqrt a}(atoms') boxplus gamma_{1-a}
    law = EmpiricalLaw(np.array(ATOMS) / np.sqrt(a), np.array([0.5, 0.5]))
    x = np.linspace(0.0, 3.2, 64001)
    rho = np.clip(law.density(x, a, eps=1e-4), 0.0, None)
    cdf = np.cumsum(rho) * (x[1] - x[0]); cdf /= cdf[-1]
    return np.interp((np.arange(M) + 0.5) / M, cdf, x)


def prior_sample(n, rng):
    T = int(round(n / L))
    Z = (rng.standard_normal((n, T)) + 1j * rng.standard_normal((n, T))) / np.sqrt(2)
    return np.linalg.eigvalsh(Z @ Z.conj().T / T)


def w1_quantile(a, b, m=4000):
    u = (np.arange(m) + 0.5) / m
    qa = np.interp(u, (np.arange(a.size) + 0.5) / a.size, np.sort(a))
    qb = np.interp(u, (np.arange(b.size) + 0.5) / b.size, np.sort(b))
    return float(np.mean(np.abs(qa - qb)))


def main():
    args = parse(__doc__, budget=dict(type=float, default=0.0, help="stop after this many seconds (resume by re-running)"))
    M, N = cfg(args, 800, 200), cfg(args, 400, 100)
    T, snap_dt = cfg(args, 5.0, 2.0), cfg(args, 5e-3, 1e-2)
    R = cfg(args, 5, 2)

    x = np.linspace(LO, HI, cfg(args, 3000, 900)); psi = mp_density(x, L)
    bgrid = design_drift(x, psi, f=np.sqrt(f2_var(x)))
    classes = {"f=1": (lambda z: -0.5 * mp_potential_derivative(z, L), f2_one, C_FREE),
               "f(x)": (lambda z: np.interp(z, x, bgrid), f2_var, C_LEARNED)}
    xg = np.linspace(-1.0, 4.5, 6000); pg = mp_density(xg, L)

    cfgkey = f"M{M}_N{N}_T{T}_sd{snap_dt}_R{R}_s{args.seed}" + ("_q" if args.quick else "")
    units = {k: v for k, v in cache_load("exp12").items() if k[0] == cfgkey}
    Q0 = data_quantiles(M)
    for name, (b, f2, _) in classes.items():
        key = (cfgkey, "fwd", name)
        if key not in units:
            if out_of_time(args):
                cache_save("exp12", units); print("budget reached; re-run to continue"); raise SystemExit(3)
            times, snaps, n_steps = forward_flow(Q0, b, f2, T, snap_dt, lo=LO, hi=HI)
            units[key] = dict(times=times, snaps=snaps, n_steps=n_steps)
            print(f"  forward flow {name}: {n_steps} adaptive steps, W1(mu_T, Marchenko-Pastur) = {w1_density_vs_samples(xg, pg, snaps[-1]):.4f}", flush=True)
            cache_save("exp12", units)

    res = {n: [] for n in classes}; start, floor = [], []
    track = {}
    for r in range(R):
        rng = np.random.default_rng(args.seed + 100 + r)
        noise = np.clip(prior_sample(N, rng), LO, HI); ref = data_spectrum(N, rng); ref2 = data_spectrum(N, rng)
        start.append(w1_samples(noise, ref)); floor.append(w1_samples(ref, ref2))
        for name, (b, f2, _) in classes.items():
            snaps = units[(cfgkey, "fwd", name)]["snaps"]
            if r == 0:
                y, path = reverse_probability_flow(noise, snaps, b, f2, snap_dt, lo=LO, hi=HI, return_path=True)
                S = snaps.shape[0] - 1
                track[name] = [w1_quantile(path[i], snaps[S - i]) for i in range(path.shape[0])]
                final0 = (y, ref, noise)
                final0 = final0 if name == "f=1" else final0
                units[(cfgkey, "y0", name)] = y
            else:
                y = reverse_probability_flow(noise, snaps, b, f2, snap_dt, lo=LO, hi=HI)
            res[name].append(w1_samples(y, ref))
        if r == 0:
            units[(cfgkey, "ref0")] = ref; units[(cfgkey, "noise0")] = noise
    cache_save("exp12", units)

    ref, noise = units[(cfgkey, "ref0")], units[(cfgkey, "noise0")]
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.1))
    bins = np.linspace(0, 3.2, 41)
    ax[0].hist(ref, bins=bins, density=True, color="0.82", label="data")
    ax[0].hist(noise, bins=bins, density=True, histtype="step", color=C_CLASSICAL, ls="--", label="prior sample")
    for name, (_, _, c) in classes.items():
        ax[0].hist(units[(cfgkey, "y0", name)], bins=bins, density=True, histtype="step", color=c, lw=1.4, label=f"generated, {name}")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=7)
    S = units[(cfgkey, "fwd", "f=1")]["snaps"].shape[0] - 1
    s_axis = np.arange(S + 1) * snap_dt
    for name, (_, _, c) in classes.items():
        ax[1].semilogy(s_axis, np.maximum(track[name], 1e-4), color=c, label=name)
    ax[1].axhline(np.mean(floor), color="k", ls=":", lw=1, label="two data samples")
    ax[1].set_xlabel("reverse time $s$"); ax[1].set_ylabel(r"$W_1(Y_s,\mu_{T-s})$"); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "fig17_designed_reverse.png")

    out = {"M_forward_particles": M, "N_generated": N, "T": T, "snapshot_interval": snap_dt, "repeats": R, "atoms": list(ATOMS), "v0": V0, "L": L,
           "W1_prior_to_data_mean": float(np.mean(start)), "W1_two_data_samples_mean": float(np.mean(floor)), "W1_two_data_samples_std": float(np.std(floor))}
    for name in classes:
        out[f"W1_generated_to_data_{name}_mean"] = float(np.mean(res[name])); out[f"W1_generated_to_data_{name}_std"] = float(np.std(res[name]))
        out[f"W1_final_forward_to_prior_{name}"] = w1_density_vs_samples(xg, pg, units[(cfgkey, "fwd", name)]["snaps"][-1])
        out[f"forward_steps_{name}"] = int(units[(cfgkey, "fwd", name)]["n_steps"])
        out[f"tracking_W1_max_{name}"] = float(np.max(track[name])); out[f"tracking_W1_final_{name}"] = float(track[name][-1])
    report("exp12_designed_reverse", out)


if __name__ == "__main__":
    main()
