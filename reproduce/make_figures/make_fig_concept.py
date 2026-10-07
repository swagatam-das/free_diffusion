"""Regenerates fig_concept.png (Figure 3 of the paper), written to ./out/.  Needs results/exp11_volatility.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from freeddpm.design import mp_density
from freeddpm.forward import EmpiricalLaw

plt.rcParams.update({"font.size": 6.8, "axes.titlesize": 7.2, "axes.labelsize": 6.8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6})
d = json.load(open(os.path.join(ROOT, "results", "exp11_volatility.json")))
t = np.array(d["times"])
fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.38), gridspec_kw=dict(width_ratios=[1.05, 1]))
x = np.linspace(-3.2, 3.2, 3000)
v = 0.3; a = 1 - v
two = np.clip(EmpiricalLaw(np.array([-1.6, 1.6]) / np.sqrt(a), np.array([0.5, 0.5])).density(x, a, eps=1e-4), 0, None)
cols = {"sc": "#27ae60", "mp": "#2c5aa0", "two": "#c0392b"}
ax[0].fill_between(x, np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi), color=cols["sc"], alpha=0.35, lw=0)
ax[0].plot(x, np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi), color=cols["sc"], lw=1.1, label="linear drift")
ax[0].plot(x, mp_density(x, 0.5), color=cols["mp"], lw=1.1, label="convex potential")
ax[0].plot(x, two, color=cols["two"], lw=1.1, label="double well")
ax[0].set_ylim(0, 1.45); ax[0].set_yticks([]); ax[0].set_xlabel("$x$"); ax[0].legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0.1, labelspacing=0.2)
ax[0].set_title("drift $b$: which equilibrium")
for key, lab, c in (("b", r"$f\equiv1$", "#2c5aa0"), ("c", r"$f^2\equiv2$", "0.45"), ("d", r"$f(x)$ non-constant", "#1a8a5a")):
    ax[1].semilogy(t, d[f"W1_trajectory_{key}"], color=c, lw=1.1, label=lab)
ax[1].set_xlim(0, 6); ax[1].set_xlabel("$t$"); ax[1].set_ylabel(r"$W_1(\mu_t,\mu_*)$", labelpad=1)
ax[1].legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0.1, labelspacing=0.2)
ax[1].set_title("volatility $f$: which route")
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.3, w_pad=0.6)
os.makedirs("out", exist_ok=True)
plt.savefig("out/fig_concept.png", dpi=300)
print("fig_concept saved")
