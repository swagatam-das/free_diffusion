"""Regenerates fig_warmup.png (Figure 1 of the paper), written to ./out/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "legend.fontsize": 6.8})
fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.75))

def two_atom_free_density(x, a, v):
    z = x + 1j * 1e-2
    out = np.zeros_like(z)
    for i, zz in enumerate(z):
        r = np.roots([v**2, -2 * v * zz, zz**2 - a**2 + v, -zz])
        out[i] = r[np.argmin(r.imag)]
    return np.maximum(-out.imag / np.pi, 0)

x = np.linspace(-4, 4, 600)
L = 0.5
a_mp, b_mp = (1 - np.sqrt(L))**2, (1 + np.sqrt(L))**2
mpx = np.linspace(a_mp + 1e-3, b_mp - 1e-3, 400)
mpd = np.sqrt(np.maximum((b_mp - mpx) * (mpx - a_mp), 0)) / (2 * np.pi * L * mpx)
mpx_shift = mpx - 2.0

ax = axes[0]
ax.fill_between(x, two_atom_free_density(x, 1.6, 1.0), color="#7fb3d5", alpha=0.75, label="free convolution")
ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#c0392b", lw=1.6, ls="--", label="classical convolution")
ax.set_title("Classical convolution"); ax.set_xlabel(r"$\mu_0\boxplus\gamma_v$ and $\mu_0*\mathcal{N}(0,v)$")
ax.legend(loc="upper right", framealpha=0.9)

ax = axes[1]
ax.fill_between(x, np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi), color="#27ae60", alpha=0.55, label="reachable (semicircle)")
ax.plot(mpx_shift, mpd * 0.55, color="#7f7f7f", lw=1.4, ls=":", label="target")
ax.set_title("Free OU"); ax.set_xlabel(r"$\mu_0\boxplus\gamma_v\to\gamma_1$")
ax.legend(loc="upper right", framealpha=0.9)

ax = axes[2]
ax.fill_between(mpx_shift, mpd, color="#8e44ad", alpha=0.6, label=r"equilibrium $\mu_*$")
ax.plot(mpx_shift, mpd, color="#8e44ad", lw=1.6)
ax.set_title("Designed equilibrium"); ax.set_xlabel(r"$b_*=-f^2H(f^2\mu_*)$")
ax.legend(loc="upper right", framealpha=0.9)

for ax, top in zip(axes, (0.66, 0.8, 1.25)):
    ax.spines[["top", "right"]].set_visible(False); ax.set_yticks([]); ax.set_xlim(-4, 4); ax.set_ylim(0, top)
plt.tight_layout(pad=0.4, w_pad=0.8)
import os; os.makedirs("out", exist_ok=True)
plt.savefig("out/fig_warmup.png", dpi=300)
print("fig_warmup saved")
