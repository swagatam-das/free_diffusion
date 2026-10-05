"""Regenerates fig_warmup.png (written to ./out/)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

def two_atom_free_density(x, a, v):
    z = x + 1j*1e-2
    out = np.zeros_like(z)
    for i, zz in enumerate(z):
        r = np.roots([v**2, -2*v*zz, zz**2-a**2+v, -zz])
        out[i] = r[np.argmin(r.imag)]
    return np.maximum(-out.imag/np.pi, 0)

x = np.linspace(-4, 4, 600)
a0 = 1.6

L = 0.5
a_mp, b_mp = (1-np.sqrt(L))**2, (1+np.sqrt(L))**2
mpx = np.linspace(a_mp+1e-3, b_mp-1e-3, 400)
mpd = np.sqrt(np.maximum((b_mp-mpx)*(mpx-a_mp), 0))/(2*np.pi*L*mpx)
mpx_shift, mpd_shift = mpx - 2.0, mpd

# Panel 1
ax = axes[0]
true_density = two_atom_free_density(x, a0, 1.0)
ax.fill_between(x, true_density, color="#7fb3d5", alpha=0.75, label="true (compact)")
gauss = np.exp(-x**2/2)/np.sqrt(2*np.pi)
ax.plot(x, gauss, color="#c0392b", lw=3, ls="--", label="predicted (unbounded)")
ax.set_title("Coordinatewise\n(wrong limit)", fontsize=16)
ax.set_xlabel(r"$\mu_0 * N(0,v)$ vs. $\mu_0 \boxplus \gamma_v$", fontsize=14)
ax.set_yticks([]); ax.set_xlim(-4, 4)
ax.tick_params(axis='x', labelsize=12)
ax.legend(fontsize=12.5, loc="upper right", framealpha=0.9)

# Panel 2
ax = axes[1]
sc = np.sqrt(np.maximum(4-x**2, 0))/(2*np.pi)
ax.fill_between(x, sc, color="#27ae60", alpha=0.55, label="only reachable shape")
ax.plot(mpx_shift, mpd_shift*0.55, color="#7f7f7f", lw=2.5, ls=":", label="target (unreachable)")
ax.set_title("Constant-coefficient free\n(right limit, one shape only)", fontsize=16)
ax.set_xlabel(r"$\mu_0 \boxplus \gamma_v \to \gamma_1$ always", fontsize=14)
ax.set_yticks([]); ax.set_xlim(-4, 4)
ax.tick_params(axis='x', labelsize=12)
ax.legend(fontsize=12.5, loc="upper right", framealpha=0.9)

# Panel 3
ax = axes[2]
ax.fill_between(mpx_shift, mpd_shift, color="#8e44ad", alpha=0.6, label=r"designed equilibrium $=\mu_*$")
ax.plot(mpx_shift, mpd_shift, color="#8e44ad", lw=3)
ax.set_title("Operator-valued (this paper)\n(any target law)", fontsize=16)
ax.set_xlabel(r"$b_*=-f^2H(f^2\mu_*)\ \Rightarrow\ \mu_*$", fontsize=14)
ax.set_yticks([]); ax.set_xlim(-4, 4)
ax.tick_params(axis='x', labelsize=12)
ax.legend(fontsize=12.5, loc="upper right", framealpha=0.9)

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
import os; os.makedirs("out", exist_ok=True)
plt.savefig("out/fig_warmup.png", dpi=170)
print("saved")
