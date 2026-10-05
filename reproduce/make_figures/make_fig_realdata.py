"""Regenerates fig_realdata.png (real S&P 500 correlation spectrum vs Marchenko-Pastur and the
best-fit semicircle); written to ./out/.  Run from this directory."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from sp500 import load_spectrum

eigs, N, T, _ = load_spectrum(cache_dir="../../data")
L = N / T; a_mp, b_mp = (1 - np.sqrt(L)) ** 2, (1 + np.sqrt(L)) ** 2
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
ax.hist(eigs[eigs < 5], bins=25, density=True, color="#7fb3d5", edgecolor="white", alpha=0.85, label="real spectrum (bulk)")
xg = np.linspace(a_mp * 0.9, b_mp * 1.05, 400); xg = xg[xg > 0]
ax.plot(xg, np.sqrt(np.maximum((b_mp - xg) * (xg - a_mp), 0)) / (2 * np.pi * L * xg), color="#c0392b", lw=2, label="Marchenko--Pastur")
ax.set_xlim(0, 5); ax.set_xlabel("eigenvalue"); ax.set_ylabel("density"); ax.set_title("Bulk vs. Marchenko--Pastur"); ax.legend(fontsize=8)
ax = axes[1]
ax.hist(eigs, bins=40, density=True, color="#7fb3d5", edgecolor="white", alpha=0.85, label="real spectrum (all)")
m, v = eigs.mean(), eigs.var(); r = 2 * np.sqrt(v); xs = np.linspace(m - r, m + r, 400)
ax.plot(xs, np.sqrt(np.maximum(r**2 - (xs - m) ** 2, 0)) / (2 * np.pi * v), color="#27ae60", lw=2, label="best-fit semicircle")
for x in eigs[eigs > 5]: ax.axvline(x, color="#c0392b", lw=1, ls="--", alpha=0.7)
ax.axvline(eigs[eigs > 5][0], color="#c0392b", lw=1, ls="--", alpha=0.7, label="outliers (5)")
ax.set_yscale("log"); ax.set_ylim(1e-4, 2); ax.set_xlabel("eigenvalue"); ax.set_title("Full spectrum: outliers vs. semicircle"); ax.legend(fontsize=8)
plt.tight_layout(); os.makedirs("out", exist_ok=True); plt.savefig("out/fig_realdata.png", dpi=150); print("saved out/fig_realdata.png")
