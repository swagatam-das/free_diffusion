"""Regenerates fig_pipeline.png (written to ./out/)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 2.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 2.6)
ax.axis("off")

boxes = [
    (0.4, r"target law $\mu_*$", 2.1),
    (3.0, r"Hilbert transform" "\n" r"$H(f^2\mu_*)$", 2.6),
    (6.1, r"designed drift" "\n" r"$b_*=-f^2H(f^2\mu_*)$", 3.2),
    (9.8, r"diffusion" "\n" r"$dX=b_*\,dt+f\,dS\,f$", 2.8),
]
for x, label, w in boxes:
    box = FancyBboxPatch((x, 0.9), w, 1.15, boxstyle="round,pad=0.06,rounding_size=0.10",
                          linewidth=1.6, edgecolor="#34495e", facecolor="#eaf2f8")
    ax.add_patch(box)
    ax.text(x+w/2, 1.475, label, ha="center", va="center", fontsize=14)
for i in range(len(boxes)-1):
    x, _, w = boxes[i]
    x2, _, _ = boxes[i+1]
    ax.annotate("", xy=(x2-0.05, 1.475), xytext=(x+w+0.05, 1.475),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#34495e"))

ax.text(6.5, 0.35,
        r"Theorem 4.2: the resulting diffusion's equilibrium spectral law is exactly $\mu_*$ again",
        ha="center", va="center", fontsize=13.5, color="#8e44ad")

plt.tight_layout()
import os; os.makedirs("out", exist_ok=True)
plt.savefig("out/fig_pipeline.png", dpi=170)
print("saved")
