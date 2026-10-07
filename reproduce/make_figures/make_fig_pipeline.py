"""Regenerates fig_pipeline.png (Figure 2 of the paper), written to ./out/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(figsize=(6.6, 0.78))
ax.set_xlim(0, 13); ax.set_ylim(0, 1.5); ax.axis("off")
boxes = [(0.1, r"target law $\mu_*$", 2.4), (3.3, r"Hilbert transform" "\n" r"$H(f^2\mu_*)$", 2.9),
         (7.0, r"designed drift" "\n" r"$b_*=-f^2H(f^2\mu_*)$", 3.2), (10.6, r"diffusion" "\n" r"$\mathrm{d}X=b_*\mathrm{d}t+f\,\mathrm{d}S\,f$", 2.35)]
for x, label, w in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.1, boxstyle="round,pad=0.04,rounding_size=0.12", linewidth=1.0, edgecolor="#34495e", facecolor="#eaf2f8"))
    ax.text(x + w / 2, 0.75, label, ha="center", va="center", fontsize=8)
for i in range(3):
    x, _, w = boxes[i]; x2 = boxes[i + 1][0]
    ax.annotate("", xy=(x2 - 0.05, 0.75), xytext=(x + w + 0.05, 0.75), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#34495e"))
plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
import os; os.makedirs("out", exist_ok=True)
plt.savefig("out/fig_pipeline.png", dpi=300)
print("fig_pipeline saved")
