"""Shared figure style, so that every script produces consistent output."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.environ.get("FREEDDPM_FIGDIR", "figures")

STYLE = {
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
}

C_FREE = "#1f4e9c"
C_DATA = "#8a8a8a"
C_CLASSICAL = "#c0392b"
C_LEARNED = "#1a8a5a"


def use_style():
    plt.rcParams.update(STYLE)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path
