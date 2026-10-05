"""Section 10.2 -- the support transition (Theorem 9.3).

Two checks.

1.  A bisection on the numerically computed density at the gap centre locates
    the critical variance of ``(delta_{-a} + delta_a)/2 boxplus gamma_v``.  For
    ``a = 1.6`` the theory gives ``v* = a^2 = 2.56``; the paper reports
    ``2.5600`` to four significant figures.

2.  A phase diagram over the ``(a, v)`` plane, whose numerically detected
    boundary is compared with the parabola ``v = a^2`` throughout the range,
    so that Theorem 9.3 is tested as an identity in ``a`` and not at a point.

Produces fig12_phase.png.

The detection threshold matters.  The density at the gap centre vanishes
identically below the transition and rises continuously above it, so any
threshold-based detector reports a critical value slightly above the true one.
The bisection therefore refines the threshold as well, and the reported value
is the extrapolation to zero threshold.
"""

from __future__ import annotations

import numpy as np

from _common import cfg, parse, report, rng
from freeddpm.cauchy import density_from_g, g_two_atom_cubic
from freeddpm.forward import critical_variance
from freeddpm.plotting import save, use_style

import matplotlib.pyplot as plt


def gap_density(a, v, eps=1e-7):
    """Density of ``(delta_{-a}+delta_a)/2 boxplus gamma_v`` at the gap centre."""
    g = g_two_atom_cubic(np.array([0.0 + 1j * eps]), a, v)
    return float(density_from_g(g)[0])


def bisect_vstar(a, threshold, lo=0.05, hi=8.0, iters=60, eps=1e-7):
    """Smallest ``v`` at which the density at the origin exceeds ``threshold``."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gap_density(a, mid, eps) > threshold:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    args = parse(__doc__)
    rng(args)
    a = 1.6

    # --- 1. critical variance, extrapolated to zero threshold -------------
    thresholds = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    vals = [bisect_vstar(a, th) for th in thresholds]
    # psi(0) grows linearly in (v - v*) just above the transition, so the
    # detected value is affine in the threshold; extrapolate to zero.
    coef = np.polyfit(thresholds, vals, 1)
    v_extrap = float(coef[-1])

    # --- 2. phase diagram -------------------------------------------------
    n_a = cfg(args, 160, 60)
    n_v = cfg(args, 160, 60)
    a_grid = np.linspace(0.4, 2.4, n_a)
    v_grid = np.linspace(0.05, 6.5, n_v)
    Z = np.empty((n_v, n_a))
    for j, aa in enumerate(a_grid):
        for i, vv in enumerate(v_grid):
            Z[i, j] = gap_density(aa, vv)

    detected = np.array([bisect_vstar(aa, 1e-3, lo=0.02, hi=8.0) for aa in a_grid])
    rel_err = np.abs(detected - a_grid**2) / (a_grid**2)

    use_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.pcolormesh(a_grid, v_grid, Z, shading="auto", cmap="viridis")
    ax.contour(a_grid, v_grid, Z, levels=[1e-3], colors="w", linewidths=1.2)
    ax.plot(a_grid, a_grid**2, "r--", lw=1.4, label=r"$v^{*}=a^{2}$")
    ax.set_xlabel("$a$")
    ax.set_ylabel("$v$")
    ax.set_ylim(v_grid[0], v_grid[-1])
    ax.set_title(r"density at the gap centre of $\frac{1}{2}(\delta_{-a}+\delta_a)\boxplus\gamma_v$")
    ax.legend(loc="lower right", frameon=False)
    fig.colorbar(im, ax=ax, pad=0.02)
    save(fig, "fig12_phase.png")

    report("exp02_transition", {
        "a": a,
        "v_star_theory": critical_variance(a),
        "v_star_threshold_1e-3": bisect_vstar(a, 1e-3),
        "v_star_extrapolated": v_extrap,
        "max_relative_error_on_parabola": float(rel_err.max()),
        "median_relative_error_on_parabola": float(np.median(rel_err)),
    })


if __name__ == "__main__":
    main()
