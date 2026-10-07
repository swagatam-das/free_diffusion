"""Deterministic particle (quantile) discretisation of the free spectral flow, and its probability-flow reversal.

For the sandwich flow ``dX = b(X) dt + f(X) dS f(X)`` the spectral law ``mu_t`` solves the continuity equation
``d_t mu + d_x (mu V_t) = 0`` with ``V_t = b + f^2 H(f^2 mu_t)`` (Theorem 4.1).  Representing ``mu_t`` by ``M`` equal-mass
particles ``Q_1 < ... < Q_M`` gives the noise-free interacting system

    dQ_k/dt = b(Q_k) + f^2(Q_k) (1/M) sum_{j != k} f^2(Q_j) / (Q_k - Q_j),

which is the N -> infinity limit of the eigenvalue dynamics with the Brownian part removed (the noise is O(N^{-1/2})).

Because ``mu_t`` solves the continuity equation, ``nu_s = mu_{T-s}`` solves it with velocity ``-V_{T-s}``: particles that do
not interact and move with ``-V_{T-s}(y)`` carry a sample of ``mu_T`` back to ``mu_0``.  The field ``V_t`` is taken here from the
forward particle solution (an exact-velocity oracle, the analogue of the exact score used for the free OU flow).  This is a
statement about laws, valid wherever the characteristics are well defined; it is not a reverse-time SDE.
"""

import numpy as np


def _pair_terms(Q, w):
    """Hilbert sum ``(1/M) sum_{j != k} w_j / (Q_k - Q_j)`` for every k, and the stiffness
    ``max_k (1/M) sum_{j != k} w_k w_j / (Q_k - Q_j)^2`` of the interaction."""
    d = Q[:, None] - Q[None, :]
    np.fill_diagonal(d, np.inf)
    H = (w[None, :] / d).sum(axis=1) / Q.size
    stiff = (w[:, None] * w[None, :] / (d * d)).sum(axis=1).max() / Q.size
    return H, stiff


def particle_velocity(Q, b, f2, with_stiffness=False):
    """Velocity of the quantile particles: b(Q_k) + f^2(Q_k) * (1/M) sum_{j != k} f^2(Q_j)/(Q_k - Q_j)."""
    w = f2(Q)
    H, stiff = _pair_terms(Q, w)
    v = b(Q) + w * H
    return (v, stiff) if with_stiffness else v


def forward_flow(Q0, b, f2, T, snap_dt, cfl=0.5, lo=None, hi=None):
    """Heun integration of the particle system with a step ``cfl / stiffness`` chosen from the current configuration.

    The pair interaction is stiff where particles are dense (the largest eigenvalue of its Jacobian is about twice the
    stiffness above), so a fixed step small enough at the start wastes work once the law has spread.  Returns
    ``(times, snaps, n_steps)``; ``snaps[s]`` holds the particles at ``t = s * snap_dt``."""
    Q = np.sort(np.asarray(Q0, dtype=float)).copy()
    S = int(round(T / snap_dt))
    snaps = np.empty((S + 1, Q.size))
    snaps[0] = Q
    n_steps = 0
    for s in range(1, S + 1):
        t = 0.0
        while t < snap_dt - 1e-14:
            v1, stiff = particle_velocity(Q, b, f2, with_stiffness=True)
            h = min(snap_dt - t, cfl / max(stiff, 1e-12))
            v2 = particle_velocity(Q + h * v1, b, f2)
            Q = Q + 0.5 * h * (v1 + v2)
            if lo is not None:
                Q = np.clip(Q, lo, hi)
            t += h
            n_steps += 1
        snaps[s] = Q
    return np.arange(S + 1) * snap_dt, snaps, n_steps


def hilbert_field(y, Q, w):
    """H(f^2 mu)(y) for the discrete measure (1/M) sum_j delta_{Q_j} weighted by w_j = f^2(Q_j).

    Inside the support the principal value is evaluated at the midpoints between neighbouring particles, where the two
    nearest singular terms cancel to leading order, and interpolated; outside it is the plain sum."""
    y = np.asarray(y, dtype=float)
    mid = 0.5 * (Q[1:] + Q[:-1])
    Hm = (w[None, :] / (mid[:, None] - Q[None, :])).sum(axis=1) / Q.size
    out = np.empty_like(y)
    inside = (y >= Q[0]) & (y <= Q[-1])
    out[inside] = np.interp(y[inside], mid, Hm)
    yo = y[~inside]
    if yo.size:
        out[~inside] = (w[None, :] / (yo[:, None] - Q[None, :])).sum(axis=1) / Q.size
    return out


def reverse_probability_flow(Y0, snaps, b, f2, snap_dt, lo=None, hi=None, return_path=False):
    """Move non-interacting particles ``Y0`` (a sample of ``mu_T``) with velocity ``-V_{T-s}`` back to ``t = 0`` (Heun, one step per snapshot)."""
    y = np.sort(np.asarray(Y0, dtype=float)).copy()
    S = snaps.shape[0] - 1

    def V(yy, s):
        Q = snaps[s]
        return b(yy) + f2(yy) * hilbert_field(yy, Q, f2(Q))

    path = [y.copy()] if return_path else None
    for s in range(S, 0, -1):
        v1 = V(y, s)
        v2 = V(y - snap_dt * v1, s - 1)
        y = y - 0.5 * snap_dt * (v1 + v2)
        if lo is not None:
            y = np.clip(y, lo, hi)
        if return_path:
            path.append(y.copy())
    return (y, np.array(path)) if return_path else y
