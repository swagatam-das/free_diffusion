"""Section 10.9 -- generative modelling with a learned score.

Algorithm 1 end to end.  The target is the unitarily invariant matrix law whose
spectrum is the spiked covariance spectrum of Section 10.8.

Training.  Draw alpha ~ Unif[0.02, 0.85] and a GUE matrix, form
X_t = sqrt(alpha) X_0 + sqrt(1-alpha) s, diagonalise X_t = U Lambda U*, and
regress diag(U* (sqrt(alpha) X_0) U) -- the finite-N conditional expectation of
Theorem 6.2 -- on (lambda, alpha).  By Remark 6.3 the optimal denoiser acts
spectrally, so the object learned is a scalar function h(lambda, alpha).

Sampling.  Integrate the reverse-time matrix SDE (7.1) from the GUE by
Euler--Maruyama, re-symmetrising each step.

Three scores drive the same sampler:

    learned         h_theta above, via xi = (lambda - h)/v
    oracle          the exact free score by subordination from mu_0
    coordinatewise  -d/dx log (D_sqrt(alpha) mu_0 * N(0, v)), which is what one
                    obtains by assuming the eigenvalues carry independent
                    Gaussian noise

The coordinatewise score is the generative face of failure mode F1: it is
misspecified at fourth order (Remark 3.2), and the misspecification shows up in
the generated spectrum, not only in a comparison of two convolutions.

Produces fig9_generative.png.
"""

from __future__ import annotations

import numpy as np

from _common import BETA, cfg, parse, report, rng
from freeddpm.forward import EmpiricalLaw
from freeddpm.functionals import w1_samples
from freeddpm.learn import learned_score, make_training_pairs, train_denoiser
from freeddpm.plotting import C_CLASSICAL, C_DATA, C_FREE, C_LEARNED, save, use_style
from freeddpm.reverse import reverse_matrix_sde

import matplotlib.pyplot as plt

from exp08_spiked import spiked_spectrum


def coordinatewise_score_factory(atoms, weights, x_grid):
    """``-d/dx log(D_sqrt(alpha) mu_0 * N(0,v))``, the classical score.

    Returned with the paper's sign convention for ``xi`` (conjugate variable,
    i.e. minus the classical score), so that it can be dropped into the same
    sampler.
    """

    def score(lam, alpha):
        v = 1.0 - alpha
        c = np.sqrt(alpha) * atoms
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        d = lam[:, None] - c[None, :]
        logw = -0.5 * d**2 / v + np.log(weights)[None, :]
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        # d/dx log p = -E_w[(x - c)]/v   =>   xi = -d/dx log p = E_w[(x-c)]/v
        return np.sum(w * d, axis=1) / v

    return score


def main():
    args = parse(__doc__)
    r = rng(args)

    N = cfg(args, 400, 120)
    spectrum = spiked_spectrum(N, 2.0, [6.0, 4.0, 3.0], r)
    law = EmpiricalLaw(spectrum)

    x = np.linspace(spectrum.min() - 2.5, spectrum.max() + 2.5, cfg(args, 900, 400))
    eps = cfg(args, 2e-3, 5e-3)
    sub_kw = dict(damping=cfg(args, 0.35, 0.5), max_iter=cfg(args, 6000, 2000), tol=1e-11)

    # ---------------- training --------------------------------------------
    n_pairs = cfg(args, 168000, 24000)
    ins, tgt = make_training_pairs(spectrum, n_pairs, N, 0.02, 0.85, rng=r)
    net, hist = train_denoiser(
        ins, tgt,
        widths=(2, 64, 64, 1),
        epochs=cfg(args, 240, 60),
        batch_size=512,
        lr=3e-3,
        rng=r,
        verbose=True,
    )

    xi_learned = learned_score(net)

    # ---------------- oracle and coordinatewise scores ---------------------
    alpha_lo, alpha_hi = 0.02, 0.85
    n_steps = cfg(args, 300, 100)
    alphas = np.exp(np.linspace(np.log(alpha_lo), np.log(alpha_hi), n_steps))

    from freeddpm.cauchy import score_from_g, subordination_g

    tables = {}
    omega = None
    zgrid = x + 1j * eps
    for a in alphas:
        g, omega = subordination_g(zgrid, np.sqrt(a) * law.atoms, law.weights,
                                   v=1.0 - a, omega0=omega, return_omega=True,
                                   **sub_kw)
        tables[float(a)] = score_from_g(g)

    def xi_oracle(lam, alpha):
        key = float(alpha)
        if key in tables:
            xi = tables[key]
        else:
            j = min(max(int(np.searchsorted(alphas, alpha)), 1), alphas.size - 1)
            w = (alpha - alphas[j - 1]) / (alphas[j] - alphas[j - 1])
            xi = (1 - w) * tables[float(alphas[j - 1])] + w * tables[float(alphas[j])]
        return np.interp(lam, x, xi)

    xi_coord = coordinatewise_score_factory(law.atoms, law.weights, x)

    # ---------------- sampling --------------------------------------------
    results = {}
    generated = {}
    for name, sc in (("learned", xi_learned), ("oracle", xi_oracle),
                     ("coordinatewise", xi_coord)):
        Y = reverse_matrix_sde(N, alphas, sc, beta=BETA, rng=np.random.default_rng(args.seed + 1))
        ev = np.linalg.eigvalsh(Y)
        generated[name] = ev
        results[f"W1_{name}"] = w1_samples(ev, spectrum)

    # ---------------- figure ----------------------------------------------
    use_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))

    a_show = 0.45
    lam_grid = np.linspace(x[0] + 0.5, x[-1] - 0.5, 400)
    axes[0].plot(lam_grid, xi_oracle(lam_grid, a_show), color=C_FREE, label="exact free score")
    axes[0].plot(lam_grid, xi_learned(lam_grid, a_show), color=C_LEARNED, ls="--",
                 label="learned score")
    axes[0].plot(lam_grid, xi_coord(lam_grid, a_show), color=C_CLASSICAL, ls=":",
                 label="coordinatewise")
    axes[0].set_title(rf"scores at $\alpha_t={a_show}$")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$\xi$")
    axes[0].legend(frameon=False)

    for ax, name in zip(axes[1:], ("learned", "oracle", "coordinatewise")):
        ax.hist(spectrum, bins=45, density=True, color=C_DATA, alpha=0.6, label="data")
        ax.hist(generated[name], bins=45, density=True, histtype="step",
                color=C_FREE, label="generated")
        ax.set_title(f"{name}, $W_1={results['W1_' + name]:.3f}$")
        ax.set_xlabel("$x$")
    axes[1].legend(frameon=False)
    save(fig, "fig9_generative.png")

    report("exp09_generative", {
        "N": N,
        "n_training_pairs": n_pairs,
        "final_training_loss": float(hist[-1]),
        **results,
        "ratio_coordinatewise_to_oracle": results["W1_coordinatewise"] / results["W1_oracle"],
    })


if __name__ == "__main__":
    main()
