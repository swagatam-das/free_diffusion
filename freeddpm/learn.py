"""Free denoising score matching (Definition 6.1, Theorem 6.2).

The training objective of the paper is

.. math::
    \\mathcal L(\\theta) = \\int_0^T w(t)\\,
    \\tau\\bigl[(g_\\theta(X_t,t) - \\sqrt{\\alpha_t} X_0)^2\\bigr] dt,

and the estimated free score is
``xi_hat_t = (X_t - g_theta(X_t,t)) / v_t``.

By Remark 6.3 the optimal denoiser of a unitarily invariant law acts
spectrally, so the object to be learned is a *scalar* function
``h(lambda, alpha)`` of an eigenvalue and the noise level, not an
``N^2``-dimensional map.  That is what makes the learned score transfer across
dimensions.

Training data are generated exactly as described in Section 10.9: draw
``alpha ~ Unif[a_lo, a_hi]`` and a GUE matrix, form
``X_t = sqrt(alpha) X_0 + sqrt(1-alpha) s``, diagonalise ``X_t = U Lambda U*``,
and regress ``diag(U* (sqrt(alpha) X_0) U)`` -- the finite-``N`` conditional
expectation appearing in Theorem 6.2 -- on ``(lambda, alpha)``.

The network is a small tanh multilayer perceptron implemented directly in
NumPy with Adam, so the repository has no deep-learning dependency.  Analytic
gradients are used; :func:`check_gradients` verifies them against finite
differences and is exercised by the test suite.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLP", "make_training_pairs", "train_denoiser", "learned_score", "check_gradients"]


class MLP:
    """Tanh multilayer perceptron with a scalar output.

    Parameters are stored as a flat list of ``(W, b)`` pairs.  The input is
    standardised internally using statistics supplied at construction, which
    keeps the tanh units in their responsive range.
    """

    def __init__(self, widths=(2, 64, 64, 1), rng=None, x_mean=None, x_std=None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.widths = tuple(widths)
        self.params = []
        for nin, nout in zip(widths[:-1], widths[1:]):
            scale = np.sqrt(1.0 / nin)
            W = rng.normal(scale=scale, size=(nin, nout))
            b = np.zeros(nout)
            self.params.append([W, b])
        self.x_mean = np.zeros(widths[0]) if x_mean is None else np.asarray(x_mean, float)
        self.x_std = np.ones(widths[0]) if x_std is None else np.asarray(x_std, float)

    # -- forward / backward --------------------------------------------------
    def forward(self, X, cache=False):
        Z = (np.asarray(X, dtype=float) - self.x_mean) / self.x_std
        acts = [Z]
        for i, (W, b) in enumerate(self.params):
            Z = Z @ W + b
            if i < len(self.params) - 1:
                Z = np.tanh(Z)
            acts.append(Z)
        out = Z[:, 0]
        return (out, acts) if cache else out

    def grads(self, X, y, sample_weight=None):
        """Gradient of the mean squared error and the loss value."""
        n = X.shape[0]
        out, acts = self.forward(X, cache=True)
        resid = out - np.asarray(y, dtype=float)
        if sample_weight is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(sample_weight, dtype=float)
            w = w / w.sum()
        loss = float(np.sum(w * resid**2))

        g = (2.0 * w * resid)[:, None]  # dL/d(pre-activation of last layer)
        grads = [None] * len(self.params)
        for i in range(len(self.params) - 1, -1, -1):
            A = acts[i]
            W, b = self.params[i]
            grads[i] = [A.T @ g, g.sum(axis=0)]
            if i > 0:
                g = (g @ W.T) * (1.0 - acts[i] ** 2)
        return loss, grads

    # -- flat parameter view (used by the gradient check) --------------------
    def get_flat(self):
        return np.concatenate([p.ravel() for Wb in self.params for p in Wb])

    def set_flat(self, theta):
        k = 0
        for Wb in self.params:
            for j, p in enumerate(Wb):
                m = p.size
                Wb[j] = theta[k : k + m].reshape(p.shape).copy()
                k += m
        assert k == theta.size


def make_training_pairs(X0_spectrum, n_pairs, N, alpha_lo=0.02, alpha_hi=0.85,
                        rng=None, batch_matrices=None):
    """Generate ``(lambda, alpha)`` inputs and denoiser targets.

    ``X0_spectrum`` is the data spectrum (length ``N``); the data matrix is
    taken unitarily invariant, so ``X_0 = U diag(spectrum) U*`` with ``U`` Haar.
    Since the construction below conjugates by the eigenvectors of ``X_t``, it
    is equivalent and cheaper to take ``X_0`` diagonal.

    Returns ``(inputs, targets)`` with shapes ``(n_pairs, 2)`` and ``(n_pairs,)``.
    """
    from .matrix import gue, symmetrise

    rng = np.random.default_rng(0) if rng is None else rng
    spec = np.asarray(X0_spectrum, dtype=float)
    if spec.size != N:
        raise ValueError("spectrum length must equal N")

    n_batches = int(np.ceil(n_pairs / N)) if batch_matrices is None else batch_matrices
    ins = np.empty((n_batches * N, 2))
    tgt = np.empty(n_batches * N)

    X0 = np.diag(spec).astype(complex)
    for k in range(n_batches):
        alpha = rng.uniform(alpha_lo, alpha_hi)
        s = gue(N, rng)
        Xt = symmetrise(np.sqrt(alpha) * X0 + np.sqrt(1.0 - alpha) * s)
        lam, U = np.linalg.eigh(Xt)
        target = np.real(np.einsum("ij,jk,ki->i", U.conj().T, np.sqrt(alpha) * X0, U))
        ins[k * N : (k + 1) * N, 0] = lam
        ins[k * N : (k + 1) * N, 1] = alpha
        tgt[k * N : (k + 1) * N] = target

    return ins[:n_pairs], tgt[:n_pairs]


def train_denoiser(inputs, targets, widths=(2, 64, 64, 1), epochs=200,
                   batch_size=512, lr=3e-3, rng=None, verbose=False):
    """Fit ``h_theta(lambda, alpha)`` by Adam on the mean squared error."""
    rng = np.random.default_rng(0) if rng is None else rng
    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    net = MLP(widths, rng=rng, x_mean=inputs.mean(axis=0), x_std=inputs.std(axis=0) + 1e-12)

    m = [[np.zeros_like(p) for p in Wb] for Wb in net.params]
    vv = [[np.zeros_like(p) for p in Wb] for Wb in net.params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = inputs.shape[0]
    history = []

    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss, grads = net.grads(inputs[idx], targets[idx])
            step += 1
            for i, gWb in enumerate(grads):
                for j, g in enumerate(gWb):
                    m[i][j] = b1 * m[i][j] + (1 - b1) * g
                    vv[i][j] = b2 * vv[i][j] + (1 - b2) * g * g
                    mhat = m[i][j] / (1 - b1**step)
                    vhat = vv[i][j] / (1 - b2**step)
                    net.params[i][j] -= lr * mhat / (np.sqrt(vhat) + eps)
            ep_loss += loss
            nb += 1
        history.append(ep_loss / nb)
        if verbose and (ep % max(1, epochs // 10) == 0):
            print(f"  epoch {ep:4d}  loss {history[-1]:.6f}")

    return net, history


def learned_score(net):
    """Wrap a trained denoiser as ``xi_hat(lambda, alpha)``.

    ``xi_hat = (lambda - h_theta(lambda, alpha)) / v`` with ``v = 1 - alpha``,
    which is equation (6.3) of the paper.
    """

    def score(lam, alpha):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        a = np.full_like(lam, float(alpha))
        h = net.forward(np.stack([lam, a], axis=1))
        return (lam - h) / (1.0 - float(alpha))

    return score


def check_gradients(seed=0, tol=1e-6):
    """Finite-difference check of the analytic gradients.  Returns the max error."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(17, 2))
    y = rng.normal(size=17)
    net = MLP((2, 7, 5, 1), rng=rng)
    _, grads = net.grads(X, y)
    flat_analytic = np.concatenate([g.ravel() for gWb in grads for g in gWb])

    theta0 = net.get_flat()
    num = np.empty_like(theta0)
    h = 1e-6
    for i in range(theta0.size):
        tp = theta0.copy()
        tp[i] += h
        net.set_flat(tp)
        lp, _ = net.grads(X, y)
        tm = theta0.copy()
        tm[i] -= h
        net.set_flat(tm)
        lm, _ = net.grads(X, y)
        num[i] = (lp - lm) / (2 * h)
    net.set_flat(theta0)
    return float(np.max(np.abs(num - flat_analytic)))
