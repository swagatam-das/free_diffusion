"""Real-data experiment (paper, Section 6 'A real spectral distribution outside the semicircular
family' and Appendix L).  Prints the spectrum statistics quoted in the paper, the best-fit
semicircle W1, and builds the designed drift b* = -H(mu_hat) for a regularised (truncated,
renormalised KDE) target.  With --dynamics, also runs the matrix diffusion from a mismatched
Wishart start and reports how far the isolated market-mode eigenvalue gets (it does not reach it
within this budget -- see the paper, Appendix L)."""
import argparse, numpy as np
from common import gue, trapz
from sp500 import load_spectrum

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None); ap.add_argument("--dynamics", action="store_true")
ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--dt", type=float, default=5e-4)
ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()

eigs, N, T, ntick = load_spectrum(csv=a.csv)
L = N / T; amp, bmp = (1 - np.sqrt(L)) ** 2, (1 + np.sqrt(L)) ** 2
s = np.sort(eigs); bulk = s[:95]
print(f"tickers with complete coverage: {ntick};  N={N}, T={T}, L=N/T={L:.4f}")
print(f"MP predicted support [{amp:.4f}, {bmp:.4f}];  eigenvalue range [{s[0]:.4f}, {s[-1]:.4f}]")
print(f"eigenvalues above 1.05 x MP edge: {int(np.sum(s > 1.05*bmp))};  top-5: {np.round(s[-5:],3)}")
print(f"variance fraction in top eigenvalue: {s[-1]/s.sum():.4f}")
print(f"95 smallest eigenvalues: [{bulk.min():.3f}, {bulk.max():.3f}];  fraction inside MP support: "
      f"{np.mean((bulk >= amp) & (bulk <= bmp)):.2f}")

# best-fit (mean/variance matched) semicircle
m, var = s.mean(), s.var(); r = 2 * np.sqrt(var); q = np.linspace(0.02, 0.98, 400)
xs = np.linspace(m - r, m + r, 20000); psi = np.sqrt(np.maximum(r**2 - (xs - m) ** 2, 0)); psi /= trapz(psi, xs)
c = np.cumsum(psi); c /= c[-1]
print(f"W1(best-fit semicircle, real spectrum) = {trapz(np.abs(np.interp(q, c, xs) - np.quantile(s, q)), q):.4f}")

# designed drift for the regularised target (truncated, renormalised Gaussian KDE)
bw = 0.22; lo, hi = s.min() - 0.3, s.max() + 0.3
xg = np.linspace(lo, hi, 4000)
kde = np.exp(-0.5 * ((xg[:, None] - s[None, :]) / bw) ** 2).sum(1) / (len(s) * bw * np.sqrt(2 * np.pi))
kde = np.maximum(kde, 1e-6); kde /= trapz(kde, xg)
# principal-value Hilbert transform  H mu(x0) = PV int psi(y)/(x0 - y) dy  (symmetric exclusion of x0)
Hk = np.array([trapz(np.where(np.abs(xg - x0) > 1e-9, kde / np.where(np.abs(xg - x0) > 1e-9, x0 - xg, 1.0), 0.0), xg) for x0 in xg])
bstar = -Hk
print(f"designed drift b* = -H(mu_hat) on [{lo:.2f}, {hi:.2f}]: range [{bstar.min():.3f}, {bstar.max():.3f}]")

if a.dynamics:
    kappa = 5.0                                   # simulation-only confinement outside the fitted interval
    def bfun(x):
        out = np.interp(x, xg, bstar)
        out = np.where(x < lo, bstar[0] + kappa * (lo - x), out)
        return np.where(x > hi, bstar[-1] - kappa * (x - hi), out)
    rng = np.random.default_rng(a.seed)
    G = rng.standard_normal((N, 250)); lam0 = np.linalg.eigvalsh(G @ G.T / 250)
    lam0 = lam0 / lam0.mean() * s.mean()
    X = np.diag(lam0).astype(complex)
    for _ in range(a.steps):
        lam, U = np.linalg.eigh(X)
        X = X + (U * bfun(lam)) @ U.conj().T * a.dt + np.sqrt(a.dt) * gue(N, rng)
        X = 0.5 * (X + X.conj().T)
    ev = np.linalg.eigvalsh(X)
    qq = np.linspace(0.02, 0.98, 300)
    thr = np.quantile(s, 0.95)
    print(f"dynamics ({a.steps} steps, dt={a.dt}, seed={a.seed}): top simulated eigenvalue {ev.max():.2f} "
          f"(target market mode {s[-1]:.2f});  W1(simulated, real) = "
          f"{trapz(np.abs(np.quantile(ev, qq) - np.quantile(s, qq)), qq):.3f}")
