"""Gap-closing theorem (paper, Theorem 4.8) on the three-atom law
(-2.3, 0.4, 1.9) with weights (0.5, 0.2, 0.3): predicted thresholds
alpha_k* = T_k / (1 + T_k),  T_k = min over gap k of Theta(u) = sum w_i (u - y_i)^-2,
compared with component counting of the numerically computed support of
D_{sqrt(alpha)} mu0  boxplus  gamma_{1-alpha}."""
import numpy as np
from scipy.optimize import minimize_scalar
from common import free_conv_semicircle_density

y = np.array([-2.3, 0.4, 1.9]); w = np.array([0.5, 0.2, 0.3])
Theta = lambda u: np.sum(w / (u - y) ** 2)
th = []
for k, (lo, hi) in enumerate([(y[0], y[1]), (y[1], y[2])], 1):
    r = minimize_scalar(Theta, bounds=(lo + 1e-6, hi - 1e-6), method="bounded", options={"xatol": 1e-10})
    T = r.fun; th.append(T / (1 + T))
    print(f"gap {k}: T = {T:.4f} at u = {r.x:.3f}  ->  alpha* = {T/(1+T):.4f}   (paper: T1=0.4018, T2=0.9201; alpha*=0.2866, 0.4792)")

grid = np.linspace(-6, 6, 6001)   # fine grid: a just-opened gap is very narrow
print("\nalpha  components of {psi > 1e-3}")
prev = None
for alpha in np.arange(0.05, 1.0, 0.01):
    dens = free_conv_semicircle_density(grid, np.sqrt(alpha) * y, w, 1 - alpha)
    on = dens > 1e-3
    ncomp = int(np.sum(on[1:] & ~on[:-1]) + on[0])
    if ncomp != prev:
        print(f"{alpha:5.2f}  {ncomp}"); prev = ncomp
print("(a change of count between listed alphas marks a transition; paper: 1 component up to 0.29, 2 for 0.30-0.48, 3 from 0.49)")
