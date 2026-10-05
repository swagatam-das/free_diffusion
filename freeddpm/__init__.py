"""Free diffusion for spectral laws: reference implementation.

Companion code for *Beyond the Semicircle: Free Diffusion Models with Prescribed
Equilibria* and its extended version.  Every figure and every number reported in
Section 10 of the extended paper is produced by a script in ``experiments/``; the
modules here implement the objects the paper defines.

Modules
-------
cauchy       Cauchy transforms, subordination, free additive convolution.
forward      The forward free OU flow, its marginals, the support transition.
matrix       Finite-``N`` Hermitian models: GUE, OU diffusion, Dyson system.
reverse      Probability flow and reverse-time matrix SDE (Algorithm 1).
functionals  Free entropy, free Fisher information, transport distances,
             and the ratios appearing in the functional inequalities.
design       Prescribed equilibria (Theorem 8.11) and Marchenko--Pastur.
learn        Free denoising score matching with a NumPy MLP denoiser.
plotting     Shared figure style.
"""

__version__ = "1.0.0"

from . import cauchy, design, forward, functionals, learn, matrix, reverse  # noqa: F401

__all__ = [
    "cauchy",
    "design",
    "forward",
    "functionals",
    "learn",
    "matrix",
    "reverse",
]
