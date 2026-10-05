# `reproduce/`: checks that `experiments/` does not cover

Two short, self-contained scripts (NumPy, SciPy, pandas; independent of `freeddpm/`) for results
reported in the condensed paper (*Beyond the Semicircle*) that have no counterpart in
`experiments/exp01`-`exp09`.

```bash
bash run_all.sh                  # from this directory, or:  make reproduce   (repository root)
bash run_all.sh --with-dynamics  # also the S&P 500 matrix-diffusion run
```

Logs are written to `logs/`; the committed logs are the runs described below.

| Paper item | Script | Result |
|---|---|---|
| Gap closing on the three-atom law (atoms -2.3, 0.4, 1.9; weights 0.5, 0.2, 0.3), Thm 4.8 | `exp_gap_closing.py` | `T1 = 0.4018`, `T2 = 0.9201`, `alpha* = 0.2866, 0.4792`; the support of `D_{sqrt a} mu0 ⊞ gamma_{1-a}` has 1 component up to 0.29, 2 from 0.30 and 3 from 0.49, as in the paper. Needs a fine x-grid: a just-opened gap is very narrow. |
| Real S&P 500 correlation spectrum | `exp_realdata_sp500.py` | 470 complete tickers, `N=100`, `T=1258`; MP support [0.516, 1.643]; 5 outliers above it; top eigenvalue = 32 % of the variance; the 95 smallest eigenvalues span [0.12, 1.68] with only 47 % inside the MP support; best-fit semicircle `W1 = 2.197`. |
| Designed diffusion on the real target (`--dynamics`) | `exp_realdata_sp500.py` | Seed 7, 8000 steps, `dt = 5e-4`: `W1` over the central quantiles 0.069; largest simulated eigenvalue 3.03, so the two isolated outliers (4.93 and 32.02) are **not** reached. |

The S&P 500 scripts download one public CSV (Plotly's copy of the Kaggle *S&P 500 stock data*,
about 30 MB) into the repository-level `data/` on first use; pass `--csv path/to/all_stocks_5yr.csv`
to use a local copy.  `make_figures/` regenerates three of the paper's figures (`fig_warmup`,
`fig_pipeline`, `fig_realdata`) into `make_figures/out/`; run the scripts from that directory.

## Sign convention (the one that bites)

`H mu(x) = PV ∫ dmu(y)/(x - y)` and the designed drift is `b* = -f^2 H(f^2 mu*)`.  With the opposite
sign the drift is repulsive, the target is not stationary, and the simulated spectrum spreads out
instead of recovering the bulk.  In `freeddpm`, `reverse.stationarity_residual` and `tests/` pin the
same convention.
