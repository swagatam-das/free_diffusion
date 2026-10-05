# Provenance of the committed figures and results

`results/*.json` carries a timestamp for every run.  All nine experiments are
committed at their full settings (Section 10 of the extended paper).

| Experiment | Paper figures | Committed settings |
|---|---|---|
| `exp01_forward` | 1, 2, 3 | full: N up to 800, 12 realisations per point |
| `exp02_transition` | 4 | full |
| `exp03_inequalities` | 5 | full: 6001-point grid, 60 schedule levels |
| `exp04_free_vs_classical` | 6 | full: N = 2500 (N = 3000 for the moment check) |
| `exp05_transfer` | 7 | full: N = 50, 200, 600; 400 reverse steps |
| `exp06_design_mp` | (no paper figure) | full: N = 400, T = 8, dt = 1e-3, 4000-point grid |
| `exp07_reverse` | 8 | full: 6e4 particles, 600 midpoint steps |
| `exp08_spiked` | 9, 10 | full: N = 600, 300 score-table levels |
| `exp09_generative` | 11 | full: N = 400, 168 000 training pairs |

The figures `fig1`, `fig4`-`fig13` are byte-identical to the image files used in
the papers.

## Corrections to an earlier version of this file

An earlier version listed `exp06`, the reverse half of `exp08`, and `exp09` as
`--quick` runs.  Checked as follows:

* **`exp06` was a quick run, and has been regenerated at full settings.**  The
  previously committed numbers were reproduced bit for bit by
  `python experiments/exp06_design_mp.py --quick` (N = 120, T = 3, dt = 4e-3,
  1200-point grid).  The full run (N = 400, T = 8, dt = 1e-3, 4000-point grid)
  gives:

  | quantity | quick (previously committed) | full (committed now) |
  |---|---|---|
  | stationarity relative error, interior | 3.49e-3 | 9.39e-4 |
  | edge exponents (lower / upper; theory 0.5) | 0.416 / 0.512 | 0.432 / 0.511 |
  | terminal `W1` to Marchenko-Pastur | 0.0178 | 0.00353 |
  | empirical support (theory [0.0858, 2.914]) | [0.0935, 2.709] | [0.0926, 2.858] |

  The stationarity error and the edge exponents are properties of the analytic
  Marchenko-Pastur density on the quadrature grid, so they depend on the grid
  size (`n_grid`), not on N; the matrix run feeds only the terminal `W1` and the
  empirical support.  The full run was executed in time-boxed segments that
  checkpoint the matrix and the random-number-generator state (the integrator was
  otherwise unchanged), so the result equals that of an uninterrupted run.

* **`exp08` and `exp09` were already full runs.**  Their committed JSON records
  the settings actually used: `N = 600` and `score_table_levels = 300` for
  `exp08`, `N = 400` and `n_training_pairs = 168000` for `exp09`.  These are the
  full-mode values (quick mode gives 200 and 12, and 120 and 24 000).  They were
  not re-run when this file was corrected.

## Notes

`exp08` is the slowest experiment: it solves the subordination fixed point at
every one of 300 schedule levels.  The iteration is warm-started from the
previous level (`subordination_g(..., omega0=omega, return_omega=True)`), which
cuts the iteration count by roughly an order of magnitude; without it the run is
not practical on a single core.

The checks in `reproduce/` (three-atom gap closing, real S&P 500 spectrum) are
independent of these experiments and write their logs to `reproduce/logs/`.
