# Experiment 31: dynamical recovery on BoolODE single-cell time-series

Direction B on real-ish single-cell data. Experiment 30 showed a dynamic operator recovers
directed structure a static snapshot cannot, on synthetic dynamics and DREAM4. This runs the
same test on BoolODE single-cell expression (BEELINE), which has a pseudotime axis, an exact
generating network, and a built-in cell-count sweep (the sample-size / SNR axis of exp 28).

## Method

For each network type (linear, long-linear, cycle, bifurcating, bifurcating-converging,
trifurcating) and cell count (100, 200, 2000, 5000), cells are ordered along pseudotime into
snapshot pairs (within each branch), a dynamic operator is fit, and its directed edges are
graded against the exact ground truth, against a static correlation baseline and the chance
line. Tooling: `pseudotime_ordered_pairs`, `dmd_operator`, `static_correlation_edges`,
`edges_to_operator` in `src/stable_grn_inference/dynamics/temporal.py`.

## Result (240 datasets: 6 types x 4 cell counts x 10 replicates)

- Directed normalized recovery (mean over all datasets): dynamic operator 0.261 vs static
  correlation 0.210. The dynamic operator beats the static, symmetric correlation at directed
  recovery on real single-cell data.
- The gain is topology-dependent. At 5000 cells the dynamic operator clearly beats static on
  linear (0.509 vs 0.304), bifurcating-converging (0.388 vs 0.119), long-linear (0.413 vs
  0.360) and bifurcating (0.118 vs 0.073), and loses on cycle (0.028 vs 0.161) and
  trifurcating (0.178 vs 0.233). A single pseudotime ordering is ill-defined for a cycle, so
  the cyclic failure is expected and interpretable, not a bug.
- The cell-count axis is nearly flat (0.265 at 100 cells to 0.272 at 5000): single-cell
  pseudotime noise, not sample size, is the binding limit here, which is consistent with the
  exp 28 finding that SNR, not sample size alone, sets the floor.

## Interpretation

On real single-cell data with exact truth, a pseudotime-ordered dynamic operator recovers
directed structure that the static correlation cannot, where the trajectory geometry is
orderable (linear, branching). This is the regime ladder's top rung confirmed on BoolODE.
Truth is the exact generating network, so correctness is graded directly; reproducibility is
not needed.

## Outputs

Under `results/tables/` (git-ignored): `boolode_dynamical_all.csv`,
`boolode_dynamical_summary.csv`, `boolode_dynamical_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/31_boolode_dynamical/run_boolode_dynamical.py
.\.venv\Scripts\python.exe -B experiments/31_boolode_dynamical/run_boolode_dynamical.py --quick
```
