# Experiment 33: dynamical operator vs established lagged baselines

The corrective the methodology audit of exps 30-32 demanded. Those experiments compared the
dynamical operator only to a symmetric static correlation, which cannot orient an edge by
construction, so "beats static" was uninformative. This experiment benchmarks the operator against
the established orientable methods.

## Question / hypothesis

Does the dynamical operator (least-squares VAR(1) / DMD) match or beat the established lagged GRN
methods (lagged GENIE3 random forest, lagged LASSO, lagged correlation) at directed edge recovery
on the same time-ordered data and the same ground truth?

Prior (from exp 7 and the literature): probably not. Exp 7 already scored lagged GENIE3 RF 0.53,
lagged LASSO 0.51, lagged correlation 0.46 on DREAM4 Size10, while exp 30 scored the operator at
0.37 on the identical networks. The literature pass found deep/nonlinear methods do not beat simple
baselines on this kind of data. So the expected result is that the operator ranks below lagged
GENIE3 / LASSO, and the honest conclusion is that the time axis enabling orientation is the point,
not the operator itself.

## Method

All methods are graded with one directed AUPR (source -> target edge space) on the SAME pairs and
the SAME truth, so the comparison is fair:
- pairs: DREAM4 lagged samples (source at t, target at t+1, within trajectory) and BoolODE
  pseudotime-ordered consecutive cells (pooled within each pseudotime branch).
- methods: the dynamical operator (`dmd_operator`, converted to source->target edges), lagged
  GENIE3 random forest, lagged LASSO (alpha 0.1), lagged correlation (the established orientable
  baselines), and static correlation (symmetric, kept only as a lower bound).
- truth: DREAM4 directed gold standard; BoolODE exact generating network.

Tooling reused: `src/stable_grn_inference/inference/lagged.py` (the lagged baselines, already in
the repo from exp 7-10), `src/stable_grn_inference/dynamics/temporal.py` (the operator),
`operator_edges` for orientation conversion.

## Data

- DREAM4 Size10 time-series: 5 networks, 10 genes each, 5 trajectories x 21 timepoints, exact
  directed gold standard. The identical data exp 7 used.
- BoolODE single-cell: 6 topologies (linear, long-linear, cycle, bifurcating,
  bifurcating-converging, trifurcating) at 2,000 cells, exact generating networks, pseudotime axis.

## Result

Directed AUPR, mean, all methods on the same pairs and the same truth:

| dataset | DMD operator | lagged GENIE3 RF | lagged LASSO | lagged correlation | static (lower bound) | chance |
| --- | --- | --- | --- | --- | --- | --- |
| DREAM4 Size10 time-series (5 networks) | 0.370 | 0.536 | 0.510 | 0.458 | 0.319 | 0.158 |
| BoolODE single-cell, 2000 cells (18 datasets) | 0.408 | 0.401 | 0.448 | 0.395 | 0.359 | 0.189 |

- DREAM4: the operator ranks LAST of the four orientable methods (0.37 vs lagged GENIE3 0.54,
  LASSO 0.51, correlation 0.46). Its earlier "win" in exp 30 was entirely an artifact of comparing
  only to the symmetric static baseline (0.32), which cannot orient.
- BoolODE: the operator is mid-pack, 2nd of four (0.408). It beats lagged GENIE3 RF (0.401) and
  lagged correlation (0.395) but loses to lagged LASSO (0.448). Competitive on single-cell
  pseudotime, but not the best.
- Net: there is no benchmarked win. Established lagged methods match or beat the operator on both
  datasets. The only robust point is the already-known one: time order enables orientation.

## Decision rule (realized)

The operator does not match the best lagged method on either dataset (last on DREAM4, 2nd on
BoolODE). Direction B therefore has no benchmarked positive. The honest contribution of the whole
dynamical line is the exp 28 separability diagnostic plus the regime-ladder framing, not a new
method. Any future dynamical claim must clear lagged GENIE3 / LASSO on the same data, not chance or
static correlation.

## Outputs

Under `results/tables/` (git-ignored): `dynamical_baseline_benchmark_dream4.csv`,
`dynamical_baseline_benchmark_boolode.csv`, `dynamical_baseline_benchmark_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/33_dynamical_baseline_benchmark/run_benchmark.py
.\.venv\Scripts\python.exe -B experiments/33_dynamical_baseline_benchmark/run_benchmark.py --quick
```
