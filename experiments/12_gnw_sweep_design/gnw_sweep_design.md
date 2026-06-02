# GeneNetWeaver Simulation Sweep Design

This is a **design scaffold only**. It defines how we would test whether the
DREAM4 findings generalize under controlled GeneNetWeaver (GNW) simulation. It
does not require GNW execution. Experiment 11 prints whether GNW appears
installed; if not, this document stays a plan.

## Goal

DREAM4 gives only five networks per size and a fixed sampling regime. The
Size10 vs Size100 contrast (experiments 9-11) showed that the best sparsity
level and the usefulness of self-persistence depend on network size and density,
and that no single method wins both AUPR and AUROC. Controlled GNW sweeps would
let us vary one factor at a time and check which conclusions are real versus
artifacts of the specific DREAM4 setting.

## GeneNetWeaver Availability

GNW is a Java tool (`gnw-*.jar`) that extracts subnetworks from known organisms
(E. coli, yeast) and simulates expression with ODE/SDE dynamics plus measurement
noise. It is **not** currently installed in this repository:

- No `gnw` executable on `PATH`.
- No `*gnw*.jar` under the repo.
- No Python GNW/dynGENIE3 binding importable.

To enable execution later:

1. Download `gnw-3.1.2.jar` (or current) from the GeneNetWeaver project.
2. Place it under `tools/gnw/` (git-ignored) or set `GNW_JAR`.
3. Generate networks/datasets via the GNW CLI or GUI batch mode.
4. Re-run experiment 11's detector; wire a thin loader that maps GNW output files
   to the existing `load_expression_matrix` / `load_gold_standard_edges` contract
   (tab-delimited expression with a `Time` column for time series; headerless
   directed gold-standard edges).

Until then, this design is the deliverable.

## Sweep Dimensions

| Dimension | Values | Rationale |
|---|---|---|
| Network size (genes) | 10, 30, 50, 100 | Bridge the Size10 -> Size100 gap with intermediate points. |
| Time-series length (points/trajectory) | 21, 50, 100 | Test whether more temporal samples rescue dynamic sparse methods. |
| Number of trajectories | 5, 10, 20 | Vary replication; DREAM4 used 5 (Size10) and 10 (Size100). |
| Noise level | low, medium, high | GNW measurement + dynamics noise; test robustness. |
| Perturbation regime | wildtype/time-series, knockouts, knockdowns, multifactorial | Match DREAM4 regimes; sparse methods looked better on perturbation-rich data. |
| Network density / topology | sparse vs denser subnetworks; modular vs hub-heavy if configurable | Test whether the best alpha tracks true density directly. |

Each sweep cell is one (size, length, trajectories, noise, regime, density)
combination. For tractability, vary one axis at a time around a central anchor
(size 50, length 50, 10 trajectories, medium noise, time-series regime, native
density) rather than a full grid.

## Methods To Run

Reuse the existing implementations so DREAM4 and GNW results are comparable:

- `lagged_correlation` - cheap directional baseline.
- GENIE3 / dynGENIE3-style trees - `lagged_genie3_*_level`, `dyn_genie3_*_delta`,
  `dyn_genie3_*_derivative` (and official dynGENIE3 if installed by then).
- Dynamic sparse calibrated LASSO / Elastic Net - level and delta, include/exclude
  self, swept over the alpha grid `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`,
  with the per-cell best alpha recorded (not fixed in advance).
- Rank fusion - mean reciprocal rank, Borda, mean normalized score, plus the
  reciprocal-direction penalty variant.

## Metrics

For every (cell, method):

- Edge metrics: AUPR, AUROC, precision@k (k sized to the network).
- Topology metrics: out/in-degree Spearman, top-k out/in hub overlap,
  reciprocal false-positive pair rate, reciprocal edge count error, feed-forward
  loop error (vectorized).
- Sparsity diagnostics: predicted vs true edge density, self/non-self coefficient
  ratio, best alpha per cell.
- Oracle-density evaluation: precision/topology at the top-N-true cutoff (marked
  as non-deployable).

## Success Questions

1. **When does dynamic sparse beat tree methods?** Identify the (size, length,
   noise, regime) region, if any, where calibrated sparse AUPR/AUROC overtakes
   GENIE3/dynGENIE3-style.
2. **When does stronger regularization become necessary?** Test whether best
   alpha increases with network size and decreases with trajectory length /
   sample count, i.e. whether alpha tracks the samples-to-edges ratio.
3. **Does self-persistence help or hurt as time-series length increases?** Track
   the self/non-self ratio and include-vs-exclude gap as length grows from 21 to
   100; persistence may be an artifact of short, coarse trajectories.
4. **Does rank fusion improve robustness?** Test whether fusion (and the
   reciprocal penalty) reduces variance across cells and improves worst-case
   topology, even when it does not win mean AUPR.
5. **Do topology metrics agree with edge metrics?** Check whether AUPR/AUROC
   winners are also hub/degree/reciprocal winners, or whether the disagreement
   seen on DREAM4 persists under controlled simulation.

## Output Plan (when executed)

Mirror experiment 11 under `results/tables/`:

- `gnw_sweep_summary.csv` - mean metrics per (cell, method).
- `gnw_sweep_per_cell.csv` - per-cell, per-method rows.
- `gnw_sweep_alpha_sensitivity.csv` - best alpha per cell with density.
- `gnw_sweep_topology.csv` - topology metrics.
- `gnw_sweep_debug_report.md` - answers to the five success questions.

## Status

Design only. No GNW run is required to proceed; the immediate next concrete step
remains a literature-faithful (official) dynGENIE3 comparison plus honest
reporting of the Size100 negative scaling result. GNW sweeps become the priority
once an official dynamic baseline is in place and GNW is installed.
