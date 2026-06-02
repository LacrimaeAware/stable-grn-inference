# DREAM4 Dynamic Baseline, Calibration, and Fusion

This experiment follows the Size100 scaling result (experiment 10), where the
Size10 sparse candidate `dynamic_lasso_level_include_self_a0_03` did not scale.
It pursues three connected goals on DREAM4 Size10 and Size100 time-series data:

- **A. dynGENIE3-style baseline.** A closer temporal tree baseline using level,
  delta, and derivative targets, alongside the current level GENIE3 baselines.
- **B. Sparsity calibration.** Instead of guessing one alpha, sweep an alpha grid
  for LASSO/Elastic Net (level and delta, include/exclude self) and analyze which
  sparsity level works and whether the best alpha tracks network density.
- **C. Rank fusion.** Fuse complementary evidence (best sparse, best tree,
  correlation) with mean reciprocal rank, Borda, normalized score, and a simple
  reciprocal-direction penalty aimed at the recurring reciprocal false-positive
  problem.

## dynGENIE3 status

No official dynGENIE3 / GENIE3 / arboreto package is importable in this
environment, and no GeneNetWeaver jar is present. The delta/derivative tree
methods here are **dynGENIE3-style**, not an official reproduction. The script
detects an official package if one is later installed (`detect_official_dyngenie3`)
and reports GNW availability (`detect_gnw`). All tree methods use every gene at
time `t` as a predictor (including the target's own past value) but never emit a
self-edge, which matches the dynGENIE3 predictor design.

## Data

Trajectories are split on the `Time` reset; adjacent lagged samples are built
only within a trajectory; `Time` is dropped from predictors.

| Size | Networks | Rows/file | Trajectories | Lagged samples | Candidate edges | True-edge density |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 105 | 5 | 100 | 90 | ~0.15 |
| 100 | 5 | 210 | 10 | 200 | 9900 | ~0.02 |

`level` predicts `G_j(t+1)`; `delta` predicts `G_j(t+1) - G_j(t)`; `derivative`
divides delta by the (constant, 50.0) time step. Because the DREAM4 time grid is
uniform, delta and derivative tree rankings nearly coincide.

## Methods

- Sparse linear: LASSO and Elastic Net (l1=0.7), level and delta targets,
  include-self and exclude-self, swept over alpha `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`.
- dynGENIE3-style trees: `lagged_genie3_{rf,extra_trees}_level`,
  `dyn_genie3_{rf,extra_trees}_delta`, `dyn_genie3_{rf,extra_trees}_derivative`
  (Size10 500 trees, Size100 200 trees).
- `lagged_correlation`.
- Fusion: `mean_reciprocal_rank`, `borda`, `mean_normalized_score`, and
  `reciprocal_penalty` at penalty weights 0.5 and 0.25 (top-5% reciprocal pairs).
  Fusion inputs are the best sparse and best tree method by mean AUPR plus
  correlation; this selection is a documented diagnostic, not a deployable rule.

## Results

Best methods per size (mean across five networks):

| Size | Best AUPR | AUPR | Best AUROC | AUROC |
|---|---|---:|---|---:|
| 10 | `dynamic_lasso_level_include_self_a0_03` | 0.652712 | `dynamic_lasso_level_include_self_a0_03` | 0.821067 |
| 100 | `fusion_borda` | 0.208067 | `lagged_genie3_rf_level` | 0.753913 |

### A. dynGENIE3-style targets

Delta/derivative tree targets **hurt** relative to level on DREAM4 at both sizes:

| Size | level RF AUPR | delta RF AUPR | derivative RF AUPR |
|---|---:|---:|---:|
| 10 | 0.579485 | 0.304290 | 0.312154 |
| 100 | 0.157347 | 0.044727 | 0.045977 |

The dynGENIE3 derivative trick does not help here; the coarse 21-point trajectories
make one-step differences noisy. Level GENIE3 remains the strongest tree variant,
and `lagged_genie3_rf_level` is the Size100 AUROC leader (0.753913).

### B. Sparsity calibration

The best alpha rises with network size, and it does so by pushing predicted edge
density toward the true density. Size100 LASSO level include-self sweep:

| alpha | AUPR | predicted density | true density | nonzero non-self edges | self/non-self ratio |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.033977 | 0.930000 | 0.020687 | 9207 | 2.12 |
| 0.010 | 0.065212 | 0.536263 | 0.020687 | 5309 | 7.31 |
| 0.030 | 0.130486 | 0.254525 | 0.020687 | 2520 | 26.29 |
| **0.100** | **0.161467** | 0.070586 | 0.020687 | 699 | 117.05 |
| 0.300 | 0.112491 | 0.010404 | 0.020687 | 103 | 465.12 |
| 1.000 | 0.020687 | 0.000000 | 0.020687 | 0 | 0.00 |

- Best AUPR alpha is 0.03 at Size10 and 0.1 at Size100 (LASSO level include-self).
- Stronger regularization is better on average at Size100 (mean AUPR 0.067 at
  alpha<=0.03 vs 0.098 at alpha>=0.1).
- The peak alpha (0.1) yields the predicted density closest to the true ~0.021
  among useful operating points, so **alpha behaves like a density knob**: the
  Size10-vs-Size100 difference is largely a density/sample-ratio effect, not a
  property unique to alpha 0.03.
- Include-self beats exclude-self at both sizes even after per-config alpha
  calibration (Size10 +0.166 AUPR; Size100 +0.030 AUPR).
- Self-persistence is dominant and grows fast: self/non-self ratio reaches ~465
  at Size100 high alpha. It reads as a model-stability term, **useful to fit but
  dangerous to interpret as directed regulation** - an oracle-density evaluation
  (top-N-true cutoff) confirms the directed-edge precision stays modest.

### C. Rank fusion

Fusion behavior is regime-dependent:

| Size | Best fusion | AUPR | Best single input | AUPR | Fusion delta |
|---|---|---:|---|---:|---:|
| 10 | `fusion_borda` | 0.613662 | `dynamic_lasso_level_include_self_a0_03` | 0.652712 | -0.039 |
| 100 | `fusion_borda` | 0.208067 | `dynamic_elastic_net_delta_include_self_a0_1_l1_0_7` | 0.172850 | +0.035 |

- At Size10, the single best sparse method already wins; fusion does not help.
- At Size100, **fusion is the best AUPR method overall** (Borda 0.208 vs best
  single 0.173, +0.035 AUPR, +0.08 precision@10 at 0.84, +0.082 AUROC). Combining
  complementary sparse + tree + correlation evidence helps most in the harder,
  sparser regime.
- The reciprocal-direction penalty gives a small but consistent Size100 gain over
  base mean-reciprocal-rank fusion (AUPR 0.194 vs 0.186; reciprocal FP pair rate
  0.988 vs 0.995) and is neutral at Size10. It is a mild, fixed-weight heuristic,
  not a tuned fix.

## Answers to the experiment questions

1. dynGENIE3-style delta/derivative does **not** improve over level GENIE3 (it hurts at both sizes).
2. By AUPR, best sparse edges out best tree at both sizes (narrowly at Size100).
3. By AUROC, sparse wins at Size10 but **trees win at Size100** (RF level 0.754).
4. Topology: Elastic Net include-self variants give the best hub overlap / lowest reciprocal-FP at Size10; at Size100 a normalized-score fusion gives the best top-5 out-hub overlap.
5. Yes - alpha choice largely explains the Size10-vs-Size100 difference (best alpha 0.03 -> 0.1, tracking density).
6. Yes - stronger regularization is consistently better at Size100 on average.
7. Yes - include-self still helps after per-config alpha calibration, at both sizes.
8. Self-persistence is dominant and grows with size/alpha; useful for stability, dangerous for directed-edge claims.
9. Rank fusion helps at Size100 (not at Size10).
10. The reciprocal-direction penalty gives a small Size100 gain and lower reciprocal-FP; neutral at Size10.
11. Best per regime: Size10 `dynamic_lasso_level_include_self_a0_03`; Size100 `fusion_borda` (AUPR) / `lagged_genie3_rf_level` (AUROC).
12. Main claim: dynamic GRN inference on DREAM4 is regime-dependent - the best sparsity tracks density, fusion helps in the hard regime, and no single method wins both AUPR and AUROC; a literature-faithful dynGENIE3 baseline and GNW sweeps are the right next steps.
13. GNW sweeps: see `experiments/12_gnw_sweep_design/gnw_sweep_design.md`.

## Outputs

Under `results/tables/`:

- `dream4_dynamic_baseline_calibration_summary.csv`
- `dream4_dynamic_baseline_calibration_per_network.csv`
- `dream4_dynamic_baseline_calibration_edges.csv` (headline methods, wide)
- `dream4_dynamic_baseline_calibration_topology.csv`
- `dream4_dynamic_baseline_calibration_alpha_sensitivity.csv`
- `dream4_dynamic_baseline_calibration_pairwise_comparisons.csv`
- `dream4_dynamic_baseline_calibration_debug_report.md`

## Run

```powershell
# fast check: Size10 only, reduced trees, reduced alpha grid [0.03, 0.1]
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\11_dream4_dynamic_baseline_and_calibration\run_dynamic_baseline_and_calibration.py --quick

# full: Size10 + Size100, full alpha grid, trees, fusion
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\11_dream4_dynamic_baseline_and_calibration\run_dynamic_baseline_and_calibration.py --standard
```

Other flags: `--skip-size100`, `--skip-trees`, `--skip-fusion`, `--n-jobs`,
`--tree-estimators-size10`, `--tree-estimators-size100`.

## Interpretation Policy

This is a calibration and comparison audit. Alpha is tuned per configuration to
study density behavior, and fusion inputs are chosen by AUPR; both are diagnostic,
not deployable selection rules. The oracle-density evaluation uses the true edge
count and is explicitly not a deployable thresholding method. Deployable claims
need an official dynGENIE3 baseline and GeneNetWeaver simulation sweeps.
