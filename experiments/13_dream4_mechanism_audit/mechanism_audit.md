# DREAM4 Mechanism Audit

This experiment explains *why* the experiment 9-11 winners work or fail, rather
than adding new models. It tests five mechanism hypotheses on DREAM4 Size10 and
Size100 time-series data, reusing the existing pipeline. Alpha is tuned on the
gold standard only as an **oracle diagnostic**; density-matched and best-alpha
analyses are labeled as such and are not deployable selection rules.

## Data

Within-trajectory adjacent lagged samples (Size10: 5 trajectories, 100 samples,
90 candidate edges, true density ~0.16; Size100: 10 trajectories, 200 samples,
9900 candidate edges, true density ~0.02). Targets: `level` (`G_j(t+1)`),
`delta` (`G_j(t+1) - G_j(t)`), `derivative` (delta / constant 50-unit step).

## Hypotheses and verdicts

### H1. Alpha tracks sparsity/density — SUPPORTED (with a deployability nuance)

- Predicted edge density falls monotonically as alpha rises at both sizes
  (alpha 0.001 -> ~0.93-0.98 density; alpha 1.0 -> 0).
- The oracle best-AUPR alpha rises from **0.03 (Size10)** to **0.1 (Size100)** as
  true density drops from ~0.16 to ~0.02. Stronger regularization is needed to
  push predicted density toward the sparser truth.
- The best-AUPR alpha does not exactly density-match (predicted density at the
  best alpha is 0.59 at Size10 and 0.07 at Size100, both above true density), so
  "alpha is a density knob" is directional, not a precise density-matching rule.
- Deployable proxies (no gold labels) land within one grid step of the oracle but
  none matches both regimes:

  | Proxy | Size10 chosen (oracle 0.03) | Size100 chosen (oracle 0.1) |
  |---|---|---|
  | CV MSE | 0.03 (match) | 0.03 (one step low) |
  | BIC | 0.1 (one step high) | 0.1 (match) |
  | density prior (2 reg/gene) | 0.1 | 0.3 |
  | bootstrap stability | 0.3 | 0.3 |

  CV is best at small scale, BIC at larger scale; bracketing CV and BIC covers the
  oracle. Mean AUPR gap to oracle is small for CV/BIC (<=0.03).

### H2. Include-self controls persistence — SUPPORTED, but not via simple residualization

- Self-only persistence explains **~59% of next-step level variance** at both
  sizes (mean self R^2 = 0.593 / 0.593): autoregression is large.
- Include-self minus exclude-self AUPR is **+0.166 (Size10)** and **+0.030
  (Size100)**: include-self helps at both sizes, more at Size10.
- **Self-permutation control:** permuting the self predictor removes essentially
  all of the include-self advantage (include minus permuted = +0.158 / +0.030).
  So self-persistence is doing real control work; the benefit is not accidental.
- **Residualized model (`dynamic_lasso_self_residualized`):** regressing
  `G_j(t+1)` on `G_j(t)` and predicting the residual from other genes only
  reproduces little of the Size10 advantage (+0.016 vs +0.166) but most of the
  Size100 advantage (+0.019 vs +0.030). So part of the include-self gain comes
  from **joint estimation** (fitting self and non-self together), not just from
  removing self-variance from the target. Self-dominance grows sharply with scale
  (self/non-self ratio 8.9 -> 117).

### H3. Fusion works via complementary errors — SUPPORTED

- Base-method (sparse / tree / correlation) rank correlation is moderate-to-low
  (Spearman 0.43 at Size10, **0.37 at Size100** — more complementary where fusion
  helps).
- Fusion (`fusion_borda`) top-k **true positives carry more multi-method support
  than false positives**: support 2.47 vs 1.88 (Size10) and 2.25 vs 1.59
  (Size100). Fusion promotes edges that multiple evidence types agree on, rather
  than merely averaging noise; the TP-vs-FP support gap is larger at Size100.

### H4. Edge metrics and topology metrics disagree — SUPPORTED

- Across method/network rows, Spearman(AUPR, top-hub overlap) is weak (0.28 at
  Size10, **0.11 at Size100**), and Spearman(AUPR, reciprocal-FP rate) is weakly
  negative (-0.46 / -0.14). Topology recovery is a partly separate objective from
  edge AUPR, more so at Size100.

### H5. Level beats delta/derivative because differencing lowers SNR — SUPPORTED

- Tree level AUPR strongly beats delta at both sizes (0.575 vs 0.289 at Size10;
  0.153 vs 0.045 at Size100).
- `var(delta)/var(level) ~= 0.38` at both sizes: differencing strips ~62% of the
  variance — the shared, persistent, cross-gene-predictable component that trees
  exploit — leaving a lower-SNR target.
- Delta and derivative rankings are near-identical (rank Spearman 0.997 / 0.978)
  because the DREAM4 time step is constant, so derivative is just a scaled delta.

## Outputs

Under `results/tables/`:

- `dream4_mechanism_alpha_density.csv` (H1 per-network alpha curves)
- `dream4_mechanism_alpha_proxies.csv` (H1 deployable proxy choices vs oracle)
- `dream4_mechanism_self_persistence.csv` (H2)
- `dream4_mechanism_residualized_edges.csv` (H2 residualized vs include/exclude edges)
- `dream4_mechanism_fusion_complementarity.csv` (H3)
- `dream4_mechanism_metric_relationships.csv` (H4)
- `dream4_mechanism_summary.csv` (headline across hypotheses)
- `dream4_mechanism_debug_report.md` (14 questions)

Under `results/figures/` (matplotlib available): `alpha_vs_aupr_by_size.png`,
`alpha_vs_density_by_size.png`, `aupr_vs_topology_scatter.png`,
`method_rank_correlation_heatmap.png`.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\13_dream4_mechanism_audit\run_mechanism_audit.py
# fast: Size10 only
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\13_dream4_mechanism_audit\run_mechanism_audit.py --quick
```

Flags: `--skip-size100`, `--n-jobs`, `--tree-estimators-size10/-size100`,
`--bootstrap-resamples`, `--random-seed`.

## General lessons beyond GRN inference

These are cautious, tied to the observed results; they are statistical patterns
this benchmark illustrates, not universal laws.

- **Regularization strength should reflect sparsity and sample size.** The best
  LASSO alpha rose as the true graph got sparser and the feature-to-sample ratio
  grew. A fixed alpha that wins on a small problem can be wrong on a larger,
  sparser one; tune the penalty to the regime, and prefer deployable selectors
  (cross-validation, an information criterion, or a sparsity prior) over a value
  carried over from another dataset.
- **Autoregressive / self terms can be essential controls but can dominate.**
  Including the lagged self predictor improved non-self recovery, and destroying
  it (permutation) removed the gain — yet the self coefficient dwarfed the others
  (ratio up to ~117). Persistence/level terms are worth including as controls,
  but their coefficients should not be read as the interesting signal, and their
  dominance should be monitored.
- **Ensembling/rank fusion helps only when errors are complementary.** Fusion won
  exactly where base-method rankings were least correlated and where true edges
  drew support from multiple methods. When components are redundant (high rank
  correlation), fusion mostly averages noise. Check error complementarity before
  ensembling.
- **Predictive ranking and structural recovery are different goals.** High AUPR
  did not track hub/degree/reciprocal recovery. If the objective is the hidden
  structure (hubs, directionality, motifs), it must be measured and optimized
  directly, not assumed to follow from a good global ranking.
- **Target formulation changes signal quality.** Predicting levels versus
  differences materially changed accuracy: differencing removed the shared,
  predictable component and lowered SNR. The "more dynamical" target was not the
  better-conditioned one here. Choose the target by signal-to-noise, not by
  apparent sophistication, and beware near-duplicate targets (delta vs derivative
  on a uniform grid).
- **Validation should include regime shifts, not one dataset.** Every conclusion
  changed in magnitude (and sometimes direction) between Size10 and Size100.
  Single-regime results over-generalize; evaluate across sizes/densities/sampling
  before claiming a method or hyperparameter is "best".

## Interpretation policy

Mechanistic, explanatory audit. Oracle-alpha, density-matched, and best-input
fusion selections are diagnostics, not deployable rules. Deployable alpha
selection (CV/BIC/density-prior) is reported separately and only approximates the
oracle. The honest next steps remain an official dynGENIE3 baseline and the
GeneNetWeaver sweeps designed in experiment 12.
