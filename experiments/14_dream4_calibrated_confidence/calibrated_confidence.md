# DREAM4 Calibrated Confidence

This experiment turns the mechanism findings (experiment 13) into a **deployable,
gold-free edge-confidence pipeline**: select the sparse regularization without
gold labels, build an equal-weight confidence score from complementary methods,
check whether the confidence is calibrated, and keep topology objectives
separate. Gold labels are used only for evaluation after selection.

Main question: can the current findings become a deployable edge-confidence rule
that does not use gold labels for tuning? **Largely yes**, with the honest caveat
that the best method is regime-dependent (sparse at Size10, fusion at Size100)
rather than one universal model.

dynGENIE3 status: no official package is installed, so tree methods are
dynGENIE3-style.

## Part 1: Deployable alpha selection (no gold labels)

For `dynamic_lasso_level_include_self` (the focal model; exclude-self and
Elastic Net are also swept), selecting alpha with cross-validation MSE, BIC, AIC,
a density prior (1/2/3 regulators per gene), or bootstrap selection stability:

| Size | Rule | Chosen alpha | Oracle alpha | AUPR | % of oracle AUPR | Predicted density (true) |
|---|---|---:|---:|---:|---:|---:|
| 10 | oracle | 0.03 | 0.03 | 0.6557 | 100% | 0.551 (0.158) |
| 10 | cv | 0.03 | 0.03 | 0.6398 | 97.6% | 0.576 |
| 10 | bic | 0.10 | 0.03 | 0.6277 | 95.7% | 0.300 |
| 10 | aic | 0.03 | 0.03 | 0.6437 | 98.2% | 0.647 |
| 100 | oracle | 0.10 | 0.10 | 0.1615 | 100% | 0.071 (0.021) |
| 100 | bic | 0.10 | 0.10 | 0.1615 | 100% | 0.071 |
| 100 | cv | 0.03 | 0.10 | 0.1305 | 80.8% | 0.255 |

- **BIC is the best deployable rule overall** (smallest mean AUPR gap to oracle,
  0.014), then CV (0.023). BIC matches the oracle exactly at Size100; CV matches
  exactly at Size10.
- Deployable selection preserves **96-100%** of the oracle sparse model's AUPR
  (CV 97.6% at Size10, BIC 100% at Size100). A practical rule: use CV at small
  scale and BIC at larger/sparser scale, or take the sparser of the two.
- The density-prior heuristic and bootstrap stability tend to over-regularize;
  they are reasonable fallbacks but worse than CV/BIC here.

## Part 2: Confidence from equal-weight method agreement

Confidence inputs are the **deployable** CV-selected sparse model, lagged GENIE3
RF (level), and lagged correlation. All variants use equal weights and fixed
a-priori penalties.

- **Size10:** the single deployable sparse model (`sparse_cv_alpha`, AUPR 0.640)
  beats every fusion/agreement variant (best confidence 0.609). Fusion does not
  help in the small, denser regime.
- **Size100:** confidence fusion wins. `fusion_borda` (AUPR 0.183, precision@10
  0.72) beats the best single deployable method (`sparse_bic_alpha`, 0.162), and
  the agreement-count confidences (top 1/5/10%) are next. This matches experiment
  13: base methods are more complementary at Size100 (lower rank correlation) and
  true positives draw multi-method support.

## Part 3: Calibration diagnostics

Treating scores as confidence and binning edges into rank deciles:

| Size | Method | Conf-vs-true-rate Spearman | Top-bin true rate | Bottom-bin true rate |
|---|---|---:|---:|---:|
| 10 | sparse_cv_alpha | 0.657 | 0.689 | 0.089 |
| 10 | fusion_borda | 0.521 | 0.689 | 0.089 |
| 100 | sparse_cv_alpha | 0.707 | 0.076 | 0.006 |
| 100 | fusion_borda | 0.828 | 0.088 | 0.008 |

- The rankings are **meaningfully calibrated as confidence**: higher-confidence
  bins have clearly higher empirical true-edge rates (top bin ~8-11x the bottom
  bin), with positive confidence-vs-true-rate Spearman at both sizes.
- The ECE-style number is large (especially for fusion) because raw scores are
  not probabilities; the *ordering* is reliable even though the magnitude is not a
  probability. A deployable system should rank/threshold by confidence, not read
  the score as P(edge).

## Part 4: Topology-aware decision layer (separate winners)

Edge ranking, top-k precision, hub recovery, and reciprocal-direction control
have **different deployable winners** - they are not one objective:

| Size | Best AUPR | Best precision@10 | Best top-hub overlap | Lowest reciprocal-FP |
|---|---|---|---|---|
| 10 | sparse_cv_alpha | sparse_bic_alpha (0.72) | confidence_agreement_top1pct | sparse_cv_alpha (0.20) |
| 100 | fusion_borda (0.183) | confidence_topology_penalty (0.76) | confidence_agreement_top5pct | confidence_topology_penalty (0.00) |

- The **topology penalty** drives the Size100 reciprocal false-positive pair rate
  to 0.0 - but by construction: it removes reciprocal pairs entirely (predicted
  reciprocal pairs 0 vs 30 for Borda), so the 0.0 rate means "no reciprocal pairs
  emitted," not "perfect reciprocal classification." Usefully, it also gives the
  **best Size100 precision@10 (0.76)** at a small AUPR cost (0.157 vs 0.183), so
  suppressing reciprocity helps top-k and directionality.
- The reciprocal penalty (keep the stronger direction) reduces reciprocal-FP more
  gently and keeps more AUPR.

## Part 5: Comparison baselines

Includes lagged correlation, lagged GENIE3 RF/Extra Trees (level), `sparse_oracle_alpha`
(ORACLE, not deployable), `sparse_cv_alpha`, `sparse_bic_alpha`, `fusion_borda`,
and the confidence/agreement variants. Official dynGENIE3 was unavailable, so the
dynGENIE3-style RF/ET references are used.

## Outputs

Under `results/tables/`: `dream4_calibrated_confidence_summary.csv`,
`_per_network.csv`, `_edges.csv`, `_calibration_bins.csv`, `_topology.csv`,
`_alpha_selection.csv`, `_debug_report.md`. Figures under `results/figures/`:
`selected_alpha_by_rule_and_size.png`, `predicted_density_by_selection_rule.png`,
`confidence_bin_true_edge_rate.png`, `aupr_comparison_by_method.png`,
`reciprocal_fp_comparison.png`.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\14_dream4_calibrated_confidence\run_calibrated_confidence.py
# fast: Size10 only
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\14_dream4_calibrated_confidence\run_calibrated_confidence.py --quick
```

Flags: `--skip-size100`, `--n-jobs`, `--tree-estimators-size10/-size100`,
`--bootstrap-resamples`, `--random-seed`.

## Is this a Track A method candidate?

**Partially - it is reportable as a calibrated-confidence methodology, not yet a
single dominant method.** Deployable alpha selection retains ~96-100% of the
oracle sparse model, confidence rankings are meaningfully calibrated, and the
pipeline is fully gold-free except for evaluation. But the best method is
regime-dependent (sparse at Size10, fusion at Size100), and topology objectives
need a separate decision layer. The defensible deployable recommendation:

1. Select alpha with CV (small/dense) or BIC (large/sparse), or take the sparser.
2. Rank edges by equal-weight confidence (Borda / agreement count) of the
   deployable sparse model, a tree model, and correlation.
3. Apply a fixed reciprocal/topology penalty if directionality or top-k precision
   matters more than global AUPR.
4. Threshold by confidence rank, not by the raw score as a probability.

This should be validated with an official dynGENIE3 baseline and the GeneNetWeaver
sweeps (experiment 12) before being called a finished method.

## Interpretation policy

Alpha selection, confidence weights, and penalties use no gold labels; gold is
used only for evaluation. The `sparse_oracle_alpha` row is an upper-bound
diagnostic, not deployable. ECE magnitudes are reported but the scores are
confidence rankings, not calibrated probabilities. The topology penalty's zero
reciprocal-FP rate reflects reciprocal-pair suppression, stated explicitly to
avoid over-claiming.
