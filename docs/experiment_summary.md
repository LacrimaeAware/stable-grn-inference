# Experiment Summary

This page summarizes the DREAM4 Size10 experiments completed so far. It is meant as a public, private-safe reference for what has been run, what the results suggest, and what should be tested next.

## Current Scope

The current benchmark scope is DREAM4 Size10 in-silico networks 1-5. The gold-standard files provide directed edge labels, and the expression files provide input observations. Most experiments use 90 directed non-self candidate edges per network.

Data regimes tested so far:

- `multifactorial`
- `knockouts`
- `knockdowns`
- `timeseries`, treated only as same-time observations with `Time` dropped

No lagged time-series inference, Size100 scaling, or final stability-aware method has been implemented yet.

## Experiments

| Experiment | Script | Main Question |
|---|---|---|
| Size10 multifactorial baseline | `experiments/01_dream4_size10_baseline/run_correlation_baseline.py` | Can the repo load real DREAM4 Size10 files and score simple baselines? |
| Method comparison | `experiments/01_dream4_size10_baseline/run_method_comparison.py` | Which one-shot baseline is strongest on Size10 multifactorial data? |
| Stability audit | `experiments/02_dream4_size10_stability/run_stability_audit.py` | Does bootstrap/subsampling stability improve edge ranking on multifactorial data? |
| Data-regime audit | `experiments/03_dream4_size10_data_regimes/run_data_regime_audit.py` | Does the stability effect persist across multifactorial, perturbation, and time-series files? |
| GENIE3 baseline | `experiments/04_dream4_genie3_baseline/run_genie3_baseline.py` | How does a faithful GENIE3-style tree ensemble compare to correlation and stability correlation? |
| Topology evaluation | `experiments/06_dream4_size10_topology_evaluation/run_topology_evaluation.py` | Do top-ranked edges recover hubs, degree patterns, reciprocal structure, and motifs? |

Generated outputs are saved under ignored `results/tables/`.

## 1. Size10 Multifactorial Method Comparison

This experiment compared one-shot edge scores on the five Size10 multifactorial datasets. Methods included absolute correlation, a small LASSO alpha grid, a small Elastic Net grid, and a quick random-forest feature-importance baseline.

Mean results across the five Size10 networks:

| Method | Mean AUROC | Mean AUPR | Mean P@5 | Mean P@10 | Mean P@20 |
|---|---:|---:|---:|---:|---:|
| `correlation` | 0.665445 | 0.330159 | 0.40 | 0.36 | 0.30 |
| `random_forest_importance` | 0.600773 | 0.302161 | 0.36 | 0.40 | 0.27 |
| `lasso_alpha_0_1` | 0.593003 | 0.292166 | 0.40 | 0.32 | 0.28 |
| `lasso_alpha_0_01` | 0.504195 | 0.195631 | 0.16 | 0.20 | 0.17 |
| best Elastic Net tested: `elastic_net_alpha_0_03_l1_0_95` | 0.526004 | 0.234376 | 0.24 | 0.22 | 0.20 |

Interpretation:

- Absolute correlation was the strongest one-shot baseline on Size10 multifactorial data.
- Tuning LASSO mattered: `lasso_alpha_0_1` clearly improved over the original fixed `alpha=0.01`.
- Sparse linear baselines did not beat correlation on this small multifactorial setting.
- The quick random-forest feature-importance baseline was competitive, but it was not yet a clearly named GENIE3 implementation.

## 2. Size10 Multifactorial Stability Audit

This experiment tested whether repeated resampling could improve edge ranking on Size10 multifactorial data. Stability correlation used mean reciprocal rank across bootstrap resamples. LASSO and Elastic Net stability used selection frequency from nonzero coefficients. Random-forest stability used mean feature importance.

Mean results across the five Size10 networks:

| Method | Mean AUROC | Mean AUPR | Mean P@5 | Mean P@10 | Mean P@20 |
|---|---:|---:|---:|---:|---:|
| `one_shot_correlation` | 0.665445 | 0.330159 | 0.40 | 0.36 | 0.30 |
| `stability_correlation` | 0.657599 | 0.364574 | 0.44 | 0.38 | 0.31 |
| `one_shot_lasso_alpha_0_1` | 0.593003 | 0.292166 | 0.40 | 0.32 | 0.28 |
| `stability_lasso_alpha_0_1` | 0.609900 | 0.297757 | 0.40 | 0.32 | 0.25 |
| `stability_random_forest_importance` | 0.586114 | 0.292661 | 0.44 | 0.34 | 0.28 |

Interpretation:

- Stability correlation improved mean AUPR and precision@k over one-shot correlation, but one-shot correlation retained slightly higher mean AUROC.
- LASSO stability improved modestly over one-shot LASSO, especially AUROC, but it did not approach correlation.
- This supported continuing with stability-aware ranking as an audit path, while keeping correlation as the simple benchmark.

## 3. Size10 Data-Regime Audit

This experiment extended one-shot and stability methods across the available Size10 data regimes: multifactorial, knockouts, knockdowns, and time-series files treated as same-time observations.

Best method by mean AUPR:

| Data Regime | Best Method | Mean AUPR |
|---|---|---:|
| `knockdowns` | `stability_correlation` | 0.289342 |
| `knockouts` | `one_shot_lasso_alpha_0_1` | 0.424674 |
| `multifactorial` | `stability_correlation` | 0.365318 |
| `timeseries` | `one_shot_random_forest_importance` | 0.369037 |

Best method by mean AUROC:

| Data Regime | Best Method | Mean AUROC |
|---|---|---:|
| `knockdowns` | `one_shot_correlation` | 0.650854 |
| `knockouts` | `stability_correlation` | 0.718376 |
| `multifactorial` | `stability_correlation` | 0.674167 |
| `timeseries` | `stability_random_forest_importance` | 0.718962 |

Interpretation:

- Stability correlation improved mean AUPR over one-shot correlation in all four regimes.
- Stability correlation did not consistently improve AUROC; knockdowns were a slight counterexample.
- LASSO stability was mixed. It improved some metrics in some regimes but was not robustly better than one-shot LASSO.
- Knockouts were notable because one-shot `lasso_alpha_0_1` won mean AUPR. Perturbation-rich data may be a better setting for sparse regression than multifactorial data.
- Same-time time-series scoring favored tree-style feature importance, but this should not be read as evidence about lagged regulation.

## 4. GENIE3 Baseline

This experiment added a clearer GENIE3-style implementation: for each target gene, fit a tree ensemble using all other genes as predictors, then use feature importances as directed source-to-target edge scores. Both `RandomForestRegressor` and `ExtraTreesRegressor` variants were tested.

Best method by mean AUPR:

| Data Regime | Best Method | Mean AUPR |
|---|---|---:|
| `knockdowns` | `stability_correlation` | 0.304808 |
| `knockouts` | `genie3_extra_trees` | 0.393540 |
| `multifactorial` | `genie3_extra_trees` | 0.379097 |
| `timeseries` | `genie3_random_forest` | 0.372960 |

Best method by mean AUROC:

| Data Regime | Best Method | Mean AUROC |
|---|---|---:|
| `knockdowns` | `one_shot_correlation` | 0.650854 |
| `knockouts` | `stability_correlation` | 0.732779 |
| `multifactorial` | `stability_correlation` | 0.671269 |
| `timeseries` | `genie3_extra_trees` | 0.736178 |

Interpretation:

- GENIE3 improved mean AUPR over one-shot correlation in all four tested regimes.
- GENIE3 beat stability correlation by mean AUPR in knockouts, multifactorial, and same-time time-series scoring.
- Stability correlation still remained competitive: it won mean AUPR on knockdowns and mean AUROC on knockouts and multifactorial.
- Extra Trees was the strongest tree-ensemble variant overall, except same-time time-series AUPR where random forest was slightly higher.

## 5. Topology-Aware Evaluation

This experiment converted ranked edge lists into predicted directed graphs at top 5, top 10, top 20, and top N true edges. It evaluated hub and degree recovery, reciprocal-direction false positives, reciprocal edge counts, feed-forward loop counts, and top true hub edge precision/recall.

The primary interpretation uses top N true edges, which gives each predicted graph the same edge count as its matching gold-standard network.

Best top-3 out-hub recovery:

| Data Regime | Best Method | Top-3 Out-Hub Overlap | Out-Degree Spearman |
|---|---|---:|---:|
| `knockdowns` | `one_shot_correlation` | 0.466667 | 0.345761 |
| `knockouts` | `one_shot_correlation` | 0.533333 | 0.363751 |
| `multifactorial` | `one_shot_correlation` | 0.400000 | 0.228912 |
| `timeseries` | `genie3_random_forest` | 0.400000 | 0.151890 |

Best top-3 in-hub recovery:

| Data Regime | Best Method | Top-3 In-Hub Overlap | In-Degree Spearman |
|---|---|---:|---:|
| `knockdowns` | `one_shot_correlation` | 0.466667 | 0.284646 |
| `knockouts` | `stability_correlation` | 0.400000 | 0.219762 |
| `multifactorial` | `genie3_extra_trees` | 0.333333 | 0.199666 |
| `timeseries` | `stability_correlation` | 0.333333 | 0.107265 |

Interpretation:

- Stability correlation's AUPR gains do not consistently translate into hub or degree recovery. Against one-shot correlation, it improves top-3 out-hub overlap in 0 of 4 regimes and top-3 in-hub overlap in 1 of 4 regimes.
- GENIE3's AUPR advantage only partially translates into topology recovery. The best GENIE3 topology variant improves top-3 out-hub overlap over stability correlation in 2 of 4 regimes and top-3 in-hub overlap in 2 of 4 regimes.
- One-shot correlation remains surprisingly strong for out-degree hub recovery, winning 3 of 4 regimes by top-3 out-hub overlap.
- Correlation has a direction-symmetry issue: at top N true edges, one-shot correlation has a mean reciprocal false-positive pair rate of 0.954. Other methods also produce many reciprocal false-positive pairs, so directionality remains a broad weakness of same-time scoring.
- Topology metrics make the hidden-structure problem sharper: edge-level AUPR/AUROC gains alone are not enough.

## Cross-Experiment Conclusions

The current evidence supports a cautious story:

- Correlation is a surprisingly strong simple baseline on Size10 DREAM4 data. It should remain in every comparison.
- Stability correlation is meaningful: it repeatedly improves AUPR, especially in small-data settings where top-ranked edge quality matters more than global ranking smoothness.
- Stability correlation is not yet clearly topology-improving. Its edge-ranking gains should be evaluated against hubs, degree patterns, and reciprocal-direction errors.
- Sparse LASSO is not yet the dominant base estimator. Tuning helped, and knockouts are promising, but LASSO stability is mixed.
- GENIE3 is now the strongest non-sparse baseline family by AUPR in several regimes. Any future stability-aware method should compare against GENIE3, not only correlation and LASSO.
- GENIE3 is not a complete topology answer on Size10. It helps edge AUPR, but hub and degree recovery remain mixed.
- Same-time treatment of time-series data is only an audit shortcut. It does not answer whether lagged directed inference works.

## Current Recommendation

The next main branch should be proper lagged time-series inference or a dynGENIE3-style baseline. The topology audit shows that same-time edge rankings can score well while still struggling with directionality and graph structure, which is central to the hidden-structure question.

Stability-GENIE3 remains a useful smaller branch or ablation, but it should not replace the dynamic-inference step. Size100 multifactorial scaling should wait until the method comparison includes topology-aware evaluation and a stronger directional baseline.

## Caveats

- These are Size10 results across five networks, not final evidence.
- Multifactorial, knockout, knockdown, and time-series files have different statistical meanings.
- Bootstrap resampling with very small sample counts is fragile and should be treated as an audit, not as a validated estimator.
- Metrics can disagree: AUPR, AUROC, and precision@k each reward different ranking behavior.
- Topology metrics can also disagree: hub overlap, degree Spearman, reciprocal counts, and motif counts each capture different structure.
- The current time-series handling ignores temporal direction and should not be interpreted causally.
