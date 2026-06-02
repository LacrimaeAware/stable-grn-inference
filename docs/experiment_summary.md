# Experiment Summary

This page summarizes the DREAM4 Size10 experiments completed so far. It is meant as a public, private-safe reference for what has been run, what the results suggest, and what should be tested next.

## Current Scope

The benchmark scope is DREAM4 Size10 and Size100 in-silico networks 1-5. The gold-standard files provide directed edge labels, and the expression files provide input observations. Size10 uses 90 directed non-self candidate edges per network; Size100 uses 9900.

Data regimes tested so far:

- `multifactorial`
- `knockouts`
- `knockdowns`
- `timeseries`, first treated as same-time observations and then tested with adjacent one-step lagged samples

DREAM4 Size100 time-series has now been used for a scaling audit (experiment 10) and for a combined dynGENIE3-style baseline, sparsity-calibration, and rank-fusion audit across both sizes (experiment 11). A GeneNetWeaver simulation-sweep design (experiment 12) is scaffolded but not executed. No official dynGENIE3 package is installed, so tree delta/derivative methods are dynGENIE3-style, not an official reproduction; no final stability-aware method has been implemented yet.

## Experiments

| Experiment | Script | Main Question |
|---|---|---|
| Size10 multifactorial baseline | `experiments/01_dream4_size10_baseline/run_correlation_baseline.py` | Can the repo load real DREAM4 Size10 files and score simple baselines? |
| Method comparison | `experiments/01_dream4_size10_baseline/run_method_comparison.py` | Which one-shot baseline is strongest on Size10 multifactorial data? |
| Stability audit | `experiments/02_dream4_size10_stability/run_stability_audit.py` | Does bootstrap/subsampling stability improve edge ranking on multifactorial data? |
| Data-regime audit | `experiments/03_dream4_size10_data_regimes/run_data_regime_audit.py` | Does the stability effect persist across multifactorial, perturbation, and time-series files? |
| GENIE3 baseline | `experiments/04_dream4_genie3_baseline/run_genie3_baseline.py` | How does a faithful GENIE3-style tree ensemble compare to correlation and stability correlation? |
| Topology evaluation | `experiments/06_dream4_size10_topology_evaluation/run_topology_evaluation.py` | Do top-ranked edges recover hubs, degree patterns, reciprocal structure, and motifs? |
| Lagged time-series audit | `experiments/07_dream4_size10_lagged_timeseries/run_lagged_timeseries_audit.py` | Does using temporal order improve directed edge recovery over same-time scoring? |
| Dynamic model batch | `experiments/08_dream4_size10_dynamic_model_batch/run_dynamic_model_batch_audit.py` | Which evidence types help most: dynamic targets, trees, sparse linear models, MLPs, stability, fusion, or preprocessing? |
| Dynamic sparse validation | `experiments/09_dream4_size10_dynamic_sparse_validation/run_dynamic_sparse_validation.py` | Is the best dynamic sparse-linear result robust, interpretable, and not just a self-persistence artifact? |
| Size100 dynamic sparse scaling | `experiments/10_dream4_size100_dynamic_sparse_scaling/run_size100_dynamic_sparse_scaling.py` | Does the Size10 dynamic sparse candidate survive on DREAM4 Size100 time-series data? |
| Dynamic baseline + calibration + fusion | `experiments/11_dream4_dynamic_baseline_and_calibration/run_dynamic_baseline_and_calibration.py` | How do dynGENIE3-style trees, alpha-calibrated sparse models, and rank fusion compare across Size10 and Size100? |
| GNW sweep design | `experiments/12_gnw_sweep_design/gnw_sweep_design.md` | How would we test whether these findings generalize under controlled GeneNetWeaver simulation? (design scaffold) |

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

## 6. Lagged Time-Series Audit

This experiment split each Size10 time-series file into trajectories when `Time` reset to 0.0, then built one-step lagged samples within each trajectory only. Each network had 5 trajectories, 105 time-series rows, and 100 lagged samples. The design scores source gene expression at time `t` against target gene expression at time `t+1`.

Mean results across the five Size10 networks:

| Method | Variant | Mean AUROC | Mean AUPR | Mean P@5 | Mean P@10 | Mean P@20 |
|---|---|---:|---:|---:|---:|---:|
| `lagged_genie3_random_forest` | lagged | 0.767932 | 0.531535 | 0.68 | 0.54 | 0.40 |
| `lagged_genie3_extra_trees` | lagged | 0.767890 | 0.528333 | 0.68 | 0.58 | 0.37 |
| `lagged_lasso_alpha_0_1` | lagged | 0.755521 | 0.509534 | 0.68 | 0.50 | 0.40 |
| `lagged_lasso_alpha_0_03` | lagged | 0.729813 | 0.486495 | 0.64 | 0.48 | 0.38 |
| `lagged_correlation` | lagged | 0.712754 | 0.458295 | 0.64 | 0.48 | 0.35 |
| `same_time_genie3_random_forest` | same-time | 0.725080 | 0.373040 | 0.44 | 0.40 | 0.34 |
| `same_time_lasso_alpha_0_1` | same-time | 0.683267 | 0.336698 | 0.40 | 0.38 | 0.29 |
| `same_time_correlation` | same-time | 0.653955 | 0.302771 | 0.36 | 0.34 | 0.30 |

Key comparisons:

- `lagged_correlation` improves mean AUPR by 0.155524 and mean AUROC by 0.058798 versus `same_time_correlation`.
- `lagged_lasso_alpha_0_1` improves mean AUPR by 0.172837 and mean AUROC by 0.072254 versus `same_time_lasso_alpha_0_1`.
- `lagged_genie3_random_forest` improves mean AUPR by 0.158495 and mean AUROC by 0.042853 versus the best same-time GENIE3-style reference.
- `lagged_genie3_random_forest` is the strongest first temporal baseline by both mean AUPR and mean AUROC.
- Topology remains mixed: lagged LASSO variants have the strongest top-3 hub overlaps, while lagged GENIE3 wins edge metrics.
- Reciprocal-direction false positives remain a problem. Lagged GENIE3 random forest slightly reduces reciprocal false-positive pair rate versus same-time GENIE3, but lagged correlation and lagged LASSO increase the rate versus their same-time references.

## 7. Dynamic Model Batch

This broad batch compared target types, self-predictor modes, tree ensembles, sparse linear models, a small MLP permutation-importance baseline, trajectory-bootstrap stability, equal-weight rank fusion, and light preprocessing. It did not implement reinforcement learning or GFlowNet graph search because those require a separate graph-search formulation.

Best mean edge metrics across the five Size10 networks:

| Metric | Winning Method | Family | Value |
|---|---|---|---:|
| AUPR | `dynamic_lasso_a0_03_level_include_self_raw` | sparse linear | 0.652712 |
| AUROC | `dynamic_lasso_a0_03_level_include_self_raw` | sparse linear | 0.821067 |
| precision@5 | `dynamic_lasso_a0_03_derivative_include_self_raw` | sparse linear | 0.920000 |
| precision@10 | `dynamic_lasso_a0_1_level_include_self_raw` | sparse linear | 0.720000 |
| precision@20 | `dynamic_elastic_net_a0_03_l1_0_7_level_include_self_raw` | sparse linear | 0.490000 |

Best topology/hub metrics:

| Metric | Winning Method | Value |
|---|---|---:|
| top-3 out-hub overlap | `dynamic_elastic_net_a0_03_l1_0_7_level_include_self_raw` | 0.533333 |
| top-3 in-hub overlap | `dynamic_lasso_a0_3_delta_exclude_self_raw` | 0.733333 |

Interpretation:

- Sparse linear dynamic models are the strongest family in this batch, overtaking the earlier lagged GENIE3 random forest on edge metrics.
- The best AUPR/AUROC result uses level targets and includes the self predictor during fitting while still excluding self-edges from the output.
- Delta and derivative targets do not beat level targets by mean AUPR/AUROC, but they can be useful for top-k precision and hub recovery.
- Include-self fitting helps non-self edge recovery in this audit, likely because persistence is informative. This needs careful follow-up because it changes the regression design substantially.
- MLP permutation importance underperforms the best sparse and tree models; it is only a sanity baseline so far.
- The fixed stability variants and equal-weight rank-fusion variants do not beat the best single dynamic model.
- Moving-average smoothing hurts the RF level/exclude-self baseline. Wavelet denoising was skipped because PyWavelets is not installed.

## 8. Dynamic Sparse Validation

This focused audit stress-tested the strongest dynamic sparse-linear candidate from the broad dynamic batch. It compared LASSO level/delta targets with include-self and exclude-self predictor modes, Elastic Net include-self variants, lagged GENIE3-style tree references, and trajectory-bootstrap rankings for selected sparse candidates.

Best mean edge metrics across the five Size10 networks:

| Metric | Winning Method | Value |
|---|---|---:|
| AUPR | `dynamic_lasso_level_include_self_a0_03` | 0.652712 |
| AUROC | `dynamic_lasso_level_include_self_a0_03` | 0.821067 |
| precision@10 | `dynamic_lasso_level_include_self_a0_03` | 0.680000 |

Important comparisons:

| Method | Mean AUROC | Mean AUPR | Mean Reciprocal FP Pair Rate |
|---|---:|---:|---:|
| `dynamic_lasso_level_include_self_a0_03` | 0.821067 | 0.652712 | 0.200000 |
| `lagged_genie3_random_forest` | 0.767974 | 0.536451 | 0.950000 |
| `lagged_genie3_extra_trees` | 0.765530 | 0.515946 | 1.000000 |
| `lagged_correlation_reference` | 0.712754 | 0.458295 | 1.000000 |

Validation findings:

- `alpha=0.03` is the best LASSO alpha by mean AUPR in the tested grid.
- The result is strong but not uniform: the winning method is the per-network AUPR winner on 2 of 5 networks.
- Include-self improves matched LASSO AUPR in most tested target/alpha comparisons, with 5 of 5 network wins for several moderate-alpha level and delta settings.
- Persistence is substantial. For the winning model, mean absolute self coefficient is roughly 8.9 times the mean absolute non-self coefficient.
- Bootstrap mean-absolute-coefficient ranking is close to the one-shot winner but does not improve it; bootstrap selection-frequency ranking hurts the include-self candidates.
- The best sparse dynamic model sharply reduces reciprocal false-positive pair rate compared with lagged GENIE3 and lagged correlation references.
- Topology agrees partially, not completely. The edge-metric winner has top-3 out-hub overlap 0.466667 and top-3 in-hub overlap 0.600000, while other sparse variants win the best hub-overlap metrics.

## 9. Size100 Dynamic Sparse Scaling

This audit scaled the Size10 main candidate `dynamic_lasso_level_include_self_a0_03` to the five DREAM4 Size100 time-series networks (100 genes, 10 trajectories of 21 points, 200 lagged samples, 9900 directed non-self candidate edges, ~2% true-edge density). It kept a compact method set: the include-self candidate, a stronger-alpha include-self LASSO, a matched exclude-self control, lagged correlation, an Elastic Net include-self variant, and reduced-tree (200) lagged GENIE3 references.

Mean results across the five Size100 networks:

| Method | Mean AUROC | Mean AUPR | Mean P@10 | Self/Non-Self Ratio | Reciprocal FP Pair Rate |
|---|---:|---:|---:|---:|---:|
| `dynamic_lasso_level_include_self_a0_1` | 0.658451 | 0.161467 | 0.660000 | 117.05 | 0.950000 |
| `lagged_genie3_random_forest` | 0.754354 | 0.145445 | 0.500000 | n/a | 0.995122 |
| `lagged_genie3_extra_trees` | 0.748207 | 0.142816 | 0.500000 | n/a | 0.995652 |
| `dynamic_lasso_level_include_self_a0_03` | 0.678593 | 0.130486 | 0.460000 | 26.29 | 1.000000 |
| `lagged_correlation_reference` | 0.702563 | 0.129961 | 0.580000 | n/a | 0.992262 |
| `dynamic_lasso_level_exclude_self_a0_03` | 0.672371 | 0.119005 | 0.500000 | n/a | 0.992308 |
| `dynamic_elastic_net_level_include_self_a0_03_l1_0_7` | 0.672042 | 0.111418 | 0.400000 | 16.71 | 1.000000 |

Scaling findings:

- The specific Size10 winner does not scale. `dynamic_lasso_level_include_self_a0_03` only ties lagged correlation on mean AUPR and trails it on mean AUROC, and it wins per-network AUPR on 0 of 5 networks.
- Best mean AUPR is the higher-alpha sibling `dynamic_lasso_level_include_self_a0_1` (0.161467); the larger, sparser network favors stronger regularization.
- Best mean AUROC is `lagged_genie3_random_forest` (0.754354), which wins AUROC on all five networks.
- Include-self still beats exclude-self on mean AUPR by about 0.0115, the same direction as Size10 but small.
- Self-persistence is even more extreme: the self/non-self absolute coefficient ratio rises to 26.3 at a0.03 (117 at a0.1) versus ~8.9 at Size10, while edge recovery weakens.
- The distinctive Size10 reciprocal-direction advantage disappears: the candidate's reciprocal false-positive pair rate is 1.00 at Size100, no better than correlation or trees, versus ~0.20 at Size10.
- Hub recovery is mixed (top-5 out-hub overlap below correlation, in-hub tied) and in-degree recovery is near zero for every method.

Interpretation: the headline Size10 `a0_03` result looks substantially like a small-network effect. The include-self sparse family is not dead - it still leads mean AUPR at a higher alpha - but the specific candidate, and especially its reciprocal-direction advantage, does not reproduce at Size100.

## 10. Dynamic Baseline, Calibration, and Fusion

This audit ran across both sizes with three goals: a dynGENIE3-style tree baseline (level/delta/derivative targets), a sparsity-calibration sweep for LASSO/Elastic Net over alpha `[0.001 ... 1.0]`, and rank fusion (mean reciprocal rank, Borda, normalized score, plus a reciprocal-direction penalty). No official dynGENIE3 package is installed, so the delta/derivative trees are dynGENIE3-style.

Best method per size (mean across five networks):

| Size | Best AUPR | AUPR | Best AUROC | AUROC |
|---|---|---:|---|---:|
| 10 | `dynamic_lasso_level_include_self_a0_03` | 0.652712 | `dynamic_lasso_level_include_self_a0_03` | 0.821067 |
| 100 | `fusion_borda` | 0.208067 | `lagged_genie3_rf_level` | 0.753913 |

Findings:

- dynGENIE3-style delta/derivative tree targets do not help; they clearly hurt versus level GENIE3 at both sizes (Size100 delta RF AUPR 0.045 vs level RF 0.157). Derivative and delta tree rankings nearly coincide on DREAM4's uniform time grid.
- Sparsity calibration explains the Size10-vs-Size100 difference. The best AUPR alpha for LASSO level include-self rises from 0.03 (Size10) to 0.1 (Size100), and at Size100 that peak alpha drives predicted edge density (0.07) toward the true density (0.02). Alpha behaves like a density knob, and stronger regularization is consistently better at Size100.
- Include-self still beats exclude-self at both sizes after per-config alpha calibration (Size10 +0.166 AUPR; Size100 +0.030 AUPR).
- Self-persistence is dominant and grows with size and alpha (self/non-self ratio up to ~465 at Size100), so it reads as a model-stability term that is useful to fit but dangerous to interpret as directed regulation.
- Rank fusion helps in the hard regime: at Size100 `fusion_borda` is the best AUPR method overall (0.208 vs best single input 0.173, with precision@10 0.84), while at Size10 the single best sparse method still wins. The reciprocal-direction penalty gives a small Size100 gain over base fusion and slightly lowers the reciprocal false-positive rate.
- By AUROC, trees (`lagged_genie3_rf_level`, 0.754) remain the Size100 leader, so a literature-faithful dynGENIE3 comparison is the right next reference.

## Cross-Experiment Conclusions

The current evidence supports a cautious story:

- Correlation is a surprisingly strong simple baseline on Size10 DREAM4 data. It should remain in every comparison.
- Stability correlation is meaningful: it repeatedly improves AUPR, especially in small-data settings where top-ranked edge quality matters more than global ranking smoothness.
- Stability correlation is not yet clearly topology-improving. Its edge-ranking gains should be evaluated against hubs, degree patterns, and reciprocal-direction errors.
- Sparse LASSO is not yet the dominant base estimator. Tuning helped, and knockouts are promising, but LASSO stability is mixed.
- GENIE3 is now the strongest non-sparse baseline family by AUPR in several regimes. Any future stability-aware method should compare against GENIE3, not only correlation and LASSO.
- GENIE3 is not a complete topology answer on Size10. It helps edge AUPR, but hub and degree recovery remain mixed.
- Same-time treatment of time-series data is only an audit shortcut. The first lagged audit shows that temporal order substantially improves edge recovery.
- Lagged models are now the strongest Size10 time-series baselines by edge metrics, but reciprocal-direction and topology recovery issues remain.
- The dynamic batch suggests sparse linear models with self-persistence included during fitting may be the most promising Size10 temporal baseline so far.
- The dynamic sparse validation supports `dynamic_lasso_level_include_self_a0_03` as the current Size10 temporal sparse candidate, but it also confirms that self-persistence is a major part of the model and must be validated carefully.
- Bootstrap sparse selection is not automatically helpful in the dynamic setting tested here. Mean absolute bootstrapped coefficients are close to one-shot coefficients, while selection frequency underperforms.
- Neural MLP does not help in this small-data setting. Simple rank fusion does not help at Size10, but it does help in the harder Size100 regime (see below).
- The Size100 scaling audit weakens the Size10 headline. The specific `dynamic_lasso_level_include_self_a0_03` candidate does not scale: at Size100 it only ties lagged correlation on AUPR, trails it on AUROC, wins 0 of 5 networks, and loses its reciprocal-direction advantage entirely. Higher regularization (`a0_1`) is the best sparse setting at Size100, and lagged GENIE3 wins AUROC on every Size100 network. The include-self sparse family remains worth studying, but the small-network result should not be promoted as a main method.
- The calibration/fusion audit reframes the alpha question and revives fusion. The best alpha is not magic; it tracks network density (0.03 at Size10, 0.1 at Size100, pushing predicted density toward true density). dynGENIE3-style delta/derivative tree targets hurt versus level GENIE3. Rank fusion of complementary sparse + tree + correlation evidence is the best Size100 AUPR method (`fusion_borda` 0.208, precision@10 0.84), and a fixed reciprocal-direction penalty slightly improves Size100 AUPR and reciprocal false positives. No single method wins both AUPR and AUROC, and self-persistence remains a dominant but interpretation-dangerous term.

## Current Recommendation

The defensible current position is that dynamic GRN inference on DREAM4 is regime-dependent rather than owned by one method. Concretely:

- Keep `dynamic_lasso_level_include_self_a0_03` as a Size10 finding only; it does not scale to Size100.
- Treat alpha as a calibrated, density-tracking knob, not a fixed value. Use stronger regularization on larger, sparser networks (best alpha 0.03 at Size10, 0.1 at Size100).
- At Size100, prefer rank fusion of complementary sparse + tree + correlation evidence (best AUPR, best precision@k) and keep level GENIE3 as the AUROC reference; do not use dynGENIE3-style delta/derivative tree targets, which hurt.
- Treat self-persistence as a model-stability term, not directed-edge evidence.

The next concrete steps are: (1) add a literature-faithful (official) dynGENIE3 baseline so the dynamic comparison is fair, since no official package is currently installed; (2) execute the GeneNetWeaver simulation sweeps designed in `experiments/12_gnw_sweep_design/gnw_sweep_design.md` to test whether the density/alpha and fusion findings generalize; and (3) only then consider perturbation/knockout dynamic models.

## Caveats

- These are Size10 results across five networks, not final evidence.
- Multifactorial, knockout, knockdown, and time-series files have different statistical meanings.
- Bootstrap resampling with very small sample counts is fragile and should be treated as an audit, not as a validated estimator.
- Metrics can disagree: AUPR, AUROC, and precision@k each reward different ranking behavior.
- Topology metrics can also disagree: hub overlap, degree Spearman, reciprocal counts, and motif counts each capture different structure.
- The lagged audit uses temporal order, but it is still not causal validation.
