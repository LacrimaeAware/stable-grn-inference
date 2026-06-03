# Experiment Summary

This page summarizes the DREAM4 Size10 experiments completed so far. It is meant as a public, private-safe reference for what has been run, what the results suggest, and what should be tested next.

## Current Scope

The benchmark scope is DREAM4 Size10 and Size100 in-silico networks 1-5. The gold-standard files provide directed edge labels, and the expression files provide input observations. Size10 uses 90 directed non-self candidate edges per network; Size100 uses 9900.

Data regimes tested so far:

- `multifactorial`
- `knockouts`
- `knockdowns`
- `timeseries`, first treated as same-time observations and then tested with adjacent one-step lagged samples

DREAM4 Size100 time-series has now been used for a scaling audit (experiment 10), a combined dynGENIE3-style baseline / sparsity-calibration / rank-fusion audit across both sizes (experiment 11), a mechanism/hypothesis audit that explains those findings (experiment 13), and a deployable, gold-free calibrated-confidence pipeline (experiment 14). A GeneNetWeaver simulation-sweep design (experiment 12) is scaffolded but not executed. No official dynGENIE3 package is installed, so tree delta/derivative methods are dynGENIE3-style, not an official reproduction; no final stability-aware method has been implemented yet.

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
| GNW sweep design | `experiments/12_gnw_sweep_design/gnw_sweep_design.md` | How to test whether these findings generalize under controlled GeneNetWeaver simulation? (design scaffold) |
| Mechanism audit | `experiments/13_dream4_mechanism_audit/run_mechanism_audit.py` | Why do the current winners work or fail (alpha/density, persistence, fusion complementarity, edge-vs-topology, target choice)? |
| Calibrated confidence | `experiments/14_dream4_calibrated_confidence/run_calibrated_confidence.py` | Can the findings become a deployable, gold-free edge-confidence rule (alpha selection + agreement confidence + calibration + topology layer)? |
| Modern benchmark scouting | `experiments/15_modern_grn_benchmark_adapter/modern_grn_benchmark_adapter.md` | Which modern GRN benchmark should validate the pipeline beyond DREAM4/GNW? (chose BEELINE) |
| BEELINE adapter smoke | `experiments/16_beeline_adapter_smoke/run_beeline_adapter_smoke.py` | Can the pipeline ingest BEELINE-format single-cell datasets and run static methods end-to-end? |
| Stability + orientation diagnostics | `experiments/17_dream4_stability_orientation_diagnostics/run_stability_orientation_diagnostics.py` | Is the error skeleton or orientation? Is alpha theory-predictable? Is fusion complementarity or variance reduction? Does formal stability selection support the stability-selection thesis? |

Generated outputs are saved under ignored `results/tables/`.

## Modern-Benchmark Direction (single-cell)

Beyond DREAM4/GNW, the project is moving toward modern single-cell GRN benchmarks. Experiment 15 scouted candidates (BEELINE, CausalBench, raw Perturb-seq, curated priors) and chose **BEELINE** as the first target (lowest-friction transfer: expression matrix + directed reference + AUPR/precision@k/EPR). Experiment 16 implemented the adapter (`src/stable_grn_inference/data/beeline.py`: `GrnBenchmarkDataset` + `load_beeline_dataset`) and smoke-tested it on a synthetic BEELINE-format fixture. Correlation, GENIE3/trees, static sparse LASSO, and rank fusion transfer directly; dynamic lagged/include-self methods do not (no time/pseudotime); references are biological proxies (report EPR, not just AUPR); candidate edges are TF→gene. No real BEELINE data is committed; place datasets under `data/raw/beeline/`.

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
- Moving-average smoothing and wavelet denoising both slightly hurt the RF level/exclude-self baseline by mean AUPR (raw 0.537, moving-average-3 0.487, wavelet-denoise 0.519); wavelet denoising hurts less than moving-average but neither beats the raw baseline. (PyWavelets is now installed - see `requirements.txt` - so the wavelet-denoise ablation runs as a real variant rather than being skipped.)

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

## 11. Mechanism Audit

This audit explained the experiment 9-11 winners rather than adding models. It tested five hypotheses across Size10 and Size100 and reused the pipeline; alpha was tuned on gold labels only as an oracle diagnostic.

- **H1 (alpha tracks density) - supported.** Predicted edge density falls monotonically with alpha, and the oracle best-AUPR alpha rises from 0.03 (Size10, true density ~0.16) to 0.1 (Size100, true density ~0.02). It is directional, not exact density-matching (best-alpha predicted density still overshoots truth). Deployable proxies land within one alpha grid step: CV is best at Size10, BIC at Size100; bracketing CV and BIC covers the oracle.
- **H2 (include-self controls persistence) - supported, with a twist.** Self-only persistence explains ~59% of next-step level at both sizes. Permuting the self predictor removes the include-self advantage (so persistence does real control work), but a clean residualized model reproduces only the small Size100 gain (+0.019 of +0.030), not the large Size10 gain (+0.016 of +0.166) - so part of the benefit comes from joint estimation, not simple self-variance removal. Self-dominance grows with scale (ratio 8.9 -> 117).
- **H3 (fusion via complementary errors) - supported.** Base-method rank correlation is lower at Size100 (0.37) than Size10 (0.43), and fusion true positives carry more multi-method support than false positives (2.25 vs 1.59 at Size100). Fusion promotes multi-method-agreed edges rather than averaging noise.
- **H4 (edge vs topology) - supported.** Spearman(AUPR, top-hub overlap) is weak (0.28 Size10, 0.11 Size100); topology recovery is a partly separate objective from AUPR.
- **H5 (level beats delta/derivative) - supported.** Tree level AUPR strongly beats delta at both sizes; var(delta)/var(level) ~= 0.38 (differencing strips ~62% of the shared predictable variance), and delta ~= derivative (rank corr ~0.98-1.0) on the constant DREAM4 time grid.

General lessons (cautious, tied to results): regularization should reflect sparsity/sample size; autoregressive terms are useful controls but can dominate; fusion helps only with complementary errors; predictive ranking and structural recovery are different goals; target formulation changes signal-to-noise; validation must include regime shifts.

## 12. Calibrated Confidence (deployable, gold-free)

This experiment built a deployable pipeline that selects sparse regularization and builds edge confidence without using gold labels (gold is used only for evaluation).

- **Deployable alpha selection works.** BIC is the best rule overall (mean AUPR gap to oracle 0.014), CV next (0.023). CV matches the oracle exactly at Size10, BIC at Size100, and deployable selection retains 96-100% of the oracle sparse model's AUPR. Practical rule: CV at small/dense scale, BIC at large/sparse scale.
- **Confidence fusion is regime-dependent.** At Size10 the single deployable sparse model wins (AUPR 0.640 vs best confidence 0.609); at Size100 equal-weight `fusion_borda` of the deployable sparse + tree + correlation wins (AUPR 0.183, precision@10 0.72) over every single method.
- **Confidence rankings are meaningfully calibrated.** Higher-confidence bins have much higher empirical true-edge rates (top bin ~8-11x the bottom bin; positive confidence-vs-true-rate Spearman). ECE magnitudes are large because raw scores are not probabilities, so deployment should threshold by confidence rank, not treat the score as P(edge).
- **Topology needs a separate decision layer.** Edge ranking, top-k precision, hub recovery, and reciprocal-direction control have different deployable winners. A fixed topology penalty drives the Size100 reciprocal false-positive pair rate to 0 (by suppressing reciprocal pairs) and gives the best Size100 precision@10 (0.76) at a small AUPR cost.

Net: the regime-dependent, gold-free pipeline is reportable as a calibrated-confidence methodology - it is not yet a single dominant method, and still needs an official dynGENIE3 baseline and GNW validation.

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
- The mechanism audit makes those reframings causal/explanatory. Alpha is a density knob (best alpha rises as truth sparsens) and deployable CV/BIC proxies pick within one grid step. Include-self helps by controlling persistence (a self-permutation control removes the benefit), but residualizing self does not reproduce the Size10 gain, so joint estimation matters. Fusion wins precisely where base methods are least correlated and true positives draw multi-method support. Edge AUPR and topology recovery are only weakly correlated, and level beats delta/derivative because differencing removes the shared predictable variance (var(delta)/var(level) ~ 0.38; delta ~ derivative on the uniform grid). These read as general statistical lessons, with the specific numbers being DREAM4-specific.
- The calibrated-confidence experiment turns the lessons into a deployable, gold-free pipeline. Alpha can be chosen without gold labels (CV/BIC retain 96-100% of oracle AUPR), edge confidence from equal-weight method agreement is meaningfully calibrated (high-confidence bins have ~8-11x the true-edge rate of low-confidence bins), and the best method stays regime-dependent (deployable sparse at Size10, fusion at Size100). Topology objectives need a separate decision layer; a fixed topology penalty zeroes reciprocal false positives (by suppressing reciprocal pairs) and maximizes Size100 precision@10.
- The stability+orientation diagnostics (experiment 17) sharpen and partly overturn the story, with paired-over-networks CIs because n=5 is underpowered. (a) The error is **skeleton detection, not orientation**: orientation-accuracy-given-skeleton is ~0.88-0.96 (vs exactly 0.50 for a symmetric static-correlation control) and the undirected-vs-directed AUPR gap is small, so the recurring reciprocal false positives are mostly false *pairs*, not mis-oriented true edges. (b) A **theory / square-root-LASSO penalty (proportional to sqrt(log p / n)) matches or beats the grid oracle at Size100** (paired delta +0.006, CI excludes 0), so alpha is sample-complexity-predictable, not magic. (c) Fusion's Size100 gain is **genuine complementarity** (cross-method - within-method-bootstrap = +0.068, CI [0.049, 0.084]; ensembling the same method does not help), absent at Size10. (d) **Formal stability selection does not support the original stability-selection thesis**: the Meinshausen-Buhlmann false-positive bound is valid but too loose to be informative at p>>n/n~200, and the selection-probability ranking underperforms a single CV/theory-tuned fit.
- The BEELINE Curated diagnostics (experiment 18) port the *exact* experiment-17 battery to a **real single-cell benchmark in the opposite regime** (n cells >> p genes; exact directed ground truth on GSD/HSC/VSC/mCAD, 4 models x 5 replicates, paired CIs over replicates). It tests which DREAM4 conclusions survive a regime change. (a) **The "skeleton-bound, orientation is free" conclusion is regime- and network-specific**: static single-cell orientation-accuracy-given-skeleton is weaker and far more variable (mean ~0.6 vs DREAM4's tight 0.88-0.96; GSD collapses to <=0.50, VSC reaches 1.00), while the symmetric correlation control sits exactly at 0.50 everywhere as it must. GSD's collapse tracks its reciprocal-heavy truth (18/76 true edges bidirectional), where orientation is partly ill-posed. (b) **The theory-predictable penalty holds and intensifies**: with n>>p the optimal alpha is tiny, CV/BIC land within 0.002-0.06 of oracle, square-root LASSO is sensible (beats oracle on HSC, +0.015, CI excludes 0), and the density-prior selector overshoots badly. (c) **Fusion is regime-dependent**: cross-method complementarity helps only the low-signal dense case GSD (+0.019, CI [0.016,0.022]), is neutral on VSC, and *hurts* HSC and mCAD - it does not transfer as a general win. (d) **The stability-selection negative result transfers cleanly**: the MB bound is still far too loose, selection-probability precision approximately equals edge density (no signal separation), and ECE ~0.60. A `--quick` GSD-only pass had over-claimed a universal orientation "collapse"; the full 4-model run corrects this to "weaker and network-dependent," logged as a lesson against generalizing the two most network-dependent diagnostics from one graph.
- The interventional benchmark scouting (experiment 19) picks the next regime after the DREAM4 (temporal) and BEELINE (static observational) ladder: **interventional / perturbation data, where direction is identifiable by design**. After verifying candidates on the web, **CausalBench** (Chevalley et al.; `pip install causalscbench`; Replogle RPE1 ~163k cells / 383 interventions and K562 ~163k / 622) is chosen because it is built around **held-out interventional evaluation** (mean Wasserstein of a child under parent-knockdown vs control, plus a Mann-Whitney false-omission-rate), which sidesteps the proxy-reference problem rather than importing it. The key methodological point: most exp17/18 diagnostics transfer, but **the observational orientation-given-skeleton metric must be REPLACED** by an interventional-asymmetry test (predict A->B iff perturbing A moves B more than perturbing B moves A; requires both endpoints perturbed; exploratory because of indirect/compensation/off-target/cell-state effects). To de-risk exp20 with zero download, the experiment ships an `InterventionalDataset` adapter, a synthetic linear-SEM fixture, and a dry-run: on the positive control the rebuilt diagnostic orients at **1.000** vs the observational control's **0.500**, and true edges separate from false by +5.0 Wasserstein. No large data was downloaded.
- The CausalBench RPE1 diagnostics (experiment 20) are the **first run on real interventional data** (Replogle/Weissman RPE1 Perturb-seq; 651 perturbed&measured genes, 139,825 cells, 11,485 controls; loaded from the raw 8.7 GB genome-wide h5ad via a memory-efficient chunked `load_replogle_raw_h5ad`). They **complete the regime ladder** and confirm the central thesis empirically. (a) **Orientation becomes identifiable under intervention**: 60.6% of both-perturbed pairs have a decisive interventional asymmetry (effect(A→B) vs effect(B→A) beyond a control-null), versus 0.5 (undecidable) for static observational scores , direction is identifiable here in a way it was not on BEELINE. (b) **Observational orientation is anti-correlated with interventional direction** (agreement 0.329 over ~5k pairs, <0.5): static single-cell edge direction actively misleads. (c) **Observational co-expression is a weak predictor of interventional response**: against a control-null interventional reference the transfer AUROC is only 0.571 (correlation) / 0.506 (sparse), and AUPR sits near the 0.82 density floor , because Perturb-seq responses are broad (direct+indirect+global), so the reference is dense and AUPR is uninformative; AUROC is the appropriate metric and it is weak. (d) **The theory-predictable penalty transfers**: theory α=0.063 lands between CV (0.05) and BIC (0.1). Caveats: decidability is "a direction can be decided," not verified accuracy (no exact truth); the interventional reference is "any measurable shift," not a sparse direct-causal graph (density 0.82); the >100-cell filter is looser than CausalBench's strong-perturbation filter.
- The perturbation-response geometry experiment (experiment 21) uses a response-geometry framing: an intervention produces a displacement vector (gene perturbation -> expression-response delta D[g,j]), and the geometry of those deltas is the object, with edges as one compressed explanation of D. It applies a delta-subspace spectrum, split-half stability, and global-mode removal to the real RPE1 response matrix (651×651). Findings: (a) QC , 99.7% of self-knockdown responses are negative, so the matrix is real biology; (b) the response is **low-rank/global-mode dominated** (top SVD mode = 53% of variance, rank@90% = 121/651) , this *is* exp20's dense-reference problem, now quantified as a real broad transcriptional response; (c) ~50% of perturbation responses are split-half reproducible (a trustworthiness filter); (d) **the headline , interventional orientation is REPRODUCIBLE across independent cell halves**: the asymmetry-implied direction agrees 0.70 across halves (vs 0.5 chance) over ~118k pairs, upgrading exp20's "decidability" to ground-truth-free *verified* directionality; (e) **observational co-expression barely aligns with interventional response** (Spearman ρ≈0.12 correlation / 0.04 sparse), and direct-effect filtering does not rescue it , the sharpest statement that observational GRN inference ≠ causal structure. Note: blind top-mode removal only modestly sharpens density (0.695->0.632), increases per-row diffuseness, and slightly lowers orientation reproducibility (0.70->0.62), since the dominant mode carries some directional signal. A cleaner direct-effect operator (regress out explicit global covariates) is the exp22 target.
- The covariate-aware direct-effect experiment (experiment 22) tests whether the broad response is a removable confound or real biology, using interpretable covariates instead of blind SVD deletion. Findings (real RPE1, 651×651): (a) **the global mode is a real cell-cycle/proliferation program** , its top genes are CCNB1, MCM3, RRM2, DNMT1, H2AFZ, NASP, CENPW, tubulins; gene-side mode correlates with abundance (ρ=0.60) but not with knockdown strength (0.04) or #cells (−0.18), so it is biology concentrated in high-abundance genes, not a technical artifact; (b) **covariate-aware cleaning does NOT separate direct from broad** , removing the shared program hurts (stability 0.51→0.38, it is real signal), and removing amplitude covariates (knockdown strength, #cells) is ~neutral (stability 0.51→0.55), so the broad response is intrinsic and not residualizable; (c) **no observational structure explains interventional response well** , Spearman alignment ranks correlation (0.13) > sparse (0.04) > **GENIE3 ≈ 0**, i.e. a standard GRN method has essentially zero alignment with real interventional effects (the project's strongest cautionary result for observational inference); (d) the verifiable cross-split orientation result (~0.64) is robust across raw and cleaned targets. Constructive turn: stop trying to delete the cell-state axis , model it as a latent factor and infer structures that explain the interventional response itself, not observational co-expression.
- The response inverse / deconvolution experiment (experiment 23) tests the "solve the flow field backward for the stick" idea: a sparse direct operator W generates the total response by propagation, D = (I−W)⁻¹−I, with exact inverse W = I−(I+D)⁻¹. **Mandatory synthetic control passed** (noiseless recovery exact; inverse beats raw |D| at recovering direct edges through ~25% noise , the machinery is correct when the linear model holds). **On real RPE1 it does not help** (verdict MIXED→negative, as pre-registered): inverse operators are sparser and strip the global mode but are *less* split-half stable (edge rank-corr 0.345→0.09–0.19), do not improve direction reproducibility (0.62→0.52), and do not reconstruct held-out response better; a sparse Lasso deconvolution's apparent 0.999 direction-reproducibility is a zeros-agree-with-zeros artifact (true reconstruction ≈0). Conclusion: real perturbation response is not well-described by a simple linear (I−W)⁻¹ generator, so inverse-deconvolution is ruled out as a center.
- The response-transferability experiment (experiment 24) asks the most optimistic remaining question: can a held-out perturbation's effect be predicted better from the shared structure of the *other* perturbations than from its own noisy estimate (leave-one-perturbation-out, gene-specific/residual metric)? **Negative** (disagreed with the pre-registered ~50/50 hope): every low-rank "denoiser" underperforms self-only (residual cosine: self-only 0.41 vs best low-rank 0.34). The useful point: self-only residual cosine 0.41 confirms each perturbation's gene-specific response *is* reproducible, it is just **individualistic, not a shared low-rank code**. Together exp 22/23/24 are three convergent negatives that locate the recoverable signal in **individual perturbations and their pairwise direction**, not in any shared global/low-rank structure , which is itself a finding about where to dig and where not to.
- The counterfactual factor-atlas experiment (experiment 25) implements a cross-project idea (discover reusable sub-features that cut across class labels; a feature is "core" only if removing it breaks the class AND adding it to a rival converts it , otherwise it is a transferable nuisance/shortcut). Methodology-first: **Part A is a synthetic positive control with planted ground truth** , discovery recovers the factors (ARI 1.0), the counterfactual test marks the core factor as core (necessity 0.04, sufficiency 0.96), **sees through a shortcut** spuriously correlated with a class (core_score 0.0006), and projecting out the discovered nuisance directions makes a classifier generalize to flipped class×factor combinations (held-out accuracy 0.89 raw → 1.00 factored). So the idea and the test are faithful and work where factors are separable. **Part B applies the validated tool to RPE1 genes and finds it does NOT cleanly transfer**: the dominant shared program (cell-cycle, 53% of variance) is entangled with real function, so removing it does not reveal a more reproducible "true function" core (verified across 15 k/seed settings: mean module-ARI gain −0.12, residual wins only 33% of the time , a single lucky seed suggested otherwise and the multi-seed check killed it). Summary: the sub-feature/counterfactual idea works where the nuisance is genuinely orthogonal to identity (synthetic, separable-factor settings), and partially breaks for genes because the biggest shared axis *is* part of the biology. New tested `analysis/factor_atlas.py` module.

## Current Recommendation

The defensible current position is that dynamic GRN inference on DREAM4 is regime-dependent rather than owned by one method. Concretely:

- Keep `dynamic_lasso_level_include_self_a0_03` as a Size10 finding only; it does not scale to Size100.
- Treat alpha as a calibrated, density-tracking knob, not a fixed value. Use stronger regularization on larger, sparser networks (best alpha 0.03 at Size10, 0.1 at Size100), and select it with a deployable proxy (cross-validation at small scale, BIC/density-prior at larger scale) rather than gold-standard tuning.
- At Size100, prefer rank fusion of complementary sparse + tree + correlation evidence (best AUPR, best precision@k) and keep level GENIE3 as the AUROC reference; do not use dynGENIE3-style delta/derivative tree targets, which hurt because differencing strips the shared predictable signal.
- Treat self-persistence as a model-stability control, not directed-edge evidence: it explains ~59% of next-step level and its coefficient dominates (ratio up to ~117), and a self-permutation control confirms it carries the include-self benefit.
- Stop investing in orientation/reciprocal-penalty machinery: the diagnostics show the binding error is **skeleton detection** (sample-complexity-limited), not orientation. Set the LASSO penalty from theory (square-root LASSO, ~sqrt(log p / n)) rather than grids. Treat the original stability-selection thesis as **tested and not supported in its strong form** on DREAM4 (loose bound, underperforms a single tuned fit) , keep stability frequencies only as a confidence object, not a reliability claim.

The original stability-selection thesis (stability-aware ranking improves reliability) has now been adjudicated rather than chased: experiment 17 tested it with formal stability selection and found it not supported at these sample sizes. The defensible reframed thesis is: **directed GRN inference here is skeleton- and sample-complexity-limited; the penalty is theory-predictable; and fusing complementary evidence (not ensembling one method) helps in the hard regime.** The deployable calibrated-confidence pipeline (experiment 14) stands, with the caveat that it is a methodology, not a single dominant method.

The next concrete steps are: (1) run the same three diagnostics (skeleton-vs-orientation, theory-alpha, fusion-complementarity) on the real **BEELINE** single-cell data now in `data/raw/`, since the decomposition and the symmetric control transfer directly; (2) since the error is skeleton/sample-complexity limited, the highest-value method tier is finite-sample-controlled selection (model-X knockoffs / scaled-LASSO debiasing) and interventional data (CausalBench) where orientation is identifiable , staged, not all at once; (3) add a literature-faithful dynGENIE3 baseline and the GeneNetWeaver sweeps to test generalization of the theory-alpha and fusion-complementarity findings.

## Caveats

- These are Size10 results across five networks, not final evidence.
- Multifactorial, knockout, knockdown, and time-series files have different statistical meanings.
- Bootstrap resampling with very small sample counts is fragile and should be treated as an audit, not as a validated estimator.
- Metrics can disagree: AUPR, AUROC, and precision@k each reward different ranking behavior.
- Topology metrics can also disagree: hub overlap, degree Spearman, reciprocal counts, and motif counts each capture different structure.
- The lagged audit uses temporal order, but it is still not causal validation.
