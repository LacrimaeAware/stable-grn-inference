# Project Plan

This public plan summarizes the intended Track A progression without depending on private planning notes.

For a consolidated record of completed experiments, metrics, and current interpretation, see `docs/experiment_summary.md`.

## Phase 1: DREAM4 Baseline Pipeline

- Completed initial DREAM4 Size10 data inventory for local expression and gold-standard files.
- Implemented basic loaders for DREAM4 expression matrices and directed gold-standard edge labels.
- Implemented one-shot Size10 multifactorial baselines for correlation, LASSO, Elastic Net, and random-forest feature importance.
- Computed basic edge-recovery metrics: AUROC, AUPR, and precision@k.
- Saved generated result tables under ignored `results/tables/`.

## Phase 2: Stability Layer

- Started with a small-data Size10 stability audit.
- Added bootstrap and subsampling index generation.
- Added resampled edge-score summaries: mean score, mean reciprocal rank, top-k frequency, and selection frequency.
- Compared one-shot and stability-ranked correlation, LASSO, Elastic Net, and random-forest audit baselines.
- Current result: stability correlation improves mean AUPR over one-shot correlation, while one-shot correlation still wins mean AUROC.
- Extended the Size10 audit across multifactorial, knockout, knockdown, and time-series regimes using same-time association scoring.
- Current data-regime result: stability correlation improves mean AUPR in all four regimes, while LASSO stability is mixed; one-shot LASSO performs best by mean AUPR on knockouts.
- Added a clearer GENIE3-style target-wise tree ensemble baseline using random forests and Extra Trees.
- Current GENIE3 result: GENIE3 improves mean AUPR over one-shot correlation in all four Size10 regimes and beats stability correlation by mean AUPR in knockouts, multifactorial, and same-time time-series scoring. Stability correlation remains competitive, especially on knockdowns and AUROC.
- Added the first topology-aware evaluation layer for hub recovery, degree-pattern recovery, reciprocal-direction errors, and simple motif counts.
- Current topology result: edge-ranking gains do not consistently translate into topology recovery. Correlation is still strong for out-degree hubs, stability correlation is mixed for topology, and GENIE3's topology gains are partial.
- Added a first lagged time-series audit that splits trajectories by `Time` resets and builds within-trajectory source(t) -> target(t+1) samples.
- Current lagged result: lagged methods substantially improve edge AUROC/AUPR over same-time references. `lagged_genie3_random_forest` is the strongest first temporal baseline by mean AUPR/AUROC, while topology and reciprocal-direction recovery remain mixed.
- Added a broad dynamic model batch comparing target types, self-predictor modes, tree models, sparse linear models, MLP permutation importance, stability, rank fusion, and light preprocessing.
- Current dynamic batch result: sparse linear level-target models with `include_self_predictor_no_self_edge` win mean AUPR/AUROC. Rank fusion, simple MLP, and moving-average preprocessing do not beat the best single dynamic model.
- Added a focused validation audit for the dynamic sparse-linear include-self result.
- Current dynamic sparse validation result: `dynamic_lasso_level_include_self_a0_03` remains the best Size10 temporal sparse candidate by mean AUPR/AUROC, improves reciprocal false-positive behavior relative to lagged GENIE3 references, and has partially encouraging topology metrics. However, it wins per-network AUPR on only 2 of 5 networks and relies heavily on self-persistence, so it needs validation on richer data or a literature-faithful dynGENIE3 comparison.
- Added the first DREAM4 Size100 time-series scaling audit for the dynamic sparse candidate.
- Current Size100 scaling result: the Size10 winner does not scale. `dynamic_lasso_level_include_self_a0_03` ties lagged correlation on mean AUPR, trails it on mean AUROC, wins 0 of 5 Size100 networks, and loses its reciprocal-direction advantage (pair rate 1.00 vs ~0.20 at Size10). Stronger regularization (`a0_1`) is the best sparse setting at Size100, lagged GENIE3 wins mean AUROC and every per-network AUROC, and self-persistence is even more extreme (self/non-self ratio ~26 vs ~8.9). The include-self sparse family stays worth studying but should not be promoted as a main method.
- Added a combined dynGENIE3-style baseline, sparsity-calibration, and rank-fusion audit across Size10 and Size100 (experiment 11), plus a GeneNetWeaver sweep design scaffold (experiment 12).
- Current calibration/fusion result: dynamic GRN inference on DREAM4 is regime-dependent. The best alpha tracks network density (0.03 at Size10, 0.1 at Size100), stronger regularization is better at Size100, and include-self still helps after calibration. dynGENIE3-style delta/derivative tree targets hurt versus level GENIE3. Rank fusion of complementary sparse + tree + correlation evidence is the best Size100 AUPR method (`fusion_borda` 0.208, precision@10 0.84) and a fixed reciprocal-direction penalty slightly improves Size100 AUPR and reciprocal false positives; trees still lead Size100 AUROC. No official dynGENIE3 is installed, so tree delta/derivative methods are dynGENIE3-style.
- Added a mechanism/hypothesis audit (experiment 13) that explains the experiment 9-11 winners rather than adding models.
- Current mechanism result: all five hypotheses are supported, with nuances. Alpha is a density knob (best alpha 0.03 -> 0.1 as true density drops ~0.16 -> ~0.02), and deployable CV/BIC proxies choose within one alpha grid step of the oracle. Include-self helps by controlling persistence (a self-permutation control removes the benefit; self explains ~59% of next-step level), but a residualized model reproduces only the Size100 gain, not the Size10 gain, so joint estimation matters. Fusion wins where base methods are least correlated and true positives carry multi-method support. Edge AUPR and topology recovery are only weakly correlated. Level beats delta/derivative because differencing strips the shared predictable variance (var(delta)/var(level) ~ 0.38; delta ~ derivative on the uniform grid).
- Added a deployable, gold-free calibrated-confidence pipeline (experiment 14): gold-free alpha selection, equal-weight agreement confidence, calibration diagnostics, and a topology-aware decision layer.
- Current calibrated-confidence result: deployable alpha selection retains 96-100% of the oracle sparse model's AUPR (BIC best overall, CV best at Size10, BIC exact at Size100). Equal-weight `fusion_borda` of the deployable sparse + tree + correlation wins Size100 AUPR/precision@10 while single deployable sparse wins Size10. Confidence rankings are meaningfully calibrated (top-confidence bins have ~8-11x the true-edge rate of bottom bins). Topology needs a separate decision layer; a fixed topology penalty zeroes reciprocal false positives (by suppressing reciprocal pairs) and maximizes Size100 precision@10. Reportable as methodology, not yet a single dominant method.
- Next: add a literature-faithful (official) dynGENIE3 baseline, then execute the GeneNetWeaver sweeps designed in experiment 12 to test whether the density/alpha, residualization, fusion-complementarity, and calibrated-confidence findings generalize; consolidate the regime-dependent pipeline into a methodology report before opening perturbation/knockout data branches.

## Phase 3: Sparsity Calibration and Topology-Aware Evaluation

- Studied thresholding/sparsity rules that avoid choosing a network density arbitrarily (experiments 11 and 13): an alpha sweep shows the best alpha tracks true edge density and behaves like a density knob, and an oracle-density (top-N-true) evaluation is recorded as a non-deployable diagnostic.
- Added deployable alpha-selection proxies (cross-validation MSE, BIC, a sparsity-prior density heuristic, and bootstrap selection stability) and compared them to the oracle best alpha; CV is best at Size10, BIC at Size100, each within one grid step.
- Added evaluation beyond edgewise scores, including degree Spearman, hub overlap (top-3/5/10), reciprocal false-positive pairs, reciprocal edge counts, vectorized feed-forward loop counts, and true-hub edge precision/recall, plus an explicit AUPR-vs-topology correlation analysis showing they are partly separate objectives.
- Added a reciprocal-direction penalty to rank fusion, a self-residualized sparse model, and a self-permutation control to target persistence and reciprocal false positives.
- Built a deployable, gold-free confidence pipeline (experiment 14): gold-free alpha selectors (CV/BIC/AIC/density-prior/stability), equal-weight agreement confidence, calibration reliability/ECE-style diagnostics, and a topology-aware decision layer with separate winners for edge ranking, top-k precision, hub recovery, and reciprocal-direction control.

## Phase 4: GeneNetWeaver Simulation Sweeps

- Design scaffolded in `experiments/12_gnw_sweep_design/gnw_sweep_design.md` (network sizes 10/30/50/100, trajectory lengths 21/50/100, trajectory counts 5/10/20, noise levels, perturbation regimes, methods, metrics, and success questions).
- GNW is not yet installed; the experiment-11 detector reports availability and execution is not blocked on it.
- Use GeneNetWeaver synthetic networks to vary noise, sampling length, perturbation setting, and network size, and measure where calibrated dynamic sparse inference and rank fusion help, fail, or change calibration.

## Optional Later Work

- Signal-denoising preprocessing is now runnable: PyWavelets is installed (`requirements.txt`), and the experiment 08 wavelet-denoise ablation runs as a real variant (slightly hurts the Size10 RF baseline, AUPR 0.519 vs 0.537 raw). Graph-wavelet / wavelet-scattering preprocessing can be explored next.
- Explore Track B as a separate pilot on structured representation learning. Kymatio is installed for 1D wavelet scattering (use `stable_grn_inference._compat.ensure_scipy_sph_harm()` before importing `kymatio.numpy` on SciPy >= 1.15).
- Consider finance transfer only after the core hidden-network benchmark is solid.
