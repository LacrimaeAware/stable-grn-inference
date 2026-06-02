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
- Next: prioritize proper lagged time-series inference or a dynGENIE3-style baseline before Size100 scaling; keep stability-GENIE3 as a smaller ablation.

## Phase 3: Sparsity Calibration and Topology-Aware Evaluation

- Study thresholding rules that avoid choosing a network density arbitrarily.
- Added evaluation beyond edgewise scores, including degree Spearman, hub overlap, reciprocal false-positive pairs, reciprocal edge counts, feed-forward loop counts, and true-hub edge precision/recall.
- Compare whether future stability ranking changes both edge metrics and recovered topology.

## Phase 4: GeneNetWeaver Simulation Sweeps

- Use GeneNetWeaver-style synthetic networks after the baseline pipeline is working.
- Vary noise, sampling length, perturbation setting, and network size.
- Measure where stability-aware sparse inference helps, fails, or changes calibration.

## Optional Later Work

- Test graph-wavelet or signal-denoising preprocessing as an ablation.
- Explore Track B as a separate pilot on structured representation learning.
- Consider finance transfer only after the core hidden-network benchmark is solid.
