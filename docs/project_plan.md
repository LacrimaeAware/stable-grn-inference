# Project Plan

This public plan summarizes the intended Track A progression without depending on private planning notes.

## Phase 1: DREAM4 Baseline Pipeline

- Manually locate and download the relevant DREAM4 or DREAM4-style data.
- Inspect the expression matrix format before writing format-specific loaders.
- Inspect the gold-standard edge format before writing scoring code.
- Implement one naive baseline for sparse edge ranking.
- Compute basic edge-recovery metrics: AUROC, AUPR, and precision@k.
- Produce one result plot to confirm the pipeline works end to end.

## Phase 2: Stability Layer

- Add bootstrap or subsampling routines around the baseline estimator.
- Refit the baseline across resampled datasets.
- Convert repeated edge selections into per-edge stability frequencies.
- Compare ordinary edge ranking against stability-ranked edges.

## Phase 3: Sparsity Calibration and Topology-Aware Evaluation

- Study thresholding rules that avoid choosing a network density arbitrarily.
- Add evaluation beyond edgewise scores, such as degree patterns or hub recovery.
- Compare whether stability ranking changes both edge metrics and recovered topology.

## Phase 4: GeneNetWeaver Simulation Sweeps

- Use GeneNetWeaver-style synthetic networks after the baseline pipeline is working.
- Vary noise, sampling length, perturbation setting, and network size.
- Measure where stability-aware sparse inference helps, fails, or changes calibration.

## Optional Later Work

- Test graph-wavelet or signal-denoising preprocessing as an ablation.
- Explore Track B as a separate pilot on structured representation learning.
- Consider finance transfer only after the core hidden-network benchmark is solid.
