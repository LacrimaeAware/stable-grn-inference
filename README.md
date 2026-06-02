# stable-grn-inference

Stability-aware sparse inference for recovering gene regulatory networks from noisy time-series and perturbation data.

This repository is a minimal starting scaffold for a research-engineering project on stability-aware sparse gene regulatory network inference. The first objective is to build a clean DREAM4/GeneNetWeaver-style baseline pipeline before adding more ambitious stability, calibration, or simulation experiments.

## Current Milestone

The first milestone is a reproducible baseline pipeline for a real DREAM4-style dataset:

- Locate and download DREAM4 data manually.
- Inspect the expression matrix format.
- Inspect the gold-standard edge format.
- Implement one naive sparse edge-ranking baseline.
- Compute AUROC, AUPR, and precision@k.
- Produce one result plot.

## Track A Scope

Track A asks whether sparse directed network inference can be made more reliable by ranking edges with explicit stability information, such as bootstrap or subsampling selection frequencies. The initial repo scope is only the baseline pipeline: data loading, one simple inference method, basic edge-recovery metrics, and plots.

Track B, graph-wavelet extensions, and finance applications are not part of the initial repository scope. They may become separate pilots or later extensions after the Track A baseline is working.

## Repository Layout

```text
stable-grn-inference/
+-- docs/
|   +-- project_plan.md
+-- src/
|   +-- stable_grn_inference/
|       +-- data/
|       +-- evaluation/
|       +-- inference/
|       +-- stability/
+-- private_docs/        # ignored; private planning notes
+-- requirements.txt
+-- README.md
```
