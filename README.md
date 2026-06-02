# stable-grn-inference

Stability-aware sparse inference for recovering gene regulatory networks from noisy time-series and perturbation data.

This repository is a minimal starting scaffold for a research-engineering project on stability-aware sparse gene regulatory network inference. The current work is focused on DREAM4 Size10 baselines and small stability audits before moving to richer data or larger networks.

## Current Milestone

The first milestone is a reproducible DREAM4 Size10 baseline and stability-audit pipeline:

- Inspect DREAM4 expression and gold-standard file formats.
- Run one-shot edge-ranking baselines across Size10 multifactorial networks.
- Compare correlation, sparse regression, Elastic Net, and random-forest audit baselines.
- Run bootstrap/subsampling stability audits on Size10 multifactorial data.
- Compare Size10 data regimes and a GENIE3-style tree ensemble baseline.
- Add topology-aware evaluation for hubs, degree patterns, reciprocal edges, and simple motifs.
- Compute AUROC, AUPR, and precision@k.
- Use the audit results to decide what should move to richer data.

## Track A Scope

Track A asks whether sparse directed network inference can be made more reliable by ranking edges with explicit stability information, such as bootstrap or subsampling selection frequencies. The initial repo scope is only the baseline pipeline: data loading, one simple inference method, basic edge-recovery metrics, and plots.

Track B, graph-wavelet extensions, and finance applications are not part of the initial repository scope. They may become separate pilots or later extensions after the Track A baseline is working.

## Repository Layout

```text
stable-grn-inference/
+-- docs/
|   +-- project_plan.md
|   +-- data_inventory.md
|   +-- experiment_summary.md
|   +-- update_map.md
+-- src/
|   +-- stable_grn_inference/
|       +-- data/
|       +-- evaluation/
|       +-- inference/
|       +-- stability/
+-- experiments/
+-- results/             # ignored; generated tables/reports
+-- private_docs/        # ignored; private planning notes
+-- requirements.txt
+-- README.md
```

For a consolidated record of completed DREAM4 experiments and current conclusions, see `docs/experiment_summary.md`.
