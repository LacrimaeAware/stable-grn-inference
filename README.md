# stable-grn-inference

Stability-aware sparse inference for recovering gene regulatory networks from noisy time-series and perturbation data.

This repository is a minimal starting scaffold for a research-engineering project on stability-aware sparse gene regulatory network inference. The work so far is focused on DREAM4 Size10 baselines and small stability audits, plus a first DREAM4 Size100 scaling check on the leading dynamic sparse candidate.

## Current Milestone

The first milestone is a reproducible DREAM4 Size10 baseline and stability-audit pipeline:

- Inspect DREAM4 expression and gold-standard file formats.
- Run one-shot edge-ranking baselines across Size10 multifactorial networks.
- Compare correlation, sparse regression, Elastic Net, and random-forest audit baselines.
- Run bootstrap/subsampling stability audits on Size10 multifactorial data.
- Compare Size10 data regimes and a GENIE3-style tree ensemble baseline.
- Add topology-aware evaluation for hubs, degree patterns, reciprocal edges, and simple motifs.
- Run a first lagged Size10 time-series audit using within-trajectory source(t) to target(t+1) samples.
- Run a broad dynamic model batch comparing sparse linear, tree, MLP, stability, rank-fusion, and preprocessing variants.
- Validate the strongest dynamic sparse-linear candidate for alpha sensitivity, self-persistence, bootstrap behavior, reciprocal errors, and topology metrics.
- Scale the dynamic sparse candidate to DREAM4 Size100 time-series data as a first scaling check.
- Compare dynGENIE3-style trees, alpha-calibrated sparse models, and rank fusion across Size10 and Size100, and scaffold a GeneNetWeaver simulation-sweep design.
- Run a mechanism audit that explains the findings: alpha as a density knob, include-self as a persistence control, fusion via complementary errors, edge-vs-topology disagreement, and level-vs-delta target quality.
- Build a deployable, gold-free calibrated-confidence pipeline: select alpha without gold labels (CV/BIC), rank edges by equal-weight method-agreement confidence, check calibration, and keep topology objectives in a separate decision layer.
- Begin a modern single-cell direction: scout modern GRN benchmarks (choose BEELINE) and add a BEELINE adapter (`data/beeline.py`) so the pipeline ingests single-cell datasets (TF→gene candidates, proxy references, EPR), smoke-tested on a synthetic fixture.
- Compute AUROC, AUPR, and precision@k.
- Use the audit results to decide what should move to richer data.

The Size100 scaling check is a cautionary result: the Size10 candidate `dynamic_lasso_level_include_self_a0_03` does not reproduce its advantage on Size100, so it is kept as a Size10 finding rather than promoted to a main method. The follow-up calibration/fusion audit shows the picture is regime-dependent: the best sparsity level tracks network density, dynGENIE3-style delta/derivative tree targets do not help, and rank fusion of complementary evidence is the strongest Size100 AUPR method. The mechanism audit then explains these results (with deployable CV/BIC alpha selection, a self-permutation control, and an AUPR-vs-topology analysis) and draws general statistical lessons. The calibrated-confidence experiment turns them into a deployable, gold-free pipeline that selects alpha without gold labels (retaining 96-100% of the oracle sparse model's AUPR), ranks edges by calibrated method-agreement confidence, and keeps topology as a separate decision layer; the best method is regime-dependent (deployable sparse at Size10, fusion at Size100), so it is reportable as a methodology rather than a single dominant method. No official dynGENIE3 package is installed, so the delta/derivative tree methods are dynGENIE3-style. See `docs/experiment_summary.md` for details.

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
