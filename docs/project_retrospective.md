# Project notes: methods, vocabulary, and findings

Extended technical notes for the repository. The README is the summary; `experiment_summary.md` is the per-experiment results log; this document adds a plain-language statistics reference and the cross-phase findings.

## Scope

The task is directed gene regulatory network (GRN) inference: given gene-expression data, rank candidate directed edges (gene A regulates gene B) and grade the ranking against a known or proxy network. Three data types were used: a simulator with known networks (DREAM4), real single-cell data with curated networks (BEELINE), and real CRISPR perturbation data (CausalBench / Replogle RPE1 Perturb-seq).

## Vocabulary review

### Setup
- Gene expression: a table of cells or samples (rows) by genes (columns).
- Directed edge A to B: the claim that gene A regulates gene B. The direction matters.
- Candidate edges: every directed pair considered (90 for 10 genes, 9,900 for 100 genes).
- Gold standard: the known true edges, used only to grade predictions.

### Grading metrics
- AUROC: probability that a random true edge is ranked above a random non-edge. 0.5 is chance. Insensitive to class imbalance.
- AUPR: rewards ranking rare true edges near the top. Its chance level equals the edge density. The relevant metric when true edges are rare.
- precision@k: fraction of the top k predicted edges that are true.
- EPR (early precision ratio): precision in the top (number of true edges) divided by density. 1.0 is no better than chance.

### Methods
- Correlation: scores A to B by co-movement. Symmetric, so it cannot recover direction.
- LASSO: predicts each gene from the others with an L1 penalty (alpha) that sets most coefficients to zero. Non-zero coefficients are the inferred regulators. Larger alpha gives fewer edges.
- Elastic Net: LASSO with an added L2 term.
- GENIE3: for each target gene, a random forest predicts it from the others; feature importances are the edge scores. Nonlinear, captures interactions.
- Rank fusion: combines several methods' rankings (for example by average rank).
- Stability selection: re-runs a sparse method on many subsamples and scores edges by selection frequency. The Meinshausen-Buhlmann bound gives an expected false-positive count.

### Diagnostics added later
- Skeleton vs orientation: separate "is the pair detected" (undirected) from "is the arrow direction correct" (orientation). A symmetric method scores 0.50 on orientation by construction.
- Square-root LASSO: sets alpha from theory, alpha proportional to sqrt(2 log p / n), with no noise estimate.
- Wasserstein distance: measures how different two distributions are. Used to quantify how much a knockout shifts a target gene's distribution.
- SVD / participation ratio: identify whether a matrix is dominated by a few components.
- Split-half stability: compute a quantity on two random halves of the cells and compare, as a reproducibility check without a gold standard.

### Data terms
- Knockout: a gene fully disabled. Knockdown / CRISPRi: a gene's expression strongly reduced.
- Multifactorial: many small random perturbations at once (a DREAM4 regime).
- Control / non-targeting: cells with no perturbation.
- Observational data: the system is only observed. Interventional data: a gene is actively perturbed and the response is measured.

## Findings by phase

### Phase 1: DREAM4 (simulated)
- Correlation was the strongest single baseline (AUPR 0.33). Tuned LASSO (0.29) and GENIE3-style random forests (0.30) were competitive but did not exceed it.
- Temporal ordering (source at t, target at t+1) raised AUPR from 0.30 to 0.53.
- The best Size10 dynamic sparse model reached AUPR 0.65 / AUROC 0.82 but did not scale to Size100.
- Best LASSO alpha tracked edge density (0.03 at Size10, 0.1 at Size100), and the theory penalty matched cross-validation and BIC within one grid step.
- Stability selection did not beat a single tuned fit.
- Rank fusion improved AUPR at Size100 (0.21 vs 0.17) where methods made complementary errors, and not at Size10.

### Phase 2: BEELINE (real single-cell)
- Static methods transfer; lagged methods do not (no time axis). Reference networks are proxies, so EPR is reported alongside AUPR.
- Orientation accuracy is regime-dependent: 0.81 to 0.96 on DREAM4 time-series, 0.50 to 1.00 (mean 0.60) on BEELINE static data.

### Phase 3: CausalBench / RPE1 (real CRISPR)
- Working set: 651 perturbed genes, about 140,000 cells, 11,485 controls.
- Direction was decidable for 61% of perturbed pairs and reproducible across cell halves at 0.64 to 0.70.
- The perturbation response is dominated by one component (53% of variance) corresponding to a cell-cycle program (CCNB1, MCM3, RRM2, DNMT1, H2AFZ).
- Observational scores weakly predict interventional effects (Spearman 0.13 correlation, 0.04 sparse, 0.00 random forest).
- Inverse deconvolution, low-rank transfer, and a counterfactual feature test each recovered structure on synthetic data and dropped to near-random on RPE1.
- A synthetic phase diagram with known ground truth (experiment 28) separates two failure axes that the RPE1 negatives had conflated: the dominant-mode variance fraction is removable by deflating a clean rank-1 mode, but the specific signal-to-noise ratio is a hard floor below which no method recovers structure at any dominant-mode fraction. RPE1 sits at high dominant-mode fraction and low specific-SNR, so the bottleneck is the noise floor, not the dominant mode, and removing the cell-cycle program cannot expose a cleaner signal that is not there.

## Methods that did not transfer to real perturbation data

The cell-cycle program is a convergent response: most knockouts trigger it, so it accounts for most of the response variance (53%, versus about 4% in control cells). Gene-specific effects are small relative to it and are not linearly separable from it: subtracting the component reduces split-half stability rather than isolating a cleaner signal. Approaches that assume a clean linear or separable structure (matrix inversion to recover direct edges, low-rank prediction across perturbations, removing a global component) therefore work on synthetic data but not on RPE1. Published benchmarks (PerturBench; the CausalBench Challenge) report the same: simple baselines match deep-learning models on this task, and gene-network recovery from this data is unsolved.

## Related mathematics

The recurring sub-problem (separate a dominant shared mode from small specific structure in dynamic data) is studied directly in several fields: state-space and dynamical-systems models (dx/dt = f(x) + u), Dynamic Mode Decomposition and Koopman operator theory (decomposing dynamics into modes with fixed frequency and growth rate), and SINDy (sparse identification of the governing equations from data). The RPE1 cascade is one instance of subtracting a dominant mode to expose smaller structure. Experiment 28 maps when that subtraction can succeed, on synthetic systems with known answers: deflation removes a dominant mode cleanly only above a specific signal-to-noise floor, and RPE1 sits below it.
