# Identifiability, regime, and response geometry in directed GRN inference

*A consolidated report on experiments 17–22. Public-safe; numbers are reproducible from the
scripts under `experiments/`. Generated tables/figures live in the git-ignored `results/`.*

## Executive summary

This project began as "rank directed gene-regulatory edges from expression data and chase
AUPR." Six experiments across three data regimes turned it into a sharper, better-supported
thesis:

> **Directed GRN inference is identifiability-limited, and identifiability is set by the
> data regime — observational vs interventional — not by the estimator.** The durable
> contributions are (1) a portable diagnostic framework that decomposes where inference
> fails, (2) a theory-predictable sparsity penalty that holds across regimes, and (3) direct
> evidence on real CRISPR Perturb-seq that popular observational network methods barely track
> what interventions actually do.

Three headline results, each with the honest caveat attached:

1. **A regime ladder for orientation.** On lagged time-series (DREAM4) edge direction is
   nearly free; on static single-cell snapshots (BEELINE) it collapses to near-chance and is
   highly network-dependent; on real interventional data (CausalBench RPE1) it returns and is
   **verifiable**. Caveat: each regime measures a slightly different quantity; the ladder is a
   qualitative claim backed by per-regime statistics, not a single controlled knob.

2. **Interventional direction is reproducible without ground truth.** On RPE1, the
   intervention asymmetry (perturbing A moves B more than B moves A) agrees **~0.64–0.70**
   across independent halves of the cells, versus 0.5 chance, over >10⁵ gene pairs. Caveat:
   reproducibility is not verified *correctness* (no exact causal graph exists); it shows the
   direction is a stable property of the data, and intervention asymmetry is evidence, not
   proof (indirect effects, compensation, off-target, cell-state shifts).

3. **Observational structure does not track interventional response.** Co-expression, sparse
   regression, and GENIE3 computed from control cells align with the real perturbation
   response at Spearman **0.13 / 0.04 / ≈0** respectively. A standard GRN method (GENIE3) has
   essentially zero alignment with what interventions do. Caveat: the response is dominated by
   a real broad cell-cycle program, so the "honest" alignment metric is rank-based, and the
   comparison is against a perturbation-derived target, not an exact graph.

## 1. The question and why it changed

The original Track A scope was: load data, run one sparse directed inference method, score
edge recovery, and (the thesis) improve reliability by ranking edges with bootstrap/subsampling
**stability** information. Experiment 17 tested that thesis directly and **did not support it**
in its strong form (below). More importantly, the diagnostics revealed that the binding
difficulty is not "which scorer" but **what the data regime can identify**. Experiments 18–22
followed that thread across regimes and onto real interventional data.

## 2. The diagnostic framework (the portable product)

Rather than a leaderboard, the project built a battery that decomposes *where* inference fails,
with paired confidence intervals because sample sizes are small. The same definitions are
reused across regimes by construction (later experiments import the earlier diagnostic code):

- **Skeleton vs orientation.** Undirected edge recovery (collapse i→j, j→i) vs directed
  recovery; and **orientation-accuracy-given-skeleton**: among true edges whose undirected
  pair is detected, the fraction scored in the correct direction. A symmetric control
  (correlation) sits at exactly 0.50 by construction — a built-in sanity check.
- **Sample-complexity penalty.** Square-root / scaled LASSO sets the ℓ₁ penalty from theory
  (λ ∝ √(2 log p / n)) with no σ estimate, compared against CV, BIC, and a grid oracle.
- **Complementary-evidence fusion.** A 3-arm test separating genuine cross-method
  complementarity from within-method ensembling.
- **Formal stability selection.** Meinshausen–Bühlmann false-positive bound + regime-aware
  subsampling (trajectory-level for time-series, cell-level for single-cell).
- **Response geometry** (experiments 21–22): the perturbation-response matrix
  D[g,j] = E[X_j | do g] − E[X_j | ctrl]; its SVD spectrum (low-rank/global modes), split-half
  stability, interpretable program/covariate decompositions, and a ground-truth-free
  cross-split orientation-reproducibility test.

All response-geometry pieces are general library functions (`src/.../data/interventional.py`)
tested on synthetic fixtures, so the test suite never depends on large downloads.

## 3. The regime ladder (central finding)

| regime | data | orientation-given-skeleton | reading |
| --- | --- | --- | --- |
| DREAM4 Size10/100 | lagged time-series | **0.88–0.96** (vs 0.50 control) | temporal precedence ≈ hands you direction |
| BEELINE Curated | static single-cell, exact labels | **0.50–1.00, network-dependent** (GSD ≈0.4 collapses, VSC ≈1.0); mean ≈0.6 | a snapshot can't orient; depends on graph structure |
| CausalBench RPE1 | real CRISPRi Perturb-seq | **decidability 0.61; cross-split repro ≈0.64–0.70** | intervention restores *and verifies* direction |

The DREAM4 conclusion "error is skeleton-bound, orientation is essentially free" (experiment
17) is therefore **regime-specific** (experiment 18): it held only because lagged data encodes
time. On static observational data, orientation is a genuine near-non-identifiable problem; on
RPE1, observational edge direction is even **anti-correlated** with the interventional direction
(agreement 0.33 < 0.5), i.e. it actively misleads.

## 4. What transfers across all three regimes

| claim | DREAM4 | BEELINE | CausalBench RPE1 |
| --- | --- | --- | --- |
| theory/√-LASSO penalty ≈ CV/BIC/oracle | ✓ (+0.006 AUPR vs oracle, CI excludes 0) | ✓ (beats oracle on HSC) | ✓ (theory α 0.063 between CV 0.05 / BIC 0.1) |
| strong stability-selection thesis | ✗ not supported (bound too loose at p≫n) | ✗ (precision ≈ density) | (not retested; expected ✗) |
| fusion = genuine complementarity | ✓ at Size100 (+0.068, CI [0.049,0.084]) | regime-dependent (helps GSD only) | n/a |
| orientation from static scores | n/a (had time) | weak / variable | **anti-correlated (0.33)** |
| orientation under intervention | — | — | **identifiable + reproducible** |

The penalty result is the most robust positive: the right amount of regularization is
**predictable from sample-complexity theory**, not magic, in every regime tested.

## 5. Real interventional data: three results (experiments 20–22)

Data: raw Replogle/Weissman RPE1 CRISPRi Perturb-seq (dense 247,914 cells × 8,749 genes, ~8.7
GB), loaded with a memory-efficient chunked reader that keeps only the perturbed∩measured gene
block, normalizes per cell (UMI + log1p), and filters to perturbations with >100 cells →
**651 genes, 139,825 cells, 11,485 controls** (the control count matches the published RPE1
figure exactly).

**5.1 Orientation is identifiable and verifiable.** Decidability (a direction can be chosen
from the effect asymmetry) is 0.61 of perturbed pairs; cross-split reproducibility (the chosen
direction agrees across independent cell halves) is ~0.64–0.70 over >10⁵ pairs — far above the
0.5 chance line. This is the first point in the project where direction could be *verified*
without an answer key.

**5.2 The broad response is real cell-cycle biology, not a removable confound.** The response
matrix is low-rank: the top SVD mode explains **53%** of variance. That mode is a coherent
**cell-cycle / proliferation program** (top genes CCNB1, MCM3, RRM2, DNMT1, H2AFZ, NASP,
CENPW, tubulins); it correlates with gene abundance (ρ=0.60) but not with knockdown strength
(0.04) or cell count (−0.18). QC confirms the matrix is real biology: **99.7%** of
self-knockdown responses are negative (CRISPRi lowers the targeted gene). Attempts to remove
the broad component fail cleanly: deleting the shared program *hurts* (split-half stability
0.51→0.38, because it is real signal), and regressing out amplitude covariates is neutral. So
"direct vs broad" is not a simple subtraction — the broad axis is intrinsic biology.

**5.3 Observational methods barely predict interventional response.** Alignment (Spearman of
|inferred score| from control cells vs |interventional response|): **correlation 0.13 > sparse
0.04 > GENIE3 ≈ 0**. Even the best is weak; GENIE3 — a widely used GRN method — has essentially
zero alignment with what the perturbations actually do. This is the project's strongest
cautionary statement: **observational network inference is not causal inference.**

## 6. The Track A ↔ Track B synthesis

A parallel project (Track B) studies intervention-response geometry in *representation* space:
Δ_k(x) = Φ(T_k x) − Φ(x) for a controlled visual factor T_k. Track A has the same object in
*expression* space: Δ_g = E[X | do g] − E[X | ctrl]. The shared spine is

> **intervention → displacement vector → geometry of hidden structure.**

The bridge is *not* "apply wavelets/scattering to genes" — gene vectors have no domain geometry,
so that would be forced. The bridge is the **mindset**: study the geometry of intervention
displacements (rank, stability, asymmetry, composition), and treat an inferred network as one
compressed *explanation* of that geometry rather than the object itself. Experiments 21–22
imported Track B's delta-subspace, stability, and decomposition tools and produced the
verifiable-orientation metric and the cell-cycle-program finding above. Graph-geometry tools
(graph wavelets) remain deferred until a trustworthy gene graph or response manifold exists.

## 7. Limitations (honest)

- **Small graphs / sample sizes** in places: DREAM4 n=5 networks; BEELINE 5–19 genes
  (mCAD near-degenerate). Paired CIs are used precisely because power is low.
- **No exact causal ground truth** on real Perturb-seq. The RPE1 "reference" is "any
  measurable shift," which is dense (0.82) because responses are broad; orientation results are
  reported as decidability/reproducibility, not verified accuracy.
- **A looser perturbation filter** (>100 cells) than CausalBench's strong-perturbation set, and
  **subsampling** (4,000 control / 400 per perturbation) for tractability — seeded and
  reproducible, but estimates carry sampling noise.
- **One cell line** (RPE1); K562 is an untested independent replicate.
- The regime ladder compares **different datasets**, not one dataset with a regime knob; the
  claim is qualitative-but-quantified, not a controlled experiment.

## 8. What's next (candidates, not commitments)

- **Condition on the cell-state axis** rather than delete it: estimate effects within
  proliferation-matched strata, then test whether a structure inferred to explain the
  *interventional* response finally beats correlation.
- **Harden the existing claims:** bootstrap CIs on orientation/alignment, K562 as an
  independent replicate, and CausalBench's own evaluation harness.
- Graph-geometry / multiscale tools only **after** a trustworthy gene graph exists.
- Explicitly **out of scope for now:** RL, large neural nets, and forcing wavelets/scattering
  onto unordered gene vectors.

## Appendix — experiment index (this arc)

| exp | title | key numbers |
| --- | --- | --- |
| 17 | DREAM4 stability + orientation diagnostics | orientation 0.88–0.96 vs 0.50; √-LASSO ≈ oracle (+0.006, CI excl. 0); fusion +0.068 [0.049,0.084]; stability thesis not supported |
| 18 | BEELINE Curated cross-regime validation | orientation 0.50–1.0 network-dependent (GSD≈0.4, VSC≈1.0); fusion helps GSD only; stability negative transfers |
| 19 | interventional benchmark scouting | chose CausalBench (held-out interventional eval); built InterventionalDataset + synthetic dry-run (rebuilt-orientation 1.0 vs 0.5 control) |
| 20 | CausalBench RPE1 interventional diagnostics | 651 genes / 139,825 cells; orientation decidability 0.61; observational direction anti-correlated 0.33; transfer AUROC 0.57/0.51; theory α 0.063 |
| 21 | perturbation-response geometry | top-1 mode 53% var; self-knockdown 99.7% negative; split-half cosine 0.51; cross-split orientation 0.70; observational alignment ρ 0.12/0.04 |
| 22 | covariate-aware direct-effect geometry | global mode = cell-cycle program; cleaning fails to separate direct/broad; correlation 0.13 > sparse 0.04 > GENIE3 ≈ 0; orientation ~0.64 robust |

Reproduce any experiment with `PYTHONPATH=src .venv/Scripts/python.exe -B experiments/<dir>/<script>.py`
(`--quick` for a fast pass). Full suite: `python -B -m unittest discover -s tests` (133 tests).
