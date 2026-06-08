# Literature review: GRN inference, perturbation response, and dynamical recovery

A referenceable record of the external research gathered for this project, so claims can be
checked against the field and other readers (or models) can use the same evidence base. Gathered
through multi-source web search with adversarial verification (each claim voted by three
independent checkers; claims that failed verification are listed separately and not relied on).
Compiled 2026-06; sources are preprints and primary literature as of mid-2025.

## How to read this

- "Verified" items passed 3-of-3 or 2-of-3 adversarial verification against the cited source.
- "Refuted / unverified" items failed verification (could not be confirmed from the cited
  source). They are recorded so they are not mistaken for established facts.
- Effect sizes are quoted where the source gave them.

## 1. Interventional perturbation-response asymmetry as a causal-direction cue

Verified:
- Interventional (CRISPR perturbation) data improves identifiability and removes confounding
  relative to observational data (PMC12579002).
- CausalGRN (biorxiv 2025.12.30.692369) orients an edge A to B when knockdown of A shifts B and
  the evidence is one-sided, on the exact RPE1 data (8,644 genes, 154,577 cells, 1,159
  knockdowns) and K562, after regressing out S/G2M, library size, and mitochondrial fraction.
  It handles the dominant program by adding the wild-type PC1 as a pseudo-gene node, which
  removes thousands of false edges.

Implication: the interventional-asymmetry instinct (this project's net_out, exp 21/26) is a
field-validated cue, not novel as a concept. CausalGRN is a named competitor on the same data,
with a third dominant-mode treatment (pseudo-gene node) distinct from subtraction and whitening.

Refuted / unverified: that the "inspre" method uses a bidirectional ACE matrix and instrumental
variables to exploit perturbation asymmetry (could not be confirmed). The specific numbers
"edge-overlap F1 = 0.17, eigencentrality replication rho = 0.66" failed verification; treat the
edges-do-not-reproduce-but-aggregate-does pattern as corroborated qualitatively (it matches this
project's exp 27), but do not cite those exact numbers.

## 2. Separating a dominant shared program from gene-specific effects (whitening)

Verified:
- Covariance whitening (Cholesky / ZCA-cor) is the single largest boost to CRISPR co-essentiality
  recovery, and smaller post-whitening correlations enrich better for true interactions
  (PMC9707256).
- ZCA-cor is the correct variant: it uniquely keeps whitened variables maximally similar to the
  originals (arXiv:1512.00809).

Caveats: the whitening evidence is DepMap bulk cell-line co-essentiality, not single-cell
Perturb-seq, so transfer is unproven. BaCoN documents the failure mode (whitening manufacturing
spurious proximal pairs by amplifying low-variance directions). Cross-reference exp 28: whitening
fixes dominant-mode dominance, not the SNR floor; co-essentiality is a higher-SNR aggregate
regime, so whitening's wins there are consistent with exp 28 predicting it will not rescue
single-cell RPE1.

## 3. External anchors for validating gene-gene structure

Verified, with effect sizes:
- Replogle 2022 (PMC9380471): correlation-based clustering recovers CORUM complexes at median
  r = 0.61 vs 0.10 background; tracks STRING; DepMap confirms groupings.
- Genome-wide co-essentiality (Wainberg, PMC8763319; 485 lines, 17,634 genes, 93,575 pairs)
  enriches roughly 160-fold for CORUM, 130-fold for hu.MAP, 7.5-fold for STRING under generalized
  least squares.
- DepMap gene-effect is a bias-corrected interventional knockout-fitness score (biorxiv 720243).
- Much real structure is invisible to expression similarity: at STRING confidence >= 900, about
  half the pairs have cosine <= 0.5 (arXiv:2603.20955).

Implication: STRING, CORUM, hu.MAP, and DepMap are validated proxies, stronger for complex
co-membership than for directed regulation. Used in exp 32 (STRING) and available for exp 29
Gate 1. For DIRECTED grading, these are insufficient (mostly undirected); a curated TF-target
resource (TRRUST, DoRothEA) is needed.

## 4. Do nonlinear / deep models beat simple baselines on Perturb-seq?

Verified:
- CausalBench winning solutions were standard ML (LightGBM, mean-difference, boosted trees),
  beating the deep causal method DCDI (arXiv:2308.15395).
- Five foundation models plus GEARS and CPA failed to beat no-change / additive baselines
  (Nature Methods 2025, s41592-025-02772-6).

Refuted / unverified: a claim that a contrastive MLP reached cross-boundary AUC 0.908 vs cosine
0.518 for recovering STRING pairs on K562 (failed verification), so the nonlinear-beats-linear
case is unsupported here. A claim that simpler methods (PIDC, GENIE3, GRNBoost2) definitively
lead BEELINE also failed verification; treat "simple methods are competitive on BEELINE" as
likely but not settled.

Implication: do not expect a deep / nonlinear model to beat simple baselines on this data. The
right comparators are the established simple methods, not chance.

## 5. Dynamical-systems decompositions (DMD / Koopman / SINDy)

Verified:
- SINDy recovered a bacterial competence regulatory network using rational-function libraries
  (Mangan et al. 2016, arXiv:1605.08368). General DMD theory: constant sampling assumed, but
  multiple short trajectories can be concatenated; DMD is fragile under noise / sub-Nyquist.

Gap (important): no surviving claim describes DMD / Koopman / SINDy applied to single-cell or
perturbation expression data and graded against a ground-truth GRN. This is a genuine gap, which
is a novelty opening but also means there is no off-the-shelf protocol or timepoint requirement to
copy. These methods need a time axis (RPE1 snapshots do not have one).

A directly supporting result: a 2021 study (Cell Systems, PMC8441581) shows mean time courses
cannot distinguish no-interaction from bidirectional networks (third moments are needed) and
moment inference needs fine snapshot intervals. This is an independent, quantitative statement of
this project's SNR-floor finding (exp 28) and is good contrast material.

## 6. Datasets for time-resolved GRN inference (verified scout)

Ranked best-first for "time axis + checkable truth":

1. DREAM4 in silico time-series (already local; exp 30 Part B). Exact directed gold standard;
   21 regular timepoints; 5 (size-10) or 10 (size-100) replicates per network. Validates the
   method against exact truth. Simulated, so it does not test real-data SNR. Access: Bioconductor
   networkBMA / DREAM4, Synapse syn3049712.
2. RENGE / GEO GSE213069 (downloaded; exp 32). Real time-series single-cell CRISPR in human iPSCs,
   4 timepoints (days 2 to 5), 23 knocked-out TFs, about 5,000 cells/sample, with a ChIP-seq proxy
   truth (19 genes, threshold 300) described in the paper (not in the GEO download). About 360 MB.
3. BEELINE / BoolODE (already local; exp 31). Exact truth from the generating Boolean/ODE model,
   with a cell-count sweep (100 to 5,000) mirroring the exp 28 SNR axis. Axis is pseudotime.
   Zenodo 10.5281/zenodo.3378975 (279 MB). Code: github.com/Murali-group/Beeline. BEELINE also
   publishes leaderboard AUPR for GENIE3, GRNBoost2, PIDC, SINCERITIES, SCODE on these datasets,
   which are the established baselines to compare against.

Evaluated and deprioritized: DREAM5 (mixed conditions, not clean time courses); Sci-Plex
GSE139944 (single 24 h endpoint, not a time course); generic CausalBench (static).

## 7. Sources

- arXiv:2308.15395 (CausalBench analysis); PMC12579002 (interventional identifiability);
  biorxiv 2025.12.30.692369 (CausalGRN).
- arXiv:1512.00809 (ZCA-cor); PMC9707256 (whitening boosts co-essentiality).
- PMC9380471 (Replogle 2022); PMC8763319 (Wainberg co-essentiality); biorxiv 720243 (DepMap
  gene-effect); arXiv:2603.20955 (STRING vs expression gap).
- Nature Methods 2025 s41592-025-02772-6 (foundation-model perturbation benchmark);
  PMC7098173 (BEELINE).
- arXiv:1605.08368 (SINDy on a regulatory network); PMC8441581 (higher-order moments / SNR floor).

Source caveat: CausalGRN and the STRING-gap paper are preprints; whitening evidence is bulk
cell-line co-essentiality, not single-cell; several anchor validations are K562, not RPE1; the
deep-vs-linear results are a mid-2025 snapshot.

## 9. Recovering order from static data (seriation, diffusion pseudotime)

This corner was researched to ground the idea of rebuilding a latent ordering from static,
unordered data via the geometry of a similarity matrix and its higher powers, then orienting it
with a prior. Verified evidence is thin but on-point; the parts marked open are genuinely
unresolved, not negatively settled.

Verified (Palantir, Setty et al. 2019, Nature Biotechnology, PMC7549125):
- Diffusion-based pseudotime recovers a 1D ordering of cells from static single-cell snapshots by
  running a Markov chain on a diffusion-map nearest-neighbor graph, using multiple diffusion
  components (a generalization of the single Fiedler / second eigenvector).
- The ordering's direction is NOT recovered from the geometry. The start is fixed by an external
  prior (a user-supplied "early" cell plus an assumption of one-way progression). This directly
  confirms the premise "order is recovered, orientation is supplied externally."
- Useful nuance: the terminal ends CAN be auto-detected from the geometry (boundary cells that are
  outliers in the chain's stationary distribution). Only the start needs a prior. Auto-detection
  failed under sparsity in one dataset, so it is not fully robust.
- Palantir was validated by marker-trend consistency and cross-donor reproducibility, and beat
  DPT, PAGA, Monocle2, Slingshot, FateID qualitatively. It reported NO quantitative accuracy metric
  (no correlation against a known timeline). So order-recovery accuracy against truth is, in this
  source, unmeasured.

Established background math (textbook, not separately verified in this pass): spectral seriation
recovers a 1D ordering from a symmetric similarity matrix via the Fiedler vector (second
eigenvector of the Laplacian); for a Robinson similarity matrix the ordering is recovered up to
reversal (Atkins, Boman, Hendrickson 1998). The reversal ambiguity is exactly "order recovered,
direction not."

Open (unaddressed by the verified evidence, so untested here, not disproven):
- Whether higher-order / iterated correlation (correlation of correlation profiles, von Neumann
  diffusion, network propagation on a gene-gene graph) captures real indirect A->C->D structure or
  mostly inflates spurious transitive edges.
- Whether pseudotime / ordering-based GRN methods (SCODE, SINCERITIES, LEAP, GENIE3 on
  pseudotime-ordered cells) beat static co-expression baselines on BEELINE. The project's core
  motivating question; no verified quantitative answer. The field has not clearly shown a win.

Implication: the idea of recovering an order from static geometry is real and established
(diffusion pseudotime). The open part, and the only place a fresh result could come from, is
(a) quantifying order-recovery accuracy against truth (which Palantir skipped) and (b) testing
whether the recovered order actually improves network recovery over static co-expression. Both are
testable on BoolODE, which has a true ordering and a true network. Sources: PMC7549125 (Palantir).

## 10. Beyond edge recovery: what the field values (the reframe)

Researched to find a goal other than directed-edge recovery. Verification abstained on most claims
this round (only the GEARS capability claim survived 3-vote at 2-0), so treat the rest as consistent
across independent sources but not individually confirmed.

- Perturbation-effect PREDICTION has the same wall as edge recovery. Across PerturBench
  (arXiv:2408.10609), scPerturb / scPerturBench (Nat Methods s41592-025-02980-0), and Nat Methods
  2025 (s41592-025-02772-6), sophisticated deep predictors (GEARS, CPA, scGen, SAMS-VAE, BioLord,
  foundation models) do NOT reliably beat simple baselines (latent-additive, linear, or mean
  prediction); for double perturbations a plain additive baseline (sum of single-gene effects) often
  wins. One source (biorxiv 2024.12.23.630036) contests this. So pivoting to prediction-for-accuracy
  hits the same wall.
- The valued contribution is CAPABILITY, not accuracy. GEARS' framed novelty (verified 2-0) is
  generalization to UNSEEN perturbations, predicting the outcome of gene combinations where both
  genes were never perturbed, by extrapolating through a gene knowledge graph. The field rewards
  methods that extrapolate to unseen perturbations/contexts, not ones that fit seen data better.
- Program / factor discovery is a separately-valued, tractable deliverable. cNMF (consensus NMF),
  MOFA, and recurrent metaprograms are judged by pathway/GO enrichment and cross-dataset RECURRENCE,
  not predictive accuracy, and these methods CAN beat simple clustering baselines. Interpretable
  factor structure is a publishable contribution on its own. This is the structured-transform idea
  applied to genes.
- scPerturb contributes the E-distance / E-test (energy distance): a standardized statistic for the
  MAGNITUDE and significance of a perturbation effect, a characterization tool rather than an
  edge-recovery method.
- The recurring OPEN problem, flagged as tractable for a small team: generalization to unseen
  cellular contexts/perturbations, and the failure of current models to capture HETEROGENEOUS
  single-cell responses rather than the population average. The per-cell deviation from the mean
  response is exactly the "small ripples under the dominant mode" intuition, and it is the stated
  frontier.

Implication: the reframe with signal is interpretable program/structure discovery (validated by
reproducibility and external enrichment, where the bar is not beating GENIE3 on edges) and, more
ambitiously, characterizing single-cell response heterogeneity (the field's open problem, and the
"ripples" intuition). Both use the project's existing response-geometry and anchor tooling. Sources:
arXiv:2408.10609; Nat Methods s41592-025-02980-0, s41592-025-02772-6; GEARS (Nat Biotechnol
s41587-023-01905-6, PMC11180609); biorxiv 2024.12.23.630036.

## 8. Net implications for this project

- The interventional-asymmetry idea is validated and already implemented by others on RPE1; the
  project's net_out (exp 26) reproduces it.
- Deep / nonlinear models do not beat simple baselines here; the correct comparators are the
  established simple methods (GENIE3, GRNBoost2, PIDC, lagged regression), not chance. This is the
  standing methodology requirement for any positive.
- The dynamical-recovery direction (exp 30 to 32) sits in a genuine gap (no DMD/Koopman GRN
  benchmark against truth on single-cell time-series), but the gap also means the claim must be
  established against the BEELINE/DREAM4 leaderboard methods, not against chance or plain
  correlation, to be meaningful.
