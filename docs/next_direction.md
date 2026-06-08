# Next direction: synthesis after experiment 28

One consolidated reference for the next research direction, written to replace the
scattered cross-model and cross-session notes. It states the goal, what is already
established, the user's pairwise-difference intuition and its mathematical reality,
why different models gave conflicting advice, and the weighted directions forward
with concrete experiment specs. Companion to `notes_and_next_steps.md` (short
author log) and `experiment_summary.md` (per-experiment results).

## 0. Purpose and honest scope

What this project is. A controlled study of one question: how much directed
regulatory structure can be recovered from real data, and what sets the ceiling.
The data regimes (DREAM4 simulated, BEELINE static, RPE1 interventional) were chosen
to vary how much directional information the data contains, so the ceiling could be
measured rather than asserted. GRN inference is the test case; the durable interest
is recovering stable hidden structure from noisy snapshots. The direction changes
are not random trials. They are a narrowing search that has been eliminating
branches of this one question, and on static data the branches are nearly exhausted.

What it has produced (real, defensible, field-consistent):

- A regime result: edge-direction recoverability is set by the data regime, not the
  method. Time-series orient well (~0.9), static data does not (~0.5 to 0.6),
  interventional snapshots decide direction in aggregate but their individual edges
  do not reproduce. This matches the published field (CausalBench, BEELINE, the 2025
  foundation-model benchmark): simple baselines match deep models.
- A mechanism for the RPE1 failure: one convergent program (cell-cycle, ~53% of
  response variance) drowns gene-specific edges and is not linearly separable from
  them. Subtraction, inversion, and transfer all fail (exp 22 to 24).
- A framework (exp 28): the separability phase diagram. Recovery has two axes, a
  fixable dominant-mode fraction and an unfixable SNR floor, and RPE1 sits below the
  floor. This is a quantitative criterion for which future datasets are worth the
  effort, and it explains every prior negative as one point on a map.

What it will not produce, stated plainly. It will not beat a perturbation-prediction
or GRN benchmark on RPE1. The literature is clear that nonlinear and deep models do
not beat simple linear baselines on this data (section 8), and exp 28 explains why
for this dataset specifically. Direction A is a near-certain informative negative;
its value is closing the pairwise/whitening question and leaving behind reusable
external-anchor validation, not a leaderboard number.

Where a positive result actually lives. A genuine positive needs the one ingredient
RPE1 lacks: a time axis (Direction B). Real dynamics, graded against truth, on data
with specific-SNR above the exp 28 floor. That is also the honest answer to the
project's founding goal of understanding real dynamic data: real perturbation
snapshots are hard for now-characterized reasons, and real dynamics require
time-resolved measurement.

Paper-worthiness, honestly. The benchmark-beating framing is not available here. The
defensible contribution is the regime ladder plus the exp 28 separability phase
diagram, framed as a diagnostic and negative-results contribution with a concrete
data-selection criterion. That is workshop-paper, methods-note, or preprint
territory, not a top-venue method-beats-SOTA paper. It is the level this project
operates at, and it is real.

## 0b. Relation to the original idea, and the role of each dataset

Plain reconciliation, to keep the threads straight.

The original idea was pairwise response geometry: nonlinear functions of perturbation-response
differences D_gh = Delta_g - Delta_h across perturbations, to find dynamic or nonlinear structure.
What became of it: the specific object Delta_g - Delta_h is mostly circular (section 3), and the
one part that is genuinely pairwise, the response-matrix asymmetry A = M - M^T, is what exp 29
tested. Exp 29 did not beat the per-gene axes (net_out, magnitude) it is built on. So the original
object did not yield a new signal; the asymmetry it pointed at is real but already measured by
net_out (exp 26). The intuition kept pointing at real structure; that structure was already captured.

The dynamical line (exps 30-33) is a SEPARATE thread, not the pairwise idea. The dynamical operator
(DMD) is the simplest time method: fit one matrix A so expression at t+1 is approximately A times
expression at t; the off-diagonal entries are candidate directed edges, and it needs a time axis. It
is the same family as lagged regression (exp 7); in fact it is the plain, un-regularized version of
what exp 7 already did with GENIE3 and LASSO, so testing it overlaps the DREAM4 time-series work
already done.

DREAM4 number context (to avoid confusion): on DREAM4 Size10 the project already reached AUPR 0.65
with a dynamic sparse model including the self-edge (exp 8/9), 0.53 with lagged GENIE3 on non-self
edges (exp 7), and correlation 0.33 on the static multifactorial data (exp 1). The dynamical
operator's 0.37 (exp 33, non-self) is below the project's own earlier DREAM4 results, so it is not a
new high.

What each dataset is for:
- DREAM4 (simulated, time-series, exact truth): the clean validation testbed; characterized in exps
  1-14.
- BEELINE / BoolODE (simulated single-cell, pseudotime, exact truth): the single-cell version with
  exact truth and a cell-count knob.
- RPE1 (real CRISPR, static snapshot, no time axis, no clean truth): the hard real problem (the
  dominant cascade).
- RENGE (real CRISPR, time-resolved): real data that has the time axis RPE1 lacks.

These vary simulated-vs-real and static-vs-time. The risk this raised, which the dynamical line
realized, is breadth of datasets without a method that has a reason to beat the established
baselines.

## 1. Theme

The durable interest is not gene biology. It is: **recover stable hidden
structure from noisy snapshots** — the same question as the structured-transform
project (latent factors / transformation axes in representation space), here in
expression space (regulatory structure under a dominant response program). GRN
inference is the test case; the data regimes (DREAM4 simulated, BEELINE static,
RPE1 perturbation) were chosen to vary how much directional information the data
contains.

## 2. What is established (the load-bearing findings)

The arc converged on a clean story. The recoverable structure in RPE1 is a small
set of reproducible per-gene and per-pair axes; every attempt to pull a cleaner
gene-specific signal out from under the dominant program has failed.

Reproducible and real:

| Object | What it is | Reproducibility |
|---|---|---|
| magnitude / essentiality (exp 26) | per-gene response size `‖Δ_g‖`, breadth, cascade | split-half 0.92–0.97 |
| `net_out` (exp 26) | `mean_h(|Δ_g[h]| − |Δ_h[g]|)`, cascade position / upstreamness | split-half 0.986 |
| response-matrix orientation asymmetry (exp 21/22) | `|D[g,h]|` vs `|D[h,g]|` | 0.70 cross-split over 118k pairs; survives residualization |
| dominant program (exp 22) | cell-cycle / proliferation axis (CCNB1, RRM2, MCM3, histones, tubulins) | ≈53% of perturbation-response variance; real biology, abundance-linked, not a technical artifact |

Failed — every linear attempt to remove, invert, or transfer the dominant program:

| Attempt | Result |
|---|---|
| subtract the shared program (exp 22) | stability 0.51 → 0.38; orientation 0.644 → 0.600 (worse) |
| covariate residualize (exp 22) | roughly neutral (0.51 → 0.55), no orientation gain |
| linear deconvolution `W = I − (I+D)^{-1}` (exp 23) | exact on synthetic, fails on real RPE1 (noise + nonlinearity + cell-cycle break it) |
| low-rank transfer to a held-out perturbation (exp 24) | shared structure loses to `self_only` (0.34 vs 0.41) |
| factor-atlas reweight/removal (exp 25) | multi-seed gain −0.12; cell-cycle entangled with real biology, not a separable nuisance |
| observational scorers vs interventional response (exp 22) | correlation ρ≈0.13 best; sparse 0.04; GENIE3 ≈ 0 |
| `net_out`-ordering-adjacent pairs as edges (exp 27) | mediation ≈3.3 for ~every pair; cascade-local reproducibility drops to 0.425 |

Today (exp 28, the separability phase diagram, synthetic + ground truth): the
dominant-mode-vs-specific-structure separation has **two distinct failure axes**
that the project had conflated — `rho` (dominant-mode variance fraction), which is
*fixable* by deflation when the mode is a clean low-rank component, and `SNR`
(specific signal vs noise), which is *not* fixable: below a noise floor no method
clears chance at any `rho`. RPE1 sits at high `rho` (≈0.53) and low specific-SNR,
the corner where nothing recovers. The refined conclusion: **RPE1's bottleneck is
the SNR floor, not the dominant mode per se** — removing the cell-cycle axis does
not help because there is little recoverable specific signal underneath it
(exactly what exp 21/22/25 observed).

External corroboration (literature pass, section 8): the field independently reports
the two facts this project found, that individual directed edges do not reproduce
while aggregate and per-gene structure does, and that deep models do not beat simple
linear baselines on Perturb-seq. The negatives are not a local artifact of this
pipeline.

## 3. The pairwise-difference intuition, stated precisely

The user's idea: apply a function `F` to many pairwise differences — cell-to-cell
`d_ij = x_j − x_i` or response-to-response `D_gh = Δ_g − Δ_h` — to find nonlinear
structure that correlation misses, and optionally orient part of it with an anchor.

### 3a. The symmetry trichotomy (what `F` can be)

"A function of pairwise differences" is not one method. Its landing spot is fixed
by the symmetry of `F`:

| `F` | What it sees | Identity |
|---|---|---|
| **linear** | only the data covariance (the set of all pairwise differences has covariance `2Σ`) | = correlation / PCA. Nothing beyond what we are trying to escape. |
| **symmetric**, `F(d)=F(−d)` (distances, kernels, `|d|`, neighborhood graphs) | the full symmetrized distribution — all even moments, the manifold geometry | genuinely **more than correlation**, but the known kernel / manifold family. Carries **no direction**. |
| **antisymmetric**, `F(d)=−F(−d)` | sign / order | can carry direction, but over **unordered** pairs it averages to zero. With an anchor (control→perturbed) ordering each pair, it becomes interventional asymmetry = **`net_out`, already built**. |

So the intuition is real but it is three known families wearing a trench coat;
which one you get is determined by `F`, not chosen freely. **Direction never falls
out of a symmetric object.** It must be *injected* by the regime (time:
`corr(A_t, B_{t+1}) ≠ corr(B_t, A_{t+1})`; intervention: `net_out`) or *extracted*
from higher-order structure (non-Gaussianity, which ICA/LiNGAM exploit — the same
machinery the user independently re-derived in the structured-transform project).

### 3b. The object error (the key correction)

`D_gh = Δ_g − Δ_h` is the difference of two **rows** of the response matrix
`M[g,h] = Δ_g[h]`. The only non-circular, genuinely pairwise target —
*does perturbing g move h more than perturbing h moves g* — lives in the
**off-diagonal pair** `(M[g,h], M[h,g])`, i.e. the asymmetry `A = M − Mᵀ`. The row
difference `Δ_g − Δ_h` does **not** contain `M[h,g]` in any recoverable form. So
the proposed object structurally cannot represent the one target that would
justify a pairwise method. This is why it kept feeling like the idea was being
swapped: the good target (asymmetry) needs a different object (`A = M − Mᵀ`) than
the one stated (`Δ_g − Δ_h`).

### 3c. Circularity (what to drop)

Targets of the form `f(g) − f(h)` for a per-gene scalar `f` — magnitude
difference, essentiality difference, `net_out` difference — are algebraic
functions of the inputs `Δ_g, Δ_h`. A model "predicting" them is recomputing `f`,
not discovering structure; the matrix of all such differences is antisymmetric and
fixed by 651 per-gene numbers, not 651² facts. exp 26 already computes the per-gene
`f` at 0.97–0.986. Differencing cannot beat the node predictor it is built from.

## 4. Why the models diverged (the GPT-vs-Opus question)

The split was not "one model is smarter." Two reasons:

- **Context.** The GPT take reasoned from the research profile alone, with no
  repo access, so it could not know `net_out`, exp 21, exp 24, exp 26 already
  exist; building the idea out was reasonable given only the profile. The
  skeptical take scouted the repo, found the overlap, and weighted "already built
  + prior negatives" heavily. "Blind" only means blind to context.
- **Stance.** A model asked to *engage* a direction finds reasons it could work; a
  model asked to *audit* it finds reasons it is redundant. Both correct at their
  jobs.

On the specifics: the **circularity claim is correct** (textbook — several targets
are algebraic functions of the inputs). GPT did **not** miss it; it silently
dropped the circular targets and ran with the one live target (asymmetry/direction)
— a sin of omission (it never flagged them), not an error. Both are right about
different halves: GPT that **direction is the prize** (the goal), the skeptic that
**a symmetric statistic cannot deliver it** (the tool). The reconciliation is §3a:
direction is injected or non-Gaussian. The math is the arbiter, not either model.

The one genuinely new contribution from the GPT side is worth keeping: **whiten the
dominant mode, do not subtract it** (see §5, Direction A). The project tried
subtracting (exp 22) and reweighting/removal (exp 25) and both failed; whitening —
keep every direction, rescale so the small specific directions are comparable in
scale to the dominant one — is the one operation on this exact bottleneck that has
not been run.

## 5. Weighted directions forward

### Direction A — whitened interventional-asymmetry gate (RPE1; near-term, decisive). Weight: run first.

The honest resolution of the pairwise/asymmetry intuition, scoped to the one
non-circular question, with the external anchor exp 26 flagged and never added.

- **Object:** the response matrix `M[g,h]=Δ_g[h]` on the 651 perturbed∩measured
  genes and its asymmetry `A = M − Mᵀ`. Not `Δ_g − Δ_h`.
- **Function (a race, not a model zoo):** raw `|A|` (the exp 21 baseline, 0.70) vs
  **shrinkage-whitened** `A` (downweight the top modes, regularize the tail; sweep
  strength) vs `net_out`/magnitude-residualized `A`.
- **Gate 0 (cheap, decisive, no ML):** split-half reproducibility of the residual
  asymmetry after removing `net_out(g)−net_out(h)` and the magnitude difference. If
  the residual is not reproducible, **stop** — clean negative, no stable pairwise
  signal beyond the known axes.
- **Target / external anchor (closes the self-reference gap):** DepMap gene-effect
  essentiality (per-gene severity) and CORUM/STRING complex co-membership
  (relational/pairwise). Validate held out by **gene**, never by pair.
- **Baselines to beat:** raw asymmetry, `net_out`, magnitude, correlation/PCA, and
  a **per-gene-label-shuffle** null (not a pair-shuffle, which is too weak against
  an antisymmetric target).
- **Prior art / competitor (literature pass, section 8):** asymmetry orientation is
  already operationalized on this exact RPE1 data by CausalGRN, which treats the
  dominant mode by adding the wild-type PC1 program as a pseudo-gene node. Race
  whitening and residualization against that pseudo-gene-node treatment, not only
  against raw. Whitening has strong precedent (it beats subtraction in CRISPR
  co-essentiality, PMC9707256) but only in a higher-SNR bulk regime; exp 28 predicts
  it fixes dominance, not the SNR floor RPE1 sits below, so a negative here is the
  likely outcome and the reason will be SNR rather than the operation.
- **Failure / kill:** whitened `A` no more reproducible than raw **and** beating no
  baseline against the external anchor → the directional-geometry line stops; the
  contribution is the diagnostic plus a clean negative.
- **Calibrated odds of a real positive: low, ≈10–20%.** Value even on a negative:
  it settles the question, delivers the external-anchor validation exp 26 left
  undone, and runs the one untried operation (whitening). Whitening caveat: full
  `Σ^{-1/2}` amplifies the lowest-variance (noisiest) directions and on
  split-half-cosine ≈0.51 data can destroy reproducibility — hence *shrinkage*
  whitening with a strength sweep, watching reproducibility.

### Direction B — separability / ground-truth dynamical decomposition. Weight: durable frontier; already started today (exp 28).

Stop fighting on data whose specific-structure truth is unknowable; study the
separation itself where the answer is generated and gradeable.

- **Object:** synthetic systems with known specific structure under a dominant mode
  (the exp 28 generator); sweep `rho`, `SNR`, `entanglement`. Then add a **time
  axis** so DMD / Koopman / SINDy operators can be identified and graded against
  truth.
- **Why:** it grades against ground truth — the one thing every RPE1 experiment
  lacked — and attacks the dominant-mode-vs-specific separation directly. exp 28
  already shows that separation is bounded by an SNR floor.
- **Open question (literature, pending):** DMD/Koopman/SINDy need a **time axis**
  that RPE1 snapshots do not have. So this path is the synthetic ladder **plus a
  real time-resolved perturbation/trajectory dataset**, not RPE1 snapshots. exp 28's
  phase diagram becomes the screen that says which real datasets are worth the
  effort (specific-SNR above the floor, not merely low dominant-mode fraction).
- **Calibrated value: higher long-term.** It is the one place a method can be
  validated, and where the cascade/asymmetry instinct can be tested non-circularly
  — time supplies a real arrow that a static snapshot cannot.

### Direction C — raw pairwise-difference model zoo / learned `F` / kNN / embedding on `Δ_g`. Weight: mostly closed; fold in.

Low priority. It is the known symmetric family (§3a); exp 25 already ran kNN /
embedding on `Δ_g`, and the structured-transform prior says a learned `F` (metric
learning) overfits and ties the closed form. Keep only the symmetric-geometry
diagnostic, and only as Gate 0 of Direction A. The literature pass confirms this in
domain: on Perturb-seq, deep and nonlinear models do not beat simple linear
baselines (CausalBench winners were standard ML; five foundation models plus
GEARS/CPA failed to beat additive baselines; see section 8).

## 5a. Status (built) and an honest evaluation of the results

Both directions are implemented with tests (187 in the suite). An adversarial methodology audit
of experiments 28-32 corrected the earlier framing; this section is the corrected reading.

- Direction A is experiment 29 (`experiments/29_whitened_asymmetry/`,
  `src/stable_grn_inference/analysis/asymmetry.py`). Gate 0 passes on a SYNTHETIC positive
  control only; it has not been run on real RPE1, where a negative is predicted per the exp 28
  SNR floor. The whitening sweep shows whitening does not help (best alpha 0.00), and CausalGRN's
  pseudo-gene-node treatment (section 8), the established competitor on this exact data, is not
  yet raced. So exp 29's only current finding is that raw asymmetry is reproducible on synthetic
  data.

- Direction B (experiments 30-32) is NOT a benchmarked positive. The dynamic-mode operator is
  textbook least-squares VAR(1) / DMD; its field equivalents are Granger causality and dynGENIE3.
  Every real-data comparison was against `static correlation`, which is symmetric and cannot
  orient an edge by construction, so beating it at a directed task is guaranteed and uninformative
  about method quality; it only re-demonstrates the regime ladder (time order enables orientation).
  - On synthetic VAR(1), the operator reaches 0.705 normalized directed recovery vs 0.117 for the
    symmetric static baseline: a controlled demonstration that a time-using operator orients edges
    a symmetric snapshot cannot, not an independent positive.
  - On DREAM4 Size10 time-series the operator reaches directed AUPR 0.37, which ranks below all the
    lagged methods this project already ran on the identical networks and pairs in exp 7: lagged
    GENIE3 RF 0.53, GENIE3 ET 0.53, LASSO 0.51 / 0.49, lagged correlation 0.46. It beats only the
    static baseline, which cannot orient, and is below every established lagged method.

  - On BoolODE single-cell (exp 31) the operator beats the static baseline on 4 of 6 topologies
    (orderable trajectories); it LOSES on cycles (0.028 vs 0.161) and trifurcating branches
    (0.178 vs 0.233); the overall margin is marginal (+0.05 normalized) and flat in cell count,
    contradicting the SNR/sample-size rationale for the sweep. No orientable established method
    (lagged GENIE3, BEELINE leaderboard) was compared.
  - On real RENGE Perturb-seq (exp 32) the time-resolved STRUCTURE is real and reproducible
    (response grows ||D|| 2.79 to 3.79 over days; net_out ordering reproducible 0.75 and
    stabilizing to 0.92). But the graded recovery is against STRING, which is UNDIRECTED, so it
    is SKELETON recovery, not directed recovery; the win over a hand-built observational
    correlation control is marginal (0.366 vs 0.323) and reverses on day 4 (0.407). No established
    GRN method (GENIE3/GRNBoost2) and no directed truth were used.

The single durable, possibly-novel contribution is exp 28's separability phase diagram (the
rho-vs-SNR decomposition with an SNR floor, RPE1 placed on it; note the reported grid is the
quick-mode run, 50 genes / 2 seeds, so the SNR-floor threshold is provisional until the full grid
runs) and the regime-ladder diagnostic framing. These are legitimate negative-results /
methods-note contributions and do not depend on a method baseline. None of exps 30-32 currently
establishes a novel positive, because none beats an established method, and on DREAM4 the operator
already loses to lagged GENIE3.

The corrective is the benchmarked comparison (experiment 33): put the operator in the same table as
lagged GENIE3 and lagged LASSO on the same data, and grade RENGE against a DIRECTED truth (TRRUST /
DoRothEA / the RENGE ChIP set). Decision rule: if the operator matches or beats lagged GENIE3,
Direction B has a real modest positive; if it underperforms (as DREAM4 indicates), the honest
contribution is "time order enables orientation, and simple lagged feature-importance is the method
of choice", a reproduction/negative stated as such.

Experiment 33 ran this benchmark (directed AUPR, same pairs and truth). DREAM4: the operator ranks
LAST of four orientable methods (DMD 0.37 vs lagged GENIE3 0.54, LASSO 0.51, correlation 0.46).
BoolODE single-cell: the operator is mid-pack, 2nd of four (DMD 0.41; lagged LASSO 0.45 best, lagged
GENIE3 0.40, correlation 0.40). There is no benchmarked win on either dataset; the established lagged
methods match or beat the operator. The dynamical line therefore has no novel positive, and its
durable contribution is the exp 28 separability diagnostic and the regime-ladder framing. The
operator's earlier "positives" came only from comparing to a symmetric baseline that cannot orient
edges, and on the time-series data the established lagged methods already do better.

## 5b. Direction B datasets (verified scout, ranked)

For grading a dynamical operator against checkable truth, best first:

1. DREAM4 in silico time-series (already local, used in exp 30 Part B). Exact directed gold
   standard; 21 regular timepoints; 5 (size-10) or 10 (size-100) replicates per network.
   Validates the method against exact truth. Simulated, so it does not test real-data SNR.
2. RENGE / GEO GSE213069: the best real time-resolved option. Time-series single-cell CRISPR
   in human iPSCs, 4 regularly spaced timepoints (days 2 to 5), 23 knockout TFs, about 103
   genes modeled, about 5,000 cells per sample, with a ChIP-seq proxy truth (19 genes, binding
   threshold 300). Moderate download (low single-digit GB). The perturbations supply the
   directional leverage the static RPE1 analysis lacked. Thin on timepoints, so use
   pseudobulk-per-timepoint lagged regression / DMD.
3. BEELINE / BoolODE (Zenodo 10.5281/zenodo.3378975, 279 MB; adapter exists from exp 15 to
   16). Exact truth from the generating Boolean/ODE model, with a built-in cell-count sweep
   (100 to 5,000) that mirrors the exp 28 SNR/sample-size floor. Axis is pseudotime, so bin
   along pseudotime or regenerate full ODE trajectories for a clean operator fit.

Supporting literature (captured for reference):
- A scRNA-seq snapshot result independently states the project's floor: mean time courses
  cannot distinguish no-interaction from bidirectional networks (they need third moments), and
  moment inference needs fine snapshot intervals (Cell Systems 2021, PMC8441581). Useful
  contrast material for a write-up.
- SINDy recovered a bacterial competence regulatory network with rational-function libraries
  (Mangan et al. 2016, arXiv:1605.08368). No clean DMD/Koopman GRN-recovery paper grades
  against a ground-truth network on single-cell time-series; that is a genuine gap and a
  novelty opening for Direction B.

## 6. Recommendation

1. **Run Direction A, Gate 0 first.** It is cheap, needs no ML, and decisively
   settles whether any pairwise signal survives beyond `net_out`/magnitude. Most
   likely outcome: a clean negative that re-finds the cascade. If Gate 0 passes,
   run the whitening race + external anchor (Gate 1).
2. **Continue Direction B in parallel** as the durable frontier: deepen exp 28
   (entanglement axis, then a time axis + DMD/Koopman/SINDy graded on synthetic
   truth), and use its phase diagram to choose a real time-resolved dataset.
3. **Fold Direction C** into A's Gate 0; do not build the model zoo.

## 7. Honest calibration

I do not think there is an undiscovered method inside the pairwise-difference idea.
It is three known families selected by the symmetry of `F`, and its one
non-circular target needs a different object (`A = M − Mᵀ`, already half-built and
0.70-reproducible). The intuition is a good **compass** — it keeps landing on real
landmarks (covariance, manifolds, ICA/LiNGAM, interventional asymmetry) — but
reliably finding landmarks is not the same as there being an undiscovered one
beside them. The expected outcome of Direction A is a **diagnostic plus a clean
negative**, not a benchmark win — the same verdict the structured-transform project
reached in a different domain (explicit factor structure was valuable for
interpretability and debiasing, not for raw accuracy). Stating that plainly is the
point of this document, not discouragement: the directions above are the honest,
concrete, non-redundant moves, ranked.

## 8. Literature evidence (resolved)

A cross-model deep-research pass (5 angles, 23 sources fetched, 106 claims
extracted, 25 adversarially verified, 20 confirmed and 5 refuted) answered the
questions this section previously held open. Findings below carry effect sizes
where available; claims that did not survive verification are flagged so the
document does not over-state them.

**Asymmetry is a validated orientation cue, and is already operationalized on this
exact dataset.** CausalGRN (preprint, biorxiv 2025.12.30.692369) orients an edge
A to B when knockdown of A shifts B and the evidence is one-sided, on RPE1 (8,644
genes, 154,577 cells, 1,159 knockdowns) and K562, after regressing out S/G2M,
library size, and mitochondrial fraction. Interventional data improves
identifiability over observational (PMC12579002). Consequence: the asymmetry
instinct (Direction A) is field-validated, not novel, and CausalGRN is a named
competitor. Its dominant-mode treatment differs from whitening: it adds the
wild-type PC1 program as a pseudo-gene node, which removes thousands of false
edges. That is a third dominant-mode handling to benchmark against subtraction
(exp 22) and whitening (Direction A).

**Edges do not reproduce; aggregate structure does. The field matches exp 21/27.**
The specific external numbers floated for this (edge-overlap F1 near 0.17 versus
eigencentrality replicating near 0.66) were among the claims the verification pass
refuted, meaning they could not be confirmed from the cited source; treat them as
unverified. The qualitative pattern is corroborated and matches the project's own
exp 27.

**Whitening beats subtraction, with strong precedent in a sibling data type and a
documented failure mode.** Covariance whitening (Cholesky / ZCA-cor) is the single
largest boost to CRISPR co-essentiality recovery, and the smaller post-whitening
correlations enrich better for true interactions (PMC9707256). ZCA-cor is the
correct variant: it uniquely keeps whitened variables maximally similar to the
originals (arXiv:1512.00809). Two caveats bound Direction A. First, that evidence
is DepMap bulk cell-line co-essentiality, not single-cell Perturb-seq, so transfer
is unproven. Second, BaCoN documents the exact amplification failure mode, whitening
manufacturing spurious proximal pairs by inflating low-variance directions. Cross
reference exp 28: co-essentiality is a higher-SNR aggregate regime, RPE1 single-cell
deltas are low-SNR, and the exp 28 diagram predicts a dominant-mode operation
(deflation or whitening) fixes rho but not the SNR floor. Whitening's documented wins
are therefore consistent with exp 28, and exp 28 predicts whitening will not rescue
RPE1 specifically. Direction A Gate 0 is still worth running because it is cheap and
settles it; the prior is now sharper (likely negative on RPE1, for the SNR reason,
not the dominance reason).

**External anchors are validated with concrete effect sizes (closes the
self-reference gap in Direction A).** Replogle 2022 (PMC9380471): correlation-based
clustering recovers CORUM complexes at median r=0.61 versus 0.10 background, tracks
STRING, and DepMap confirms the groupings. Genome-wide co-essentiality (Wainberg,
PMC8763319; 485 lines, 17,634 genes, 93,575 pairs) enriches roughly 160-fold for
CORUM, 130-fold for hu.MAP, and 7.5-fold for STRING under generalized least squares.
DepMap gene-effect is itself a bias-corrected interventional knockout-fitness score
(biorxiv 720243). Much real structure is invisible to expression similarity: at
STRING confidence at least 900, about half the pairs have cosine at most 0.5. Use
CORUM/STRING for the relational anchor and DepMap for the per-gene severity anchor,
validated held out by gene. Caveat: most of these validations are K562, not RPE1.

**Nonlinear and deep models do not beat simple linear baselines here (confirms
Direction C is mostly closed).** CausalBench winning solutions were standard ML
(LightGBM, mean-difference, boosted trees) beating the deep causal method DCDI
(arXiv:2308.15395). Five foundation models plus GEARS and CPA failed to beat
no-change / additive baselines (Nature Methods 2025, s41592-025-02772-6). A claim
that a contrastive MLP crushed correlation for recovering STRING pairs (AUC 0.908
versus 0.518) was refuted in verification, so the nonlinear-beats-linear case is
unsupported in this corpus. The structured-transform prior (learned F overfits, ties
the closed form) now has field-level confirmation in this domain.

**DMD / Koopman / SINDy: no surviving evidence, a genuine gap, and a time-axis
dependency (bounds Direction B).** The pass fetched dynamical-systems sources but no
claim about these methods on single-cell or perturbation data survived verification.
This is a gap, not a disproof. Combined with the standard time-axis requirement, it
confirms Direction B must use a synthetic ladder plus a real time-resolved dataset,
not RPE1 snapshots.

Genuinely still open: whether any aggregation of asymmetry yields a reproducible
RPE1 directed-edge set given that individual edges do not reproduce anywhere; and a
head-to-head of CausalGRN's pseudo-gene-node against shrinkage whitening on RPE1
(which Direction A's race would itself produce).

### 8a. Sources

- Causal orientation / interventional asymmetry: CausalGRN (biorxiv
  2025.12.30.692369); interventional identifiability (PMC12579002); CausalBench
  analysis (arXiv:2308.15395).
- Whitening / signal separation: ZCA-cor optimality (arXiv:1512.00809); whitening
  boosts co-essentiality (PMC9707256).
- External anchors: Replogle 2022 (PMC9380471); genome-wide co-essentiality / GLS
  (PMC8763319); DepMap gene-effect (biorxiv 720243); STRING-vs-expression gap
  (arXiv:2603.20955).
- Linear vs deep benchmarks: foundation-model perturbation benchmark (Nature Methods
  2025, s41592-025-02772-6); CausalBench (arXiv:2308.15395); BEELINE (PMC7098173).

Source caveat: CausalGRN and the STRING-gap paper are preprints; the whitening
evidence is bulk cell-line co-essentiality, not single-cell; several anchor
validations are K562, not RPE1; the deep-vs-linear results are a mid-2025 snapshot.

## 9. Pointers

- Per-experiment results: `experiment_summary.md`; methods/stats: `project_retrospective.md`.
- Reusable response/geometry tooling: `src/stable_grn_inference/data/interventional.py`
  (`perturbation_response_matrix`, `shared_response_program`, `direct_effect_filter`,
  `split_half_stability`, `deconvolve_response`).
- Separability tooling (exp 28): `src/stable_grn_inference/dynamics/separability.py`.
- Real RPE1 data is a large, git-ignored Perturb-seq `h5ad` under
  `data/raw/causalbench/` (not committed); the asymmetry submatrix is 651×651.
