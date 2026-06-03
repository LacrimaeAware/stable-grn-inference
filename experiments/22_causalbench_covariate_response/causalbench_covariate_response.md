# Experiment 22 — Covariate-aware direct-effect response geometry (RPE1)

## The question in plain terms

Experiment 21 found that when you knock down a gene in RPE1, the cell's response is **broad**
(one dominant mode explains ~53% of all response variance) and that *blindly deleting* that
mode hurt — it removed real directional signal. So the obvious next question:

> Is the broad response a removable confound (technical / amplitude), or is it real biology?
> And can we separate sharp, gene-specific "direct" effects from it **without deleting real
> signal** — so that an inferred network explains the cleaned response better?

This experiment answers that with covariate-aware methods instead of blind SVD deletion.

## What we did

On the real RPE1 response matrix (651 perturbations × 651 genes), all on the same data:
1. **Characterized the global mode** — does it track gene abundance / library size / #cells
   (technical) or a coherent gene program (biological)?
2. **Built two cleaned targets without deleting SVD modes**: a *shared-program residual*
   (subtract the average response profile, but keep it as an object) and a *covariate
   residual* (regress out per-perturbation self-knockdown strength + log #cells).
3. **Compared raw vs cleaned** on diffuseness, split-half stability, cross-split orientation
   reproducibility (exp21's ground-truth-free metric), and alignment with inferred graphs
   (correlation, sparse LASSO, GENIE3 — all from control cells only).
4. **Kept the global program as its own object** and read off its top genes.

## Results

### 1. The broad response is REAL biology — a cell-cycle/proliferation program
- Gene-side dominant mode correlates with gene mean-expression (Spearman **0.60**) and
  variance (0.57): it lives in high-abundance genes.
- But it is **not** driven by knockdown strength (0.04) or #cells (−0.18) — not a simple
  technical amplitude artifact.
- Its top genes are unambiguous: **CCNB1, MCM3, RRM2, DNMT1, H2AFZ, NASP, CENPW, TUBA1B,
  TUBB, PTMA, HMGB1…** — cyclins, replication licensing, S-phase, histones. This is a
  textbook **cell-cycle / proliferation program**. The broadness is real: knocking down an
  essential gene shifts cells along the proliferation axis.

This vindicates exp21's caution: the global mode is signal, not noise.

### 2. Covariate-aware cleaning does NOT cleanly separate "direct" from "broad"

| target | median effective responders | split-half stability | cross-split orientation |
| --- | --- | --- | --- |
| raw | 295 | 0.51 | 0.644 |
| shared-program residual | 319 (worse) | **0.38 (much worse)** | 0.600 (worse) |
| covariate residual | 299 (~same) | 0.55 (slightly better) | 0.638 (~same) |

- **Removing the shared program hurts** everything — it is reproducible biology, so taking it
  out lowers stability (0.51→0.38) and orientation. Same lesson as exp21's SVD removal.
- **Removing amplitude covariates is roughly neutral** — a small stability gain (0.51→0.55),
  no orientation change. So the response is not merely a knockdown-strength/sampling artifact.

The honest conclusion: **the broad response is intrinsic, reproducible biology and cannot be
cleanly residualized away** by these covariates. "Direct vs broad" is not a simple subtraction.

### 3. No observational structure explains interventional response well — GENIE3 ≈ 0

Spearman alignment between |inferred score| (control cells) and |interventional response|:

| structure | raw | program-resid | covariate-resid |
| --- | --- | --- | --- |
| correlation | 0.119 | 0.065 | **0.127** |
| sparse LASSO | 0.041 | 0.028 | 0.041 |
| GENIE3 (random forest) | **−0.003** | 0.006 | −0.012 |

Even the best cell (correlation on the covariate-residual) is only **ρ=0.13**. The ranking is
**correlation > sparse > GENIE3 ≈ 0**: GENIE3 — a standard, widely-used GRN method — has
essentially **zero** alignment with what interventions actually do. Covariate cleaning gives
correlation a small bump (0.119→0.127). This is the project's strongest cautionary result for
observational GRN inference: the methods the field ranks on do not track causal response.

### 4. Verifiable orientation survives
Cross-split orientation reproducibility stays ~0.60–0.64 across all targets (vs 0.5 chance),
confirming exp21's finding that interventional *direction* is a real, reproducible property —
and that it is not an artifact of the broad mode (it survives residualization).

## What this means (the constructive turn)

The "just residualize out the broad response" hypothesis is **closed**: the broad response is a
real cell-cycle/proliferation program, not a confound. So the path is not to *delete* it but to
**model it as a latent biological factor** and study what regulates engagement of that axis,
plus the gene-specific residual on top of it. And since observational scorers (correlation,
sparse, GENIE3) barely explain interventional response, the inference target should be the
**interventional response geometry itself**, not an observational edge list.

## Engineering
New general, synthetic-tested library functions: `shared_response_program` (interpretable
program/residual decomposition) and `residualize_against_covariates`. +2 tests (suite 133
green). Both are representation-agnostic and reusable. No wavelets/scattering, no RL, no neural
nets — as planned.

## Verdict and next step
A clean, productive **negative result**: covariate-aware cleaning does not manufacture a sparse
direct-effect target, because the broad response is genuine proliferation biology, and standard
observational GRN methods (notably GENIE3) do not predict interventional effects. The
verifiable-orientation result is robust.

Best next step (exp23): stop trying to *remove* the cell-state axis and instead **condition on
it** — e.g. estimate responses within proliferation-matched cell strata (or partial out a small
set of explicit cell-state factors jointly with the perturbation), and ask whether a structure
inferred to explain the *interventional* response (not observational co-expression) finally
beats correlation. That is the natural continuation of "infer structures that explain
perturbation-response geometry."
