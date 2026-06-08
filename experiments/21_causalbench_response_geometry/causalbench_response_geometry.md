# Experiment 21 — Perturbation-response geometry & direct-effect filtering (RPE1)

## Why this experiment exists (the Track A ↔ Track B synthesis)

Track B studies intervention-response geometry in *representation* space:
$\Delta_k(x)=\Phi(T_kx)-\Phi(x)$ for a controlled factor $T_k$. Track A has the same object
in *expression* space: a gene perturbation produces a response delta
$\Delta_g=\mathbb{E}[X\mid\text{do }g]-\mathbb{E}[X\mid\text{ctrl}]$. Stacking these gives a
**response matrix** $D[g,j]=\Delta_g[j]$.

This reframes Track A from "rank directed edges / chase AUPR" to "**infer and evaluate
structures that explain perturbation-response geometry**." Edges become *one compressed
explanation* of $D$, not the whole object. The experiment imports Track B's geometry tools —
delta-subspace spectrum, split-half stability, nuisance/global-mode removal — and applies
them to real RPE1 Perturb-seq. It deliberately uses **no gene ordering or graph**, so no
wavelets/scattering are forced onto unordered genes (that waits until a trustworthy gene
geometry exists).

It also directly attacks experiment 20's open wound: the interventional "reference" was
density-0.82 because perturbing a gene shifts much of the transcriptome. Here we *measure*
that broadness and try to subtract it.

## Setup

Real RPE1 (Replogle/Weissman) Perturb-seq via `load_replogle_raw_h5ad`: **651 perturbations
× 651 genes** response matrix (perturbed∩measured block), 11,485 control cells. Mean-shift
responses; split-half versions for stability; top-10 global SVD modes removed for the
"direct-effect" target. Control-vs-control mean-shift null sets the "a response counts"
threshold. Full run; `--quick` caps to 200 perturbations (numbers below are the full run,
with the 200-gene quick pass in parentheses where it differs).

## Results

### 1. QC — the response matrix is real biology
Median self-knockdown response `D[g,g] = -0.30`, **99.7% negative**. CRISPRi genuinely
lowers the targeted gene, end to end through the loader. (Quick: −0.29, 99.5%.)

### 2. The response is low-rank / global-mode dominated — exp20's density explained
- **Top-1 SVD mode = 53% of the response variance**; rank@90% var = 121/651; spectral
  entropy 0.47. (Quick: top-1 46%, rank 41/200.)
- Median **295 of 651 genes respond** per perturbation (diffuse).

A single dominant axis explains over half the variance: this *is* the "perturbing A shifts
much of the transcriptome" effect from exp20, now quantified via Track B's delta-subspace
spectrum. The dense-0.82 reference was not noise — it was a real, low-rank global response.

### 3. About half the perturbation responses are split-half reproducible
Median split-half cosine **0.51** (quick: 0.67); ~50% of perturbations have cosine > 0.5.
So roughly half the response *vectors* are reproducible across independent cell halves; the
rest are noise-dominated (often few-cell perturbations). This gives a principled
trustworthiness filter that replaces the crude >100-cell cutoff.

### 4. Interventional orientation is reproducible across cell halves
This is the metric exp20 could not provide. Exp20 showed direction is *decidable* (you can
pick one); it could not show the pick is *correct* (no ground truth). Here we verify
directionality **without any ground truth**: compute the asymmetry-implied direction
($|D[A,B]|$ vs $|D[B,A]|$) **independently on two cell halves** and ask how often they agree.

| target | cross-split direction agreement | n pairs | chance |
| --- | --- | --- | --- |
| raw response | **0.701** | 117,692 | 0.5 |
| direct-effect | 0.616 | 97,859 | 0.5 |

**70% agreement vs 50% chance** over ~118k pairs. The interventional direction is a
*reproducible* property of the data, not an artifact of one sample. This upgrades exp20's
"decidability 0.61" to a reproducible directionality — the strongest evidence in the project
that the interventional direction is a stable, sample-independent property, not (absent a
ground-truth graph) a check that it is the correct one. (Quick: 0.74 raw.)

### 5. Observational co-expression barely aligns with interventional response (deflationary)
Spearman between observational |score| (control cells) and interventional |response|:

| target | correlation ρ | sparse ρ |
| --- | --- | --- |
| raw response | **0.119** | 0.042 |
| direct-effect | 0.059 | 0.029 |
| direct + split-stable sources | 0.068 | 0.035 |

ρ ≈ 0.12 (correlation) / 0.04 (sparse) is **negligible**, and direct-effect filtering does
*not* rescue it. Observational structure is almost uninformative about which interventions do
what — the sharpest statement yet that **observational GRN inference ≠ causal structure**.
This is consistent across exp20 (AUROC 0.57) and exp21 (ρ 0.12) and is the project's central
deflationary result.

## Honest nuances / what didn't work as hoped
- **Direct-effect filtering only modestly sharpens.** Removing the top-10 global modes drops
  the reference density 0.695→0.632 but *increases* per-perturbation diffuseness (295→331
  effective responders): the global mode was a *coherent low-rank* structure, so subtracting
  it leaves a broader, less-structured residual. Global-mode removal reduces cross-perturbation
  redundancy, not per-response diffuseness. A cleaner "direct effect" needs a different
  operator (e.g. regress out total-UMI/cell-cycle explicitly, or mediator-adjustment) — exp22.
- **It also slightly lowers orientation reproducibility** (0.70→0.62): the dominant mode
  carries some genuine directional signal, so removing it costs a little. Honest trade-off.
- **Stability is only ~0.5 median at full scale** — half the perturbations are noisy; results
  should be read on the stable half.

## Engineering
New, general, tested library functions (`src/.../data/interventional.py`), all validated on
synthetic data so tests never need the 8.7 GB file: `perturbation_response_matrix`
(+ split-half), `response_low_rank` (SVD spectrum), `direct_effect_filter` (global-mode
removal), `response_sparsity` (effective-responder count), `split_half_stability`. +6 tests
(suite 131 green). These are deliberately representation-agnostic — the same delta-geometry
toolkit Track B uses, now shared across both tracks.

## Verdict and next step
The response-geometry reframing paid off: it (a) explained exp20's dense reference as a real
low-rank global response, (b) produced a **ground-truth-free, reproducible orientation
metric** (0.70 cross-split agreement), and (c) hardened the deflationary finding that
observational co-expression barely tracks interventional response (ρ≈0.12).

The unifying thesis across both tracks is now concrete:
**intervention → delta vector → geometry of hidden structure** — Track B in representation
space, Track A in perturbation-response space.

Best next step (exp22): a **better direct-effect operator** (regress out the explicit global
covariates — total UMI, detected genes, cell-cycle proxies — instead of blind top-mode
removal), then re-test whether observational structure aligns with the *cleaned* direct
response and whether an inferred sparse graph explains $D$ better than co-expression. Only
after a trustworthy direct-effect graph exists would a graph-geometry (graph-wavelet) step be
non-forced.
