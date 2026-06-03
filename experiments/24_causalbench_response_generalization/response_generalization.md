# Experiment 24 — Is perturbation response *transferable* across perturbations? (RPE1)

*(My own follow-up, not the OpenAI prompt. The most optimistic test I could think of that we
hadn't run.)*

## The question

Every prior experiment hunted for **shared / global** structure (a low-rank code, a sparse
generator W, observational predictors) and mostly came up empty. This asks the most hopeful
remaining version of that: when we hold out one perturbation g, can we predict its true effect
**better** by leaning on the shared structure of all the *other* perturbations than by using
g's own noisy data alone — especially on the **gene-specific** part, not just the cell cycle?

If "yes," the response geometry has a transferable code worth modelling. If "no," each
perturbation is mostly its own thing.

Setup (leave-one-perturbation-out, two independent cell halves A and B):
- **target** = g's response on half B.
- **self_only** = g's own half-A response (its own noisy estimate).
- **mean_prog** = the average of the *other* perturbations' responses (the cell-cycle program).
- **lowrank_k** = project g's half-A response onto the top-k gene-space subspace learned from
  the *other* perturbations (the shared-structure "denoiser").

Scored by cosine to the held-out target, reported both **raw** and **residual** (cell-cycle
program removed — the honest, gene-specific metric).

**Pre-registered prediction:** low-rank denoising beats self-only on raw cosine (~70%); on the
residual it's genuinely 50/50, and that's the crux.

## Result (full run: 651 genes, 300 held-out perturbations)

| method | raw cosine | residual cosine (gene-specific) |
| --- | --- | --- |
| **self_only** | **0.514** | **0.409** |
| mean_prog (cell cycle) | 0.327 | ~0.000 |
| lowrank_1 | 0.371 | 0.071 |
| lowrank_5 | 0.418 | 0.230 |
| lowrank_20 | 0.463 | 0.316 |
| lowrank_50 | 0.477 | 0.342 |

Every low-rank denoiser **underperforms self-only**, on both metrics. As k grows, low-rank just
creeps back toward self-only (projecting onto everything = identity). The shared subspace
doesn't add information; it only throws away the part of g's own response that lies outside the
dominant shared directions — and that discarded part is real signal.

## Verdict — NEGATIVE (and it disagreed with my own prediction)

> Shared structure does **not** denoise a held-out perturbation beyond its own estimate. The
> response geometry is **not transferably predictive** across perturbations via a low-rank code.

I had put ~50% on the residual showing transferable gene-specific structure. It didn't. That's
the value of pre-registering: this is a confirmed negative, not a disappointment I can re-spin.

## The honest, non-doom nuance (this is the useful part)

The negative is about **transfer**, not about whether the signal is real:

- **self_only residual cosine = 0.41** means each perturbation's *gene-specific* response (with
  the cell cycle removed) is genuinely **reproducible** across independent cells. There is real
  per-perturbation structure beyond the global program.
- It is just **individualistic**: knowing how the other 650 knockdowns behave does **not** help
  predict the 651st. The reliable unit is the *single perturbation*, not a shared subspace.

Put together with exp 22 (cleaning failed) and exp 23 (inverse failed), the three negatives all
point the same way: **the recoverable signal lives in individual perturbations and their
pairwise relationships, not in any shared global/low-rank structure.** That convergence is
itself a finding — it tells you where *not* to keep digging.

## What should be committed
- `experiments/24_causalbench_response_generalization/` (script + this note) and its unit tests
  (`tests/test_response_generalization.py`, synthetic only). No new library code; reuses the
  response-matrix tooling. `data/` and `results/` stay git-ignored.
