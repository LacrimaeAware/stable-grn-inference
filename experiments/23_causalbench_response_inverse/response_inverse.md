# Experiment 23 — Response inverse / deconvolution ("solve the flow field for the stick")

## The idea (your stick-in-the-water, as math)

The response matrix D we measured is the **total** effect of each perturbation: direct wiring
*plus* everything that propagated downstream *plus* the global cell-state shift. In a simple
linear-propagation model, a sparse **direct** operator W generates the total response by
spreading out:

> D = (I − W)⁻¹ − I = W + W² + W³ + …  (direct + indirect + …)

and the **exact inverse** recovers the direct wiring: **W = I − (I + D)⁻¹**. That is literally
"given the whole flow field, solve backward for the stick that bent it." This experiment asks
whether any simple version of that inversion turns the dense total response into a sparser,
more stable, more *direct*-looking operator on real data.

## Pre-registered prediction (so the verdict is honest either way)

- Synthetic, model TRUE: noiseless inverse recovers W exactly; degrades gracefully with noise.
- Real RPE1: I expected the raw inverse to **not** beat the raw |D| baseline — matrix inversion
  amplifies noise, so it should be **less** split-half stable and should **not** reconstruct
  held-out response better. ~65% "mixed/failed." The one genuine unknown: whether a *sparse*
  (Lasso) deconvolution would be more stable.

## Part 0 — synthetic sanity check (mandatory)

Generated a sparse 50-gene DAG W_true, forward-propagated to D, and inverted, at several noise
levels. Recovery of the true direct edges (|W| ranked against the true nonzeros):

| noise | inverse AUPR(W) | raw \|D\| AUPR | exact-recovery error |
| --- | --- | --- | --- |
| 0% | **1.00** | 0.86 | 0.0 (exact) |
| 5% | 0.995 | 0.85 | small |
| 10% | 0.96 | 0.84 | small |
| 25% | 0.89 | 0.70 | moderate |
| 50% | 0.73 | 0.66 | large |

**The machinery is correct and genuinely useful when the model holds:** noiseless recovery is
exact, and the inverse beats the raw response at finding direct edges through moderate noise.
This is the necessary control — if it had failed here, no real-data result could be trusted.

## Part 1–3 — real RPE1 (651 × 651 response operator)

| operator | participation ratio | split-half edge stability | direction reproducibility | reconstruction (D̂₁ vs D₂) | global-mode alignment |
| --- | --- | --- | --- | --- | --- |
| **raw \|D\|** (baseline) | 3.45 | **0.345** | **0.619** | **0.402** | 1.00 |
| ridge inverse (λ=0.1) | 1.55 (sparser) | 0.091 | 0.531 | 0.402 | 0.05 |
| ridge inverse (λ=1.0) | 1.44 (sparser) | 0.188 | 0.524 | 0.402 | 0.09 |
| pinv inverse | 264 (unstable) | 0.042 | 0.514 | −0.000 | 0.04 |
| sparse Lasso deconv | 4.13 | 0.314 | 0.999* | −0.010 | 0.02 |

**The inverse does not help on real data.** It can make the operator sparser and it strips out
the global cell-cycle mode (alignment 1.0 → ~0.05), but it is **less split-half stable** than
the raw response (0.345 → 0.09–0.19), does **not** improve direction reproducibility (0.62 →
0.52), and does **not** reconstruct held-out response any better. No inverse operator beat raw
|D| on both stability and reconstruction.

\*The sparse Lasso deconvolution's apparent **0.999 "direction reproducibility" is an artifact,
not a result**: it produces a near-all-zeros operator, and the metric counts "both directions
are zero" as agreement. Its true reconstruction is ≈0 (−0.01), confirming it captures almost
nothing of the response. Flagged honestly rather than reported as a win.

## Verdict — MIXED, leaning NEGATIVE (as predicted)

> The linear-propagation model that makes the inverse work *perfectly on synthetic data*
> does **not** hold well enough on real RPE1 for the inversion to add value. The inverse is
> sparser but noisier; it buys nothing over the raw response.

This matches the pre-registration. It is a **clean, bounded swing that mostly failed on real
data** — and that failure is itself informative: it rules out the tempting "just deconvolve the
response into a sparse graph" route, and tells us the real perturbation response is not
well-described by a simple linear (I − W)⁻¹ generator (noise, nonlinearity, and the cell-cycle
confound break the assumption). Do **not** center the project on inverse-response.

## What should be committed
- `src/.../data/interventional.py`: `make_sparse_dag`, `propagation_forward`,
  `deconvolve_response`, `operator_edges` (+ tests, incl. the exact-recovery gate).
- `experiments/23_causalbench_response_inverse/` (script + this note).
- doc updates. `data/` and `results/` stay git-ignored; tests never touch the real file.
