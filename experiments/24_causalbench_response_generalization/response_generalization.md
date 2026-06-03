# Experiment 24: is the perturbation response transferable across perturbations? (RPE1)

## Question

Earlier experiments tested for shared/global structure (a low-rank code, a sparse generator W, observational predictors) and found little. This experiment tests whether holding out one perturbation g, its true effect can be predicted better from the shared structure of the other perturbations than from g's own noisy data, specifically on the gene-specific part rather than the cell-cycle program.

Setup (leave-one-perturbation-out, two independent cell halves A and B):
- target: g's response on half B.
- self_only: g's own half-A response.
- mean_prog: the average of the other perturbations' responses (the cell-cycle program).
- lowrank_k: project g's half-A response onto the top-k gene-space subspace learned from the other perturbations.

Scored by cosine to the held-out target, reported both raw and residual (cell-cycle program removed, the gene-specific metric).

Expected behavior: low-rank denoising was expected to beat self_only on raw cosine; the residual was the uncertain case.

## Result (651 genes, 300 held-out perturbations)

| method | raw cosine | residual cosine (gene-specific) |
| --- | --- | --- |
| self_only | 0.514 | 0.409 |
| mean_prog (cell cycle) | 0.327 | ~0.000 |
| lowrank_1 | 0.371 | 0.071 |
| lowrank_5 | 0.418 | 0.230 |
| lowrank_20 | 0.463 | 0.316 |
| lowrank_50 | 0.477 | 0.342 |

Every low-rank denoiser underperforms self_only on both metrics. As k grows, low-rank converges toward self_only (projecting onto the full space equals identity). The shared subspace does not add information; it discards the part of g's response that lies outside the dominant shared directions, and that discarded part is real signal.

Result: shared structure does not denoise a held-out perturbation beyond its own estimate. The response geometry is not transferably predictive across perturbations via a low-rank code.

## Nuance

The negative is about transfer, not about whether the signal is real:
- self_only residual cosine 0.41 means each perturbation's gene-specific response (cell cycle removed) is reproducible across independent cells. Per-perturbation structure beyond the global program is real.
- It is individualistic: the other 650 knockdowns do not help predict the 651st. The reliable unit is the single perturbation, not a shared subspace.

With experiments 22 (cleaning failed) and 23 (inverse failed), the three negatives converge: the recoverable signal lives in individual perturbations and their pairwise relationships, not in shared global/low-rank structure.

## Files
- `experiments/24_causalbench_response_generalization/` (script and this note) and its unit tests (`tests/test_response_generalization.py`, synthetic only). No new library code; reuses the response-matrix tooling. `data/` and `results/` are git-ignored.
