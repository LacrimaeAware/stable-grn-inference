# Experiment 29: whitened interventional-asymmetry gate (RPE1)

Direction A from `docs/next_direction.md`. The honest resolution of the
pairwise-difference intuition, scoped to the one object that can carry direction.

## Question

After accounting for the two reproducible per-gene axes (net_out cascade position,
0.986 in exp 26; response magnitude, 0.97 in exp 26), is there any reproducible
orientation asymmetry left in the response matrix, and does whitening the dominant mode
(downweighting it, not subtracting it) recover more of it than the raw asymmetry?

The object is the square response block `M[g,h] = Delta_g[h]` on the perturbed and
measured genes, and its asymmetry `A = |M| - |M|^T`. This is the correction from the
original proposal: the asymmetry lives in the off-diagonal pair `(M[g,h], M[h,g])`, not
in the row difference `Delta_g - Delta_h`, which cannot contain `M[h,g]`.

## Method

Two gates. Only Gate 0 runs here.

Gate 0 (cheap, no external data). For each whitening strength `alpha` in
{0, 0.25, 0.5, 0.75, 1.0}: whiten each cell-half response matrix
(`fractional_whiten`, SVD-based, `alpha=0` raw, `alpha=1` full ZCA), form the asymmetry,
remove its per-gene fit (the antisymmetric lifts of net_out and magnitude), and measure
the split-half reproducibility of the residual. The residual is the pairwise asymmetry
NOT recoverable from the two known axes, so its reproducibility is the non-circular
question. Pass if some `alpha` gives a reproducible residual (> 0.10); whitening helps if
the best `alpha > 0` beats `alpha = 0` by a margin.

Gate 1 (the next experiment, gated on Gate 0, not run here). Validate the surviving
asymmetry against external anchors held out by gene: DepMap gene-effect for per-gene
severity, CORUM/STRING co-membership for the relational part. This is where
reproducibility becomes correctness. Literature effect sizes that make these anchors
usable are in `docs/next_direction.md` section 8 (CORUM median r=0.61 in Replogle;
co-essentiality enriches roughly 160-fold for CORUM under GLS).

Tooling: `src/stable_grn_inference/analysis/asymmetry.py`
(`response_asymmetry`, `net_out`, `residualize_asymmetry`, `fractional_whiten`,
`pairwise_reproducibility`), tested in `tests/test_asymmetry.py`. The script runs on the
real RPE1 `h5ad` when present, else an offline synthetic DAG fixture as a positive control.

## Expected outcome

A clean negative on RPE1, predicted by exp 28: RPE1 sits below the SNR floor, and
whitening (like deflation) fixes dominant-mode dominance, not SNR. The value of running
it is closing the pairwise/whitening question and delivering the external-anchor
validation harness (Gate 1) that exp 26 flagged and never built. A pass would be a
genuine surprise and would justify Gate 1 immediately.

## Interpretation policy

Reproducibility is consistency across cell halves, not correctness. A nonlinear or
whitened score can be reproducible precisely because it re-encodes the cascade; only the
residual-after-per-gene-fit and, in Gate 1, the external anchors speak to correctness. No
external ground truth is used in Gate 0.

## Outputs

Under `results/tables/` (git-ignored): `whitened_asymmetry_sweep.csv`,
`whitened_asymmetry_summary.csv`, `whitened_asymmetry_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/29_whitened_asymmetry/run_whitened_asymmetry.py
.\.venv\Scripts\python.exe -B experiments/29_whitened_asymmetry/run_whitened_asymmetry.py --synthetic   # offline positive control
```
