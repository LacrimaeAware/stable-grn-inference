# Experiment 37: interpretable programs and single-cell response heterogeneity (RENGE)

The reframe, both directions in one pass on real single-cell CRISPR data (RENGE day 5, 5,693 cells,
1,500 high-variance genes, 23 knocked-out TFs). The bar here is reproducibility and meaning, not
beating GENIE3 on edges. RPE1 was not present in this tree, so RENGE day 5 (real cells) is the
testbed via the new high-variance-gene loader.

## Part A: interpretable program atlas

Decompose the single-cell expression into k gene programs (NMF, with PCA as the linear baseline) and
judge them by reproducibility across independent cells.

| k programs | NMF reproducibility | PCA reproducibility |
| --- | --- | --- |
| 5  | 0.939 | 0.885 |
| 10 | 0.947 | 0.873 |
| 20 | 0.879 | 0.659 |

- Interpretable programs are highly reproducible (NMF up to 0.95), and NMF beats PCA, most clearly at
  k=20 (0.88 vs 0.66). The structured, interpretable method wins at the reproducibility bar.
- The dominant program is the ribosomal / translation-machinery program (RPLP1, RPS12, RPL41, RPL28,
  RPL8, RPS3, GAPDH, RPLP0, RPL11, ...), a real, named biological program. Caveat: the most
  reproducible programs are the large housekeeping ones (ribosome), which are the easy case.

## Part B: single-cell response heterogeneity (the open problem; the "ripples")

For each knockout, the cells deviate from the population-mean response. Is that per-cell deviation
real signal?

| mean top-direction variance fraction | mean reproducibility | mean alignment with cell-state axis |
| --- | --- | --- |
| 0.202 | 0.727 | 0.889 |

- Structured: the per-cell deviation has a dominant direction holding 20% of the residual variance
  (isotropic noise would be about 1/1500 = 0.07%). The heterogeneity is low-rank, not noise.
- Reproducible: that deviation direction recurs across independent halves of a knockout's cells at
  0.73. It is a real, stable axis of variation, not a fitting artifact.
- Interpretable: the deviation aligns with the control cells' dominant state axis at 0.89. Cells
  deviate from the mean knockout response largely along their own cell-cycle / cell-state position.

## What it says (honest)

This is a genuine positive on real data, and unlike the edge-recovery experiments it is not beaten by
the trivial baseline: interpretable programs are reproducible and NMF beats PCA, and the single-cell
response heterogeneity is real, structured, reproducible, and interpretable. Cells do not respond by
the population mean; they deviate reproducibly, and the deviation tracks cell state. This is the
project's "ripples under the dominant mode" intuition, confirmed on the field's stated open problem.

The caveat that sets up the next step: the heterogeneity is largely explained by cell state
(alignment 0.89). That is interpretable and real, but it means the dominant per-cell deviation is
cell-cycle position, not necessarily perturbation-specific regulatory structure. The genuinely novel
question is therefore the residual: after removing the cell-state axis, is there reproducible
knockout-SPECIFIC heterogeneity (a ripple beyond the cell-cycle mode)? That is exactly the
ripple-under-the-dominant-mode question, now posed one level deeper, and it is the next experiment.

Limitations: one day, one dataset; reproducibility is split-half consistency, not cross-dataset or
external validation; the loader selects high-variance genes by raw-count variance.

## Outputs

Under `results/tables/` (git-ignored): `programs_and_heterogeneity_programs.csv`,
`programs_and_heterogeneity_heterogeneity.csv`, `programs_and_heterogeneity_summary.csv`,
`programs_and_heterogeneity_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/37_programs_and_heterogeneity/run_programs_and_heterogeneity.py
```
