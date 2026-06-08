# Experiment 38: heterogeneity audit (RENGE day 5)

A deliberate adversarial check of the exp 37 heterogeneity "result," which reported that the per-cell
deviation from each knockout's mean response was structured, reproducible, and aligned with a single
cell-state axis. That signature (ribosomal-dominated programs + one shared axis) is also the textbook
signature of a technical / global confound (library size, sequencing depth, cell size), so it had to
be tested before being believed.

## Method (three tests)

1. Technical confound: correlate each cell's position on the cell-state / deviation axis with raw
   total UMI (library size) and detected-gene count.
2. Knockout-specificity: pairwise cosine of the 23 per-knockout deviation axes (near 1 = one shared
   global axis, trivial; near 0 = knockout-specific).
3. Residual: regress out library size, project out the global axis, and re-test for reproducible,
   knockout-specific heterogeneity.

Tooling: `analysis/programs.py` (`heterogeneity_structure`) and the RENGE high-variance loader with
per-cell total UMI (`load_renge_day_hvg(..., return_total_umi=True)`).

## Result

| test | quantity | value |
| --- | --- | --- |
| 1 technical | corr(cell-state axis, log total UMI) | 0.849 |
| 1 technical | corr(cell-state axis, detected-gene count) | 0.964 |
| 1 technical | mean per-knockout corr(deviation, log UMI) | 0.817 |
| 2 global | mean pairwise cosine of the 23 deviation axes | 0.809 |
| 2 global | mean alignment of deviation axes with the global axis | 0.889 |
| 3 residual | residual top-direction variance fraction | 0.044 |
| 3 residual | residual reproducibility across cell halves | 0.373 |
| 3 residual | residual deviation-axis pairwise cosine | 0.361 |

## Verdict

- The heterogeneity axis is largely technical: it correlates with library size at 0.85 and with
  detected-gene count at 0.96. It is essentially sequencing depth.
- The raw heterogeneity is one global axis shared across all knockouts (pairwise cosine 0.81), aligned
  with the global cell-state / depth axis (0.89). It is not knockout-specific.
- After removing library size and the global axis, the residual is not reproducible (0.37, versus
  > 0.85 for real planted signal) and not reliably knockout-specific. No biological ripple survives.

Honest reading: exp 37's heterogeneity was a technical / global confound (library size), not a
discovery. The structured-reproducible-aligned result was expected, not novel. This is a clean
negative, and a true one.

## Why this experiment is the actual contribution

The audit is reusable and it makes the methodological lesson concrete: internal reproducibility is not
evidence of biology, because technical axes (library size) are highly reproducible too. Any future
"structure" claim in this project must (a) be checked against technical confounds (depth), (b) show
specificity (here, knockout-specificity), and (c) validate against EXTERNAL biology (enrichment), not
just internal recurrence. That is the standing bar going forward.

## Outputs

Under `results/tables/` (git-ignored): `heterogeneity_audit_per_perturbation.csv`,
`heterogeneity_audit_summary.csv`, `heterogeneity_audit_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/38_heterogeneity_audit/run_heterogeneity_audit.py
```
