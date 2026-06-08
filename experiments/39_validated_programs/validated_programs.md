# Experiment 39: depth-controlled, externally-validated programs (RENGE day 5)

Applies the exp 38 bar: a program counts only if it is not depth-driven, not housekeeping, and
externally coherent (STRING-connected beyond a permutation null). NMF, k=12, on the 250 high-variance
genes of RENGE day 5.

## Result

The automated filter (depth corr < 0.3, ribosomal fraction < 0.3, STRING z > 2) passed 1 of 12
programs: HSP90AB1, ACTB, NCL, GAPDH, NPM1, PTMA (STRING z 4.1).

That "pass" is a filter artifact, not a result. Those genes are housekeeping (GAPDH and ACTB are
textbook housekeeping; HSP90AB1, NCL, NPM1, PTMA are abundant chaperone/nucleolar genes). The filter
only excluded ribosomal proteins (RPL/RPS), so it missed the broader housekeeping cluster. STRING
scores it high because housekeeping genes are densely annotated, not because the program is biological
signal or perturbation-relevant.

The other 11 programs were either depth-correlated (program 1 is mitochondrial, depth corr 0.76),
ribosomal (programs 5, 11, depth-linked), or not STRING-coherent at all (programs 0, 3, 4, 7 with
z <= 0).

## Verdict

Negative. After controlling for sequencing depth and housekeeping, no specific, externally-coherent
gene program survives in this data; the only reproducible, STRING-coherent structure is
housekeeping/abundance. The programs are co-expression across all cells, not knockout-specific, so
even a surviving program would not be a perturbation-specific finding. The script's auto-verdict
("programs DO exist") was wrong; reading the genes is what catches it.

## Outputs

Under `results/tables/` (git-ignored): `validated_programs_programs.csv`,
`validated_programs_summary.csv`, `validated_programs_debug_report.md`. STRING network cached at
`data/raw/string/renge_hvg250_string.tsv` (git-ignored).

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/39_validated_programs/run_validated_programs.py
```
