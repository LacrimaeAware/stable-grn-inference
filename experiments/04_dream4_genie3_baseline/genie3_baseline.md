# DREAM4 Size10 GENIE3 Baseline

This experiment adds an explicit GENIE3-style tree ensemble baseline for the DREAM4 Size10 data regimes. It compares ordinary correlation, bootstrap stability correlation, target-wise random-forest GENIE3, and target-wise Extra Trees GENIE3.

## Data

The script uses the local DREAM4 Size10 expression files for:

- `multifactorial`
- `knockouts`
- `knockdowns`
- `timeseries`

Each network is evaluated against the matching Size10 directed gold-standard topology. For time-series files, `Time` is dropped and rows are treated as same-time observations. This is not lagged time-series inference.

## Methods

- `one_shot_correlation`: absolute pairwise correlation, scored as directed non-self edges.
- `stability_correlation`: bootstrap resamples of rows, scored by mean reciprocal rank across resamples.
- `genie3_random_forest`: for each target gene, fit a `RandomForestRegressor` using all other genes as predictors; feature importances become source-to-target edge scores.
- `genie3_extra_trees`: same target-wise setup using `ExtraTreesRegressor`.

The GENIE3 ranker defaults to 1000 trees in the inference module. This experiment defaults to 500 trees per target so the full four-regime Size10 audit remains practical. Increase with `--n-estimators 1000` for a heavier run.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\04_dream4_genie3_baseline\run_genie3_baseline.py
```

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_genie3_baseline_summary.csv`
- `dream4_genie3_baseline_edges.csv`
- `dream4_genie3_baseline_debug_report.md`

## Current Results

With the default 500 trees per target:

| Data regime | Best mean AUPR | Best mean AUROC |
|---|---|---|
| `knockdowns` | `stability_correlation` 0.304808 | `one_shot_correlation` 0.650854 |
| `knockouts` | `genie3_extra_trees` 0.393540 | `stability_correlation` 0.732779 |
| `multifactorial` | `genie3_extra_trees` 0.379097 | `stability_correlation` 0.671269 |
| `timeseries` | `genie3_random_forest` 0.372960 | `genie3_extra_trees` 0.736178 |

GENIE3 improves mean AUPR over one-shot correlation in all four regimes and over stability correlation in knockouts, multifactorial, and timeseries. Stability correlation remains competitive and wins mean AUPR on knockdowns plus mean AUROC on knockouts and multifactorial.

## Limitations

This is a baseline audit, not the final research method. It does not implement stability-GENIE3, Size100 scaling, or lagged time-series inference yet.
