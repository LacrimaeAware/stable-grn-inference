# DREAM4 Size10 Data-Regime Audit

This experiment checks whether the stability-ranking effect seen on Size10 multifactorial data persists across other DREAM4 Size10 data regimes.

## Data Regimes

Included if present for all five Size10 networks:

- `multifactorial`
- `knockouts`
- `knockdowns`
- `timeseries`

The time-series files include a `Time` column. This audit drops `Time` and treats all remaining rows as same-time expression observations. It does not implement lagged time-series inference.

## Methods

- `one_shot_correlation`
- `stability_correlation`
- `one_shot_lasso_alpha_0_1`
- `stability_lasso_alpha_0_1`
- `one_shot_random_forest_importance`
- `stability_random_forest_importance`

Stability methods use bootstrap resampling by default. Correlation stability is scored by mean reciprocal rank across resamples. LASSO stability is scored by nonzero coefficient selection frequency. Random-forest stability is scored by mean feature importance and is included only as a non-sparse audit comparator.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\03_dream4_size10_data_regimes\run_data_regime_audit.py
```

Defaults are `100` bootstrap resamples, fixed seed `20260602`, and `10` trees for the random-forest audit comparator.

## Outputs

```text
results/tables/dream4_size10_data_regime_summary.csv
results/tables/dream4_size10_data_regime_edges.csv
results/tables/dream4_size10_data_regime_debug_report.md
```

## Current Takeaways

With the default 100 bootstrap resamples:

- All four requested regimes were included: `multifactorial`, `knockouts`, `knockdowns`, and `timeseries`.
- Stability correlation improved mean AUPR over one-shot correlation in every included regime.
- Stability correlation improved mean AUROC in `multifactorial`, `knockouts`, and `timeseries`, but not in `knockdowns`.
- Stability LASSO improved mean AUPR in `multifactorial` and `knockdowns`, but lost mean AUPR in `knockouts` and `timeseries`.
- `knockouts` favored one-shot `lasso_alpha_0_1` by mean AUPR, which is the strongest sign so far that sparse regression may behave better on perturbation-rich data than on multifactorial data alone.
- `timeseries` favored random forest by the current same-time scoring audit, but this should not be overread because proper lagged inference has not been implemented.

Best mean AUPR by regime:

| data regime | best method | mean AUPR |
|---|---|---:|
| `knockdowns` | `stability_correlation` | 0.289342 |
| `knockouts` | `one_shot_lasso_alpha_0_1` | 0.424674 |
| `multifactorial` | `stability_correlation` | 0.365318 |
| `timeseries` | `one_shot_random_forest_importance` | 0.369037 |

Best mean AUROC by regime:

| data regime | best method | mean AUROC |
|---|---|---:|
| `knockdowns` | `one_shot_correlation` | 0.650854 |
| `knockouts` | `stability_correlation` | 0.718376 |
| `multifactorial` | `stability_correlation` | 0.674167 |
| `timeseries` | `stability_random_forest_importance` | 0.718962 |

## Limitations

This is a validation audit, not a final result. It still uses Size10 networks and same-time association scoring. The time-series regime is included only to test behavior of the existing scoring machinery; proper lagged inference is a separate next step.
