# DREAM4 Size10 Lagged Time-Series Audit

This experiment is the first temporal-order audit for DREAM4 Size10. It tests whether using adjacent time pairs improves directed edge recovery compared with same-time scoring.

## Data

The script uses the Size10 `*_timeseries.tsv` files. Each file has a `Time` column and gene-expression columns. Trajectories are split when `Time` resets, then lagged samples are built only within each trajectory:

```text
X = gene expression at time t
Y = gene expression at time t + 1
```

The candidate edge score is interpreted as source gene at `t` influencing target gene at `t+1`. Self-lag edges are excluded to match the directed non-self DREAM4 candidate edge set used in earlier experiments.

## Methods

- `same_time_correlation`
- `same_time_lasso_alpha_0_1`
- `same_time_genie3_random_forest`
- `same_time_genie3_extra_trees`
- `lagged_correlation`
- `lagged_lasso_alpha_0_01`
- `lagged_lasso_alpha_0_03`
- `lagged_lasso_alpha_0_1`
- `lagged_genie3_random_forest`
- `lagged_genie3_extra_trees`

## Metrics

The script computes AUROC, AUPR, precision@5, precision@10, precision@20, and selected topology-aware metrics at top N true edges:

- out-degree Spearman
- in-degree Spearman
- top-3 out-hub overlap
- top-3 in-hub overlap
- reciprocal false-positive pair rate

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\07_dream4_size10_lagged_timeseries\run_lagged_timeseries_audit.py
```

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_size10_lagged_timeseries_summary.csv`
- `dream4_size10_lagged_timeseries_edges.csv`
- `dream4_size10_lagged_timeseries_debug_report.md`

## Current Results

Each Size10 time-series file split into 5 trajectories with 105 rows total and 100 within-trajectory lagged samples.

Mean edge metrics across the five Size10 networks:

| Method | Variant | AUROC | AUPR | P@5 | P@10 | P@20 |
|---|---|---:|---:|---:|---:|---:|
| `lagged_genie3_random_forest` | lagged | 0.767932 | 0.531535 | 0.68 | 0.54 | 0.40 |
| `lagged_genie3_extra_trees` | lagged | 0.767890 | 0.528333 | 0.68 | 0.58 | 0.37 |
| `lagged_lasso_alpha_0_1` | lagged | 0.755521 | 0.509534 | 0.68 | 0.50 | 0.40 |
| `lagged_correlation` | lagged | 0.712754 | 0.458295 | 0.64 | 0.48 | 0.35 |
| `same_time_genie3_random_forest` | same-time | 0.725080 | 0.373040 | 0.44 | 0.40 | 0.34 |
| `same_time_lasso_alpha_0_1` | same-time | 0.683267 | 0.336698 | 0.40 | 0.38 | 0.29 |
| `same_time_correlation` | same-time | 0.653955 | 0.302771 | 0.36 | 0.34 | 0.30 |

Lagged modeling substantially improves edge AUROC/AUPR in this first audit. `lagged_genie3_random_forest` is the strongest first temporal baseline by mean AUPR and AUROC. Lagged LASSO also improves over same-time LASSO and has the best top-3 in-hub overlap among tested methods.

Topology remains mixed. Lagged methods improve some in-hub metrics, but they do not clearly solve reciprocal-direction false positives. For example, the best lagged GENIE3 method reduces reciprocal false-positive pair rate slightly versus same-time GENIE3, while lagged correlation and lagged LASSO have higher reciprocal false-positive rates than their same-time references.

## Limitations

This is a first temporal audit, not a final dynGENIE3 implementation. It uses one-step adjacent lags only, excludes self-lag candidates, and does not yet add stability over lagged models.
