# DREAM4 Size10 Baseline Comparison

This experiment runs first-pass edge-ranking baselines on the DREAM4 Size10 multifactorial expression datasets.

## Data

Expression inputs:

```text
data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_{i}/insilico_size10_{i}_multifactorial.tsv
```

Gold-standard topology labels:

```text
data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 10/DREAM4_GoldStandard_InSilico_Size10_{i}.tsv
```

The expression files are the input data. The gold-standard files are the answer keys for directed network topology.

## Baselines

- `correlation`: ranks all directed non-self edges by absolute pairwise gene-expression correlation.
- `lasso_alpha_0_01`: for each target gene, fits a LASSO model using all other genes as predictors, then ranks directed edges by absolute coefficient magnitude.
- `run_method_comparison.py` also audits a small LASSO alpha grid, a small Elastic Net grid, and a random-forest feature-importance baseline.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\01_dream4_size10_baseline\run_correlation_baseline.py
```

Run the broader comparison/audit checkpoint:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\01_dream4_size10_baseline\run_method_comparison.py
```

Outputs are written under:

```text
results/tables/
```

## Metrics

- `AUROC`: ranking quality across true and false directed edges.
- `AUPR`: precision-recall area, usually more informative when true edges are sparse.
- `precision@k`: fraction of true edges among the top `k` ranked candidate edges.

## Current Takeaways

The first Size10 multifactorial comparison is an audit checkpoint, not a final method claim. On mean AUROC and mean AUPR across the five Size10 networks, absolute correlation is still the strongest baseline in the current results.

Tuning the LASSO regularization strength improved over the original fixed `alpha=0.01` run. The best sparse candidate so far is `lasso_alpha_0_1`: it does not beat correlation, but it is the most promising sparse estimator in this first grid and is close enough to justify testing as the base model for stability selection.

The random-forest feature-importance baseline is competitive as an edge-ranking audit method, but it is less aligned with the planned sparse/stability path because it does not naturally produce a sparse selected-edge set.

Next planned step: add stability selection using `lasso_alpha_0_1` as the base sparse estimator, while keeping correlation as the simple baseline for comparison.

## Limitations

- Uses only Size10 multifactorial expression data.
- Does not use time-series data, perturbation labels, bootstrap resampling, or stability selection.
- LASSO uses a fixed regularization value rather than tuned or stability-calibrated sparsity.
- The method-comparison script includes small grids, but does not use cross-validation or held-out model selection.
- These are simple baseline rankings, not causal validation of inferred regulation.
