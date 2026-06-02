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

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\01_dream4_size10_baseline\run_correlation_baseline.py
```

Outputs are written under:

```text
results/tables/
```

## Metrics

- `AUROC`: ranking quality across true and false directed edges.
- `AUPR`: precision-recall area, usually more informative when true edges are sparse.
- `precision@k`: fraction of true edges among the top `k` ranked candidate edges.

## Limitations

- Uses only Size10 multifactorial expression data.
- Does not use time-series data, perturbation labels, bootstrap resampling, or stability selection.
- LASSO uses a fixed regularization value rather than tuned or stability-calibrated sparsity.
- These are simple baseline rankings, not causal validation of inferred regulation.
