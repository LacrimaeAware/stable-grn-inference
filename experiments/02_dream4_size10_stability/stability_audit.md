# DREAM4 Size10 Stability Audit

This experiment asks whether resampling-based stability scores make edge ranking more informative on the DREAM4 Size10 multifactorial datasets.

## Data

Expression inputs:

```text
data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_{i}/insilico_size10_{i}_multifactorial.tsv
```

Gold-standard topology labels:

```text
data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/DREAM4_Challenge2_GoldStandards/Size 10/DREAM4_GoldStandard_InSilico_Size10_{i}.tsv
```

## Methods

- `one_shot_correlation`: absolute correlation on the full expression matrix.
- `stability_correlation`: bootstrap each expression matrix, rerank edges by correlation, and score edges by mean reciprocal rank.
- `one_shot_lasso_alpha_0_1`: target-wise LASSO coefficient magnitudes using `alpha=0.1`.
- `stability_lasso_alpha_0_1`: bootstrap each expression matrix and score edges by nonzero coefficient selection frequency.
- `one_shot_elastic_net_alpha_0_03_l1_0_95` and `stability_elastic_net_alpha_0_03_l1_0_95`: sparse audit variants from the earlier Elastic Net grid.
- `one_shot_random_forest_importance` and `stability_random_forest_importance`: non-sparse feature-importance audit comparators.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\02_dream4_size10_stability\run_stability_audit.py
```

Defaults are `100` bootstrap resamples, fixed seed `20260602`, and `10` trees for the random-forest audit baseline. The small forest size keeps the repeated-resampling audit practical; it should not be read as a tuned random-forest result.

Subsampling is also supported:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\02_dream4_size10_stability\run_stability_audit.py --resampling-method subsample
```

## Outputs

```text
results/tables/dream4_size10_stability_summary.csv
results/tables/dream4_size10_stability_edges.csv
results/tables/dream4_size10_network1_stability_debug_report.md
```

## Current Takeaways

With the default 100 bootstrap resamples on Size10 multifactorial data:

- `stability_correlation` has the best mean AUPR among the audited methods.
- `one_shot_correlation` still has the best mean AUROC.
- `stability_lasso_alpha_0_1` improves over one-shot `lasso_alpha_0_1` on mean AUROC/AUPR, but does not beat correlation.
- Random forest remains useful as a non-sparse audit comparator, but it is not the main path for stability-aware sparse inference.
- The result supports continuing stability-selection experiments, but the evidence is still small-data and should be tested on richer inputs such as time-series, knockouts/knockdowns, and eventually Size100.

## Limitations

This is a fragile small-data audit. Each Size10 multifactorial matrix has only 10 rows, so bootstrap stability can reveal ranking behavior but should not be treated as a final thesis result. The goal is to decide whether stability-aware sparse ranking is worth pursuing on richer data or with better experimental design.
