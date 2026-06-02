# DREAM4 Size10 Dynamic Sparse Validation

This experiment stress-tests the strongest dynamic sparse-linear result from the broad dynamic batch audit. The goal is to check whether the result is robust, interpretable, and not just an artifact of one alpha or self-predictor choice.

## Data

The audit uses DREAM4 Size10 time-series files. Trajectories are split when `Time` resets, then adjacent lagged samples are built only within each trajectory:

```text
X_t = gene expression at time t
Y_level = gene expression at time t+1
Y_delta = gene expression at time t+1 - gene expression at time t
```

Gold-standard Size10 directed networks are used as the topology answer key.

## Methods

Focused sparse-linear grid:

- LASSO level target, include self predictor during fitting, no self-edge output
- LASSO level target, exclude self predictor
- LASSO delta target, include self predictor during fitting, no self-edge output
- LASSO delta target, exclude self predictor
- Elastic Net level target, include self predictor during fitting, no self-edge output
- Elastic Net delta target, include self predictor during fitting, no self-edge output

Reference methods:

- lagged correlation reference
- lagged GENIE3-style random forest
- lagged GENIE3-style Extra Trees

LASSO alphas are `0.003`, `0.01`, `0.03`, `0.1`, `0.3`, and `1.0`. Elastic Net uses alphas `0.01`, `0.03`, and `0.1` with l1 ratios `0.3`, `0.7`, and `0.95`.

## Validation Checks

- per-network metric breakdown to check whether the mean is dominated by one or two networks
- alpha sensitivity across the sparse grid
- include-self versus exclude-self matched comparisons
- self-predictor coefficient diagnostics for include-self models
- trajectory-bootstrap selection frequency and mean absolute coefficient rankings for selected sparse candidates
- reciprocal-direction false-positive rates
- topology-aware metrics, including degree Spearman, top-hub overlap, reciprocal errors, and feed-forward-loop error

## Current Results

Default run settings used 500 trees for lagged GENIE3-style references and 50 trajectory-bootstrap resamples for selected sparse candidates.

Best mean edge metrics across five Size10 networks:

| Metric | Winning Method | Value |
|---|---|---:|
| AUPR | `dynamic_lasso_level_include_self_a0_03` | 0.652712 |
| AUROC | `dynamic_lasso_level_include_self_a0_03` | 0.821067 |
| precision@10 | `dynamic_lasso_level_include_self_a0_03` | 0.680000 |

Reference comparison:

| Method | Mean AUROC | Mean AUPR | Mean Reciprocal FP Pair Rate |
|---|---:|---:|---:|
| `dynamic_lasso_level_include_self_a0_03` | 0.821067 | 0.652712 | 0.200000 |
| `lagged_genie3_random_forest` | 0.767974 | 0.536451 | 0.950000 |
| `lagged_genie3_extra_trees` | 0.765530 | 0.515946 | 1.000000 |
| `lagged_correlation_reference` | 0.712754 | 0.458295 | 1.000000 |

Validation takeaways:

- `alpha=0.03` is the best LASSO alpha by mean AUPR in this grid.
- The winner is not universal: it is the per-network AUPR winner on 2 of 5 networks.
- Include-self improves matched LASSO AUPR in most target/alpha comparisons, especially for `level` and moderate alphas.
- The include-self models have strong self-predictor coefficients. For the winning model, mean absolute self coefficient is about 8.9 times the mean absolute non-self coefficient, so persistence is clearly important.
- Bootstrap mean-absolute-coefficient ranking nearly matches the one-shot winner but does not improve it. Bootstrap selection-frequency ranking hurts the include-self candidates in this audit.
- The best sparse dynamic model has a much lower reciprocal false-positive pair rate than lagged GENIE3 or lagged correlation references.
- Topology is encouraging but not perfect: the edge-metric winner has top-3 out-hub overlap 0.466667 and top-3 in-hub overlap 0.600000, while other sparse variants win the best hub-overlap metrics.

Current interpretation: `dynamic_lasso_level_include_self_a0_03` is the strongest Size10 temporal sparse candidate so far, but the self-persistence mechanism needs validation on richer data before treating it as the main method.

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_size10_dynamic_sparse_validation_summary.csv`
- `dream4_size10_dynamic_sparse_validation_per_network.csv`
- `dream4_size10_dynamic_sparse_validation_edges.csv`
- `dream4_size10_dynamic_sparse_validation_topology.csv`
- `dream4_size10_dynamic_sparse_validation_debug_report.md`

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\09_dream4_size10_dynamic_sparse_validation\run_dynamic_sparse_validation.py
```

## Interpretation Policy

This is a validation audit, not a final thesis result. A method should only become the current main candidate if it performs well across networks, is not hypersensitive to one alpha, has plausible self-persistence diagnostics, and does not fail badly on topology-aware metrics.
