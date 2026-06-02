# DREAM4 Size10 Topology Evaluation

This experiment evaluates whether top-ranked DREAM4 Size10 edges recover graph-level structure, not only individual true edges.

## Input

The script reads the edge-level output from the faithful GENIE3 audit:

```text
results/tables/dream4_genie3_baseline_edges.csv
```

That file contains rankings for:

- `one_shot_correlation`
- `stability_correlation`
- `genie3_random_forest`
- `genie3_extra_trees`

The data regimes are multifactorial, knockouts, knockdowns, and time-series treated as same-time observations with `Time` dropped.

## Metrics

For each network, data regime, method, and cutoff, the ranked edge list is converted into a predicted directed graph. Cutoffs are top 5, top 10, top 20, and top N true edges.

Topology metrics include:

- edge precision at the cutoff
- out-degree and in-degree Spearman correlation
- top-1 and top-3 out-hub overlap
- top-1 and top-3 in-hub overlap
- reciprocal pair counts and reciprocal false-positive rates
- reciprocal edge count error
- feed-forward loop count error
- precision and recall for edges incident to the top true out-hub and in-hub

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\06_dream4_size10_topology_evaluation\run_topology_evaluation.py
```

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_size10_topology_summary.csv`
- `dream4_size10_topology_details.csv`
- `dream4_size10_topology_debug_report.md`

## Current Results

The primary interpretation uses `top_n_true_edges`, which thresholds each predicted graph to the same edge count as the matching gold-standard network.

Best top-3 out-hub recovery:

| Data regime | Best method | Top-3 out-hub overlap | Out-degree Spearman |
|---|---|---:|---:|
| `knockdowns` | `one_shot_correlation` | 0.466667 | 0.345761 |
| `knockouts` | `one_shot_correlation` | 0.533333 | 0.363751 |
| `multifactorial` | `one_shot_correlation` | 0.400000 | 0.228912 |
| `timeseries` | `genie3_random_forest` | 0.400000 | 0.151890 |

Best top-3 in-hub recovery:

| Data regime | Best method | Top-3 in-hub overlap | In-degree Spearman |
|---|---|---:|---:|
| `knockdowns` | `one_shot_correlation` | 0.466667 | 0.284646 |
| `knockouts` | `stability_correlation` | 0.400000 | 0.219762 |
| `multifactorial` | `genie3_extra_trees` | 0.333333 | 0.199666 |
| `timeseries` | `stability_correlation` | 0.333333 | 0.107265 |

Stability correlation improves edge AUPR in earlier audits, but that gain does not consistently translate into hub or degree recovery here. GENIE3 improves edge AUPR in several regimes, but topology recovery is mixed. Correlation remains strong for out-degree hub recovery despite a clear reciprocal-direction false-positive issue.

## Limitations

This is the first topology-aware audit layer. It uses thresholded top-k graphs and Size10 networks only. Time-series inputs are still not treated as lagged temporal data.
