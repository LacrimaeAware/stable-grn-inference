# DREAM4 Size10 Dynamic Model Batch Audit

This experiment compares a broad set of first-pass dynamic time-series edge scorers on DREAM4 Size10. The goal is not to force one clean story yet; it is to see which evidence types look promising for directed hidden-structure recovery.

## Data

The script uses Size10 time-series files. Trajectories are split when `Time` resets, and all lagged samples are built within trajectory boundaries only:

```text
X_t = gene expression at time t
Y_next = gene expression at time t+1
delta_Y = Y_next - X_t
derivative_Y = delta_Y / delta_time
```

## Compared Evidence Types

- temporal modeling with level, delta, and derivative targets
- tree-based conditional prediction with random forests and Extra Trees
- sparse linear prediction with a LASSO alpha grid plus a small Elastic Net check
- a small target-wise MLP sanity baseline with permutation importance
- trajectory-bootstrap stability for one tree and one sparse candidate
- equal-weight rank fusion by normalized score, reciprocal rank, and Borda score
- light signal preprocessing with moving-average smoothing and wavelet denoising (PyWavelets is now a declared dependency in `requirements.txt`, so this variant runs)
- topology-aware hidden-structure metrics

## Self-Predictor Modes

- `exclude_self_predictor`: when predicting target gene `G_j`, do not include `G_j(t)` as a predictor.
- `include_self_predictor_no_self_edge`: include `G_j(t)` during fitting because persistence may matter, but do not output `G_j -> G_j` as a candidate regulatory edge.

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_size10_dynamic_model_batch_summary.csv`
- `dream4_size10_dynamic_model_batch_edges.csv`
- `dream4_size10_dynamic_model_batch_topology.csv`
- `dream4_size10_dynamic_model_batch_debug_report.md`

## Current Results

Default run settings used 200 trees for one-shot tree models, 50 trees for trajectory-bootstrap stability tree models, and 30 trajectory-bootstrap resamples.

Best mean edge metrics across five Size10 networks:

| Metric | Winning method | Family | Value |
|---|---|---|---:|
| AUPR | `dynamic_lasso_a0_03_level_include_self_raw` | sparse linear | 0.652712 |
| AUROC | `dynamic_lasso_a0_03_level_include_self_raw` | sparse linear | 0.821067 |
| precision@5 | `dynamic_lasso_a0_03_derivative_include_self_raw` | sparse linear | 0.920000 |
| precision@10 | `dynamic_lasso_a0_1_level_include_self_raw` | sparse linear | 0.720000 |
| precision@20 | `dynamic_elastic_net_a0_03_l1_0_7_level_include_self_raw` | sparse linear | 0.490000 |

Best topology/hub metrics:

| Metric | Winning method | Value |
|---|---|---:|
| top-3 out-hub overlap | `dynamic_elastic_net_a0_03_l1_0_7_level_include_self_raw` | 0.533333 |
| top-3 in-hub overlap | `dynamic_lasso_a0_3_delta_exclude_self_raw` | 0.733333 |

Main takeaways:

- Sparse linear dynamic models with `include_self_predictor_no_self_edge` are the strongest result in this batch.
- Level targets beat delta and derivative targets for mean AUPR/AUROC, though delta/derivative variants can be strong for top-k precision and in-hub recovery.
- Including the self predictor during fitting helps substantially for non-self edge recovery in this Size10 audit. This may reflect useful persistence modeling, but it should be checked carefully before over-interpreting.
- The MLP permutation-importance baseline underperforms the best sparse and tree baselines.
- The fixed stability variants do not improve over their corresponding one-shot dynamic baselines in this run.
- Equal-weight rank fusion does not beat the best single model.
- Moving-average smoothing and wavelet denoising both slightly hurt the RF level/exclude-self baseline by mean AUPR (raw 0.537, moving-average-3 0.487, wavelet-denoise 0.519). Wavelet denoising (db1, soft threshold) hurts less than moving-average smoothing, but neither beats the raw baseline on this Size10 setting.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\08_dream4_size10_dynamic_model_batch\run_dynamic_model_batch_audit.py
```

## Frontier Ideas Not Implemented In This Batch

- Full reinforcement learning or GFlowNet causal graph search is a possible future branch, but it needs a graph-search formulation with states, actions, rewards, and a scoring function. It is not a drop-in edge scorer.
- Larger neural models are more appropriate after moving to many simulated GeneNetWeaver networks, larger single-cell perturbation data, or strong biological priors.
- Full wavelet scattering or structured transformation discovery belongs in a separate Track B pilot, not this core DREAM4 batch.
- Size100 scaling should come after identifying which Size10 dynamic or ensemble candidates are worth scaling.

## Limitations

This is a broad benchmark batch, not a final method claim. Equal-weight fusion does not tune weights on the gold standard. The MLP baseline uses simple permutation importance and should be treated as a sanity check, not a serious neural architecture result.
