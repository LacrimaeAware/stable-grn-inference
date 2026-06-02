# DREAM4 Size100 Dynamic Sparse Scaling

This experiment tests whether the strongest Size10 temporal sparse-linear
candidate, `dynamic_lasso_level_include_self_a0_03`, survives when scaled from
10-gene to 100-gene DREAM4 time-series networks. It is the first Size100 work in
the repository, so it doubles as a scaling sanity check for the whole dynamic
sparse branch.

## Data

The audit uses the DREAM4 Size100 time-series files
(`data/raw/dream4/DREAM4_InSilico_Size100/insilico_size100_*/`) and the matching
directed gold standards
(`.../DREAM4_Challenge2_GoldStandards/Size 100/`).

Each network file is `210 x 101` (a `Time` column plus 100 genes). `Time` resets
from `0` to `1000` in steps of `50`, giving 10 trajectories of 21 points each.
Trajectories are split on the `Time` reset, then adjacent lagged samples are
built only within each trajectory, so no lagged pair crosses a trajectory
boundary. `Time` is dropped from the predictors.

Confirmed per network:

| Quantity | Value |
|---|---:|
| Networks | 5 |
| Rows per file | 210 |
| Trajectories per network | 10 |
| Lagged samples per network | 200 |
| Candidate directed non-self edges | 9900 (= 100 x 99) |
| True edges per network | 176, 249, 195, 211, 193 |

True-edge density is roughly 2% of candidates, versus roughly 17% at Size10
(15 of 90). This much sparser regime is the main reason every method's AUPR
falls sharply relative to Size10.

## Methods

A deliberately compact set keeps the 100-gene problem cheap:

1. `dynamic_lasso_level_include_self_a0_03` - the Size10 main candidate.
2. `dynamic_lasso_level_include_self_a0_1` - stronger-regularization check.
3. `dynamic_lasso_level_exclude_self_a0_03` - matched self-predictor control.
4. `lagged_correlation_reference` - cheap baseline.
5. `dynamic_elastic_net_level_include_self_a0_03_l1_0_7` - Elastic Net check.
6. `lagged_genie3_random_forest` and `lagged_genie3_extra_trees` - reduced-tree
   (200 trees) lagged GENIE3-style references.

All sparse-linear methods use the level target (predict `G_j(t+1)`). Include-self
models fit with `G_j(t)` as an extra predictor but never emit a self-edge. Method
names match the Size10 audit so results line up across network sizes.

### Runtime note

The sparse-linear methods are essentially free (~0.07 s per network). Lagged
correlation is ~1 s per network. The two lagged GENIE3 references at 200 trees
are the only non-trivial cost (~10-14 s per network each), which is well within
budget, so they were kept rather than skipped. They can be disabled with
`--skip-trees`, and the tree count is set with `--tree-estimators`.

## Results

Mean across the five Size100 networks (top-N-true-edges cutoff for topology):

| Method | AUROC | AUPR | P@10 | P@100 | self/non-self ratio | reciprocal FP rate |
|---|---:|---:|---:|---:|---:|---:|
| `dynamic_lasso_level_include_self_a0_1` | 0.658451 | **0.161467** | 0.660000 | 0.378000 | 117.05 | 0.950000 |
| `lagged_genie3_random_forest` | **0.754354** | 0.145445 | 0.500000 | 0.308000 | n/a | 0.995122 |
| `lagged_genie3_extra_trees` | 0.748207 | 0.142816 | 0.500000 | 0.298000 | n/a | 0.995652 |
| `dynamic_lasso_level_include_self_a0_03` | 0.678593 | 0.130486 | 0.460000 | 0.328000 | 26.29 | 1.000000 |
| `lagged_correlation_reference` | 0.702563 | 0.129961 | 0.580000 | 0.284000 | n/a | 0.992262 |
| `dynamic_lasso_level_exclude_self_a0_03` | 0.672371 | 0.119005 | 0.500000 | 0.276000 | n/a | 0.992308 |
| `dynamic_elastic_net_level_include_self_a0_03_l1_0_7` | 0.672042 | 0.111418 | 0.400000 | 0.286000 | 16.71 | 1.000000 |

## Findings

- **The Size10 winner does not scale.** `dynamic_lasso_level_include_self_a0_03`
  only ties lagged correlation on mean AUPR (0.1305 vs 0.1300) and trails it on
  mean AUROC (0.6786 vs 0.7026). It is the per-network AUPR winner on 0 of 5
  networks (rank 2-6).
- **Best by AUPR is the higher-alpha sibling.** `dynamic_lasso_level_include_self_a0_1`
  leads mean AUPR (0.1615); the larger network favors stronger regularization.
- **Best by AUROC is GENIE3.** `lagged_genie3_random_forest` leads mean AUROC
  (0.7544) and wins AUROC on all five networks.
- **Include-self still helps a little.** Include-self a0.03 beats exclude-self
  a0.03 on mean AUPR by 0.0115, the same direction as Size10, but the effect is
  small.
- **Self-persistence is even more extreme.** The self/non-self absolute
  coefficient ratio rises to 26.3 at a0.03 (and 117 at a0.1), versus the ~8.9
  reported at Size10. A more extreme persistence ratio paired with weaker edge
  recovery is a warning, not a reassurance.
- **The reciprocal-direction advantage vanishes.** The Size10 candidate had a
  reciprocal false-positive pair rate near 0.20; at Size100 it is 1.00, no better
  than correlation (0.99) or the tree references. This was the most distinctive
  Size10 advantage and it does not survive scaling.
- **Hub recovery is mixed and in-degree stays hard.** Candidate top-5 out-hub
  overlap (0.24) is below correlation (0.32); top-5 in-hub overlap is tied
  (0.04). In-degree Spearman is near zero for every method.

Interpretation: the specific `a0_03` include-self result looks substantially
like a small-network effect. The include-self sparse family is not dead - it
still leads mean AUPR at a higher alpha - but the headline Size10 claim, and
especially its reciprocal-direction advantage, does not reproduce at Size100.

## Outputs

Generated outputs are saved under `results/tables/`:

- `dream4_size100_dynamic_sparse_scaling_summary.csv`
- `dream4_size100_dynamic_sparse_scaling_per_network.csv`
- `dream4_size100_dynamic_sparse_scaling_edges.csv`
- `dream4_size100_dynamic_sparse_scaling_topology.csv`
- `dream4_size100_dynamic_sparse_scaling_debug_report.md`

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\10_dream4_size100_dynamic_sparse_scaling\run_size100_dynamic_sparse_scaling.py
```

Optional flags: `--skip-trees` to drop the GENIE3 references, `--tree-estimators N`
to change the reduced tree count (default 200), `--random-seed`, `--n-jobs`.

## Interpretation Policy

This is a scaling audit, not a final result. A Size10 candidate should only be
promoted to a main method if its advantage survives at Size100 (or in controlled
simulation). Here it does not, so the honest conclusion is to record the negative
scaling result, prefer stronger regularization at scale, and add a literature-
faithful dynamic baseline (dynGENIE3) plus GeneNetWeaver sweeps before making
broader claims.
