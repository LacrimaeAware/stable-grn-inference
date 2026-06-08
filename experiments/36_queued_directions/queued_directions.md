# Experiment 36: queued directions (diversity-consensus, cycle 2D geometry)

The remaining candidate directions, run honestly. BoolODE, 18 datasets, exact truth.

## Part A: diversity-consensus

Combine three genuinely different lenses (Pearson = linear, Spearman = monotone-nonlinear, mutual
information = arbitrary dependence) by rank-agreement; does the consensus beat the best single lens at
recovering the skeleton?

| pearson | spearman | mutual_info | consensus |
| --- | --- | --- | --- |
| 0.651 | 0.640 | 0.670 | 0.644 |

Negative. The consensus (0.644) does not beat the best single lens (mutual information, 0.670). The
lenses largely agree, so combining them re-finds the same skeleton rather than a cleaner core. One
small note: the nonlinear lens (mutual information, 0.670) marginally beats linear correlation (0.651),
the only hint that a nonlinear dependence measure adds anything, and it is small.

## Part B: cycle 2D geometry

A 1D order cannot describe a loop. Recover a 2D diffusion embedding, read the cyclic order as an
angle, and compare to the 1D spectral order on the cyclic topologies (circular correlation vs true
order).

| topology | 1D spectral order | 2D circular order |
| --- | --- | --- |
| dyn-CY (the actual cycle) | 0.549 | 0.798 |
| dyn-BF (branch) | 0.864 | 0.258 |
| dyn-BFC (branch) | 0.779 | 0.140 |
| dyn-TF (branch) | 0.834 | 0.493 |

The aggregate ("2D 0.42 vs 1D 0.76, does not help") is misleading because it averages a real cycle with
three branching trees. The honest, per-topology result is a small WIN for the intuition: on the actual
cycle (dyn-CY), the 2D circular embedding recovers the order at 0.80 versus 0.55 for the 1D order. A
cycle genuinely needs 2D, and in 2D its order is recoverable. On the branching topologies, forcing a
circular embedding hurts, as it should, because trees are not loops. So the right reading is
topology-matched: use 2D where the trajectory is a cycle, 1D where it is a line.

## Verdict

- Diversity-consensus does not beat the best single lens; combining diverse lenses mostly re-finds the
  same skeleton. Mutual information is marginally the best single lens.
- Cycle 2D geometry validates the cycle-needs-richer-geometry intuition for actual cycles (dyn-CY 0.80
  vs 0.55), while correctly failing on non-cyclic trees. The geometry must match the topology.
- Neither breaks the standing wall (the skeleton is easy; direct-vs-indirect and beating simple
  baselines on edges remain unsolved), but the cycle result is a genuine, if narrow, confirmation.

## Outputs

Under `results/tables/` (git-ignored): `queued_directions_consensus.csv`,
`queued_directions_cycle2d.csv`, `queued_directions_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/36_queued_directions/run_queued_directions.py
```
