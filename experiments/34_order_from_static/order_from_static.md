# Experiment 34: recover an order from static data, and ask whether it helps

The user's idea, tested honestly: from static, unordered data, rebuild a 1D order from the geometry
of a similarity matrix and its higher powers (spectral seriation / diffusion pseudotime); the order
is recovered up to reversal and a root prior orients it; and indirect multi-step (A to C to D)
structure might be captured by iterating the correlation matrix. BoolODE is the testbed because it
has both a true pseudotime and a true network, so order-recovery accuracy can be measured against
truth (which the trajectory-inference papers do not report) and the downstream value can be tested.

## Method (three parts, run together)

- Part A, order recovery: recover a cell order from static geometry (Fiedler vector / spectral
  seriation, and the first diffusion component), score it by absolute Spearman against the true
  pseudotime (absolute, since the order is reversal-ambiguous). Baselines: ordering by the top
  principal component, and a random order.
- Part B, does the order help the network: lagged edge scoring along the recovered order (oriented
  by the true earliest cell as the root prior) vs static correlation (no order) vs the true-order
  oracle; directed AUPR against the true network.
- Part C, indirect / higher-order correlation: direct correlation vs its square, second-order
  correlation (correlation of correlation profiles), and network propagation; skeleton AUPR.

Tooling: `src/stable_grn_inference/analysis/ordering.py`, tested in `tests/test_ordering.py`.
Data: BoolODE 6 topologies at 200 cells, 3 replicates each (18 datasets).

## Result

Part A, order recovery (absolute Spearman vs true pseudotime): spectral 0.825, diffusion 0.825, PC1
0.820, random 0.061. By topology: linear 0.96, long-linear 0.96, bifurcating 0.86, trifurcating
0.83, bifurcating-converging 0.78, cycle 0.55.

Part B, network recovery (directed AUPR): recovered-order correlation 0.346, recovered-order random
forest 0.342, static correlation 0.358, true-order oracle 0.383.

Part C, skeleton recovery (AUPR): direct correlation 0.648, second-order 0.589, network propagation
0.566, correlation-squared 0.520, chance 0.342.

## What it says

- The order IS recoverable from static data. At absolute Spearman 0.825, the geometry reconstructs
  the true 1D order, very well on orderable trajectories (linear 0.96) and poorly on cycles (0.55,
  expected, since a loop has no single 1D order). This is the core intuition confirmed, with the
  accuracy number the trajectory-inference literature does not report.
- But the nonlinearity did not matter here: plain PC1 (0.820) ties the spectral and diffusion
  orders (0.825). For these trajectories the order is already a linear (PC1) direction, so the
  diffusion machinery adds nothing over the linear baseline.
- And recovering the order does not help the network. The recovered-order edges (0.346) do not beat
  static correlation (0.358), and even the true-order oracle (0.383) barely beats static. The
  reason is informative: knowing the order adds little network information beyond what static
  co-expression already captures here, so this is not a failure to recover the order (that
  succeeds), it is that the order is not the bottleneck for this network.
- The higher-order / indirect correlation hurts: direct correlation (0.648) beats its square
  (0.520), second-order (0.589), and propagation (0.566). Iterating the correlation adds spurious
  transitive edges rather than capturing useful indirect structure, the failure mode the literature
  flagged as the open risk.

Net: the idea works for what it directly does (recover an order from static, accurately), which is a
real confirmation of the intuition. It does not translate into a network-inference gain, and the
higher-order twist adds spurious edges. Direct correlation remains the strongest network method,
consistent with the rest of the project.

## Outputs

Under `results/tables/` (git-ignored): `order_from_static_all.csv`, `order_from_static_summary.csv`,
`order_from_static_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/34_order_from_static/run_order_from_static.py
```
