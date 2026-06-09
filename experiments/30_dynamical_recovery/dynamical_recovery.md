# Experiment 30: directed-structure recovery with a time axis

> Correction (methodology audit). The comparison here is against a SYMMETRIC static baseline that
> cannot orient an edge by construction, so beating it at a directed task is guaranteed and
> uninformative about method quality. On DREAM4 the operator's 0.37 directed AUPR ranks below the
> established lagged methods already run on the identical networks in exp 7 (lagged GENIE3 RF 0.53,
> GENIE3 ET 0.53, LASSO 0.51/0.49, correlation 0.46), and below the project's own earlier DREAM4
> results (0.65 with the self-edge, exp 8/9). This is a controlled demonstration that time order
> enables orientation, not a benchmarked positive. The benchmarked head-to-head is exp 33.

Direction B from `docs/next_direction.md`. The frontier where a positive result can exist:
recover directed structure that a symmetric (second-order) statistic cannot, by using a time axis,
graded against ground truth.

## Question

Experiment 28 showed specific structure is not recoverable from a static snapshot in RPE1's
regime (high dominant-mode fraction, low specific-SNR), and the project's regime ladder found
time-series orient well (~0.9) while static data does not (~0.5). This experiment turns that
into a controlled, ground-truthed test: does a dynamical operator estimated from consecutive
states recover the directed operator that a static, symmetric statistic cannot, and does it
stay above the floor as the dominant shared mode grows?

## Method

Part A (synthetic, ground truth). A linear stochastic system `x_{t+1} = A x_t + eps` with
`A = decay * I + W`, where `W` is a sparse acyclic directed operator (the truth; because `W`
is nilpotent the system is stable for any coupling). The driving noise carries a tunable
dominant shared mode `mode_strength * z_t * p` that inflates one direction of the stationary
covariance without changing the operator. Compare the dynamic-mode operator
`A_hat = X2 X1^+` (uses time order) against the static correlation of the states (ignores it),
graded by directed and skeleton AUPR, swept over dominant-mode strength and noise. The
least-squares operator estimate is unbiased by the input covariance, so the prediction is that
the dynamic operator recovers direction even under a strong dominant mode while the static
symmetric score recovers at most the skeleton.

Part B (DREAM4 Size10 time-series, realistic dynamics, known network, no download). Fit the
dynamic operator on the local DREAM4 time-series (per network, snapshot pairs built within
trajectories) and grade against the gold-standard network; compare to the static correlation
baseline and the chance line, averaged over the five networks.

Tooling: `src/stable_grn_inference/dynamics/temporal.py`
(`make_dynamical_system`, `dmd_operator`, `dmd_edges`, `static_correlation_edges`,
`edges_to_operator`, `dynamical_recovery_grid`), tested in `tests/test_temporal.py`. Grading
reuses `specific_recovery_aupr` / `normalized_recovery` from `separability` (exp 28), so the
two experiments share one recovery metric and the result is comparable to the static phase
diagram.

## Why this is the live direction

Unlike RPE1 (Direction A, exp 29), this regime has the one ingredient that makes direction
identifiable: a time axis. It grades against truth, so a positive is a real positive, and it
defines exactly what a real dataset must provide (a time axis, specific-SNR above the exp 28
floor). The companion dataset scout looks for a real time-resolved dataset with checkable
truth to place on the same diagram.

## Outputs

Under `results/tables/` (git-ignored): `dynamical_recovery_synthetic_grid.csv`,
`dynamical_recovery_dream4.csv`, `dynamical_recovery_summary.csv`,
`dynamical_recovery_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/30_dynamical_recovery/run_dynamical_recovery.py
.\.venv\Scripts\python.exe -B experiments/30_dynamical_recovery/run_dynamical_recovery.py --quick
```
