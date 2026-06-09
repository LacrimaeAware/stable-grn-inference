# DREAM4 Stability + Orientation Diagnostics

A response to a critical research review: the project drifted from its original
Track A thesis (stability-aware sparse selection makes directed inference more
reliable) into method-chasing. This experiment **adds no new model families**. It
decomposes where the error comes from, tests deployable/theory-driven penalties,
attributes the fusion gain, and adjudicates the stability thesis with proper
error control — every comparison reported with a paired-over-networks bootstrap
95% CI, because n=5 networks per size makes most point differences underpowered.

The dynamic lagged include/exclude-self LASSO is a regularized **sparse VAR(1) /
Granger** model; Granger-style recovery conflates causation with latent
confounding (an identifiability caveat).

## Headline findings

1. **The error is skeleton detection, not orientation.** Once a true pair is
   detected, methods orient it correctly ~0.81–0.96 of the time (lagged random
   forest at the low end, sparse-CV at the high end), versus exactly
   **0.50** for the symmetric `static_correlation` control (a built-in validity
   check). The undirected-vs-directed AUPR gap is small (Size10 ~0.07–0.17, Size100
   ~0.03–0.05, all paired-CI-positive but small), while even the *undirected*
   (skeleton) AUPR is low at Size100 (~0.16–0.22). So the dominant failure is not
   finding the right gene pairs — not getting the direction wrong. The recurring
   reciprocal false positives are therefore mostly **false pairs**, not mis-oriented
   true edges, which means the reciprocal-penalty machinery we kept adding was
   aimed at the wrong lever.
2. **Penalty strength is predictable from theory, not just grids.** A pivotal
   square-root/scaled LASSO penalty (λ ∝ √(2 log p / n), σ-free) **matches or beats
   the grid oracle at Size100** (AUPR 0.168 vs oracle 0.161; paired delta +0.006,
   CI [0.002, 0.012] — excludes 0), and the theory α value (~0.11) tracks the
   Size100 oracle (0.10). It overshoots at Size10 (worse than oracle, CI excludes 0).
   BIC ≈ oracle at Size100; CV is fine at Size10 but too aggressive (α=0.03) at
   Size100 (worse than oracle, CI excludes 0). So the Size10→Size100 shift is a
   **sample-complexity** effect (n vs s·log p), and a gold-free penalty can be set
   from it rather than tuned.
3. **Fusion's Size100 gain is genuine complementarity, not variance reduction.**
   3-arm control: cross-method fusion − within-method-bootstrap fusion = **+0.068
   AUPR (CI [0.049, 0.084], excludes 0)**; the within-method bootstrap arm is
   actually slightly *worse* than the single best method (−0.017, CI excludes 0).
   So averaging the same method does not help — fusing *different* methods does.
   At Size10 the single sparse model is best and fusion shows no significant gain.
4. **Formal stability selection does NOT revive the Track A thesis (on this data).**
   With trajectory-level subsampling (respecting within-trajectory dependence) and
   the Meinshausen–Bühlmann bound E[V] ≤ q²/((2π−1)p) per target: the bound holds
   but is **too loose to be informative** (e.g., Size100 π=0.9: bound ~1134 expected
   false positives vs ~79 actual; often the bound exceeds the number of edges
   selected), and the selection-probability ranking **underperforms** a single
   CV/theory-tuned fit (AUPR 0.36 Size10 / 0.11 Size100 vs sparse 0.64 / 0.13).

## Results by part (means; see paired-tests table for CIs)

### Part 1 — directed vs undirected + orientation-given-skeleton

| size | method | directed AUPR | undirected AUPR (max) | orientation gap | orientation-given-skeleton |
|---|---|---:|---:|---:|---:|
| 10 | sparse_cv | 0.640 | 0.710 | 0.070 | 0.960 |
| 10 | fusion_borda | 0.592 | 0.692 | 0.099 | 0.917 |
| 10 | lagged_correlation | 0.458 | 0.625 | 0.167 | 0.902 |
| 10 | static_correlation (symmetric control) | 0.299 | 0.574 | 0.275 | **0.500** |
| 100 | fusion_borda | 0.182 | 0.222 | 0.040 | 0.938 |
| 100 | genie3_rf_level | 0.142 | 0.188 | 0.047 | 0.885 |
| 100 | sparse_cv | 0.131 | 0.162 | 0.031 | 0.930 |
| 100 | static_correlation (symmetric control) | 0.074 | 0.144 | 0.069 | **0.500** |

### Part 2 — alpha selectors (focal lasso level include-self)

| size | selector | chosen α | AUPR | paired Δ vs oracle (CI) |
|---|---|---:|---:|---|
| 100 | oracle (not deployable) | 0.10 | 0.1615 | — |
| 100 | theory_sqrt_lasso | ~0.13 | **0.1677** | +0.006 [0.002, 0.012] |
| 100 | bic / theory_sigma_hat | 0.10 | 0.1615 | 0.000 [0, 0] |
| 100 | cv | 0.03 | 0.1305 | −0.031 [−0.048, −0.014] |
| 10 | oracle | 0.044 | 0.6557 | — |
| 10 | cv | 0.040 | 0.6398 | −0.016 [−0.040, 0.000] (tie) |
| 10 | theory_sqrt_lasso | 0.134 | 0.6059 | −0.050 [−0.095, −0.018] |

### Part 3 — fusion 3-arm (paired CIs)

| size | comparison | mean Δ AUPR | 95% CI | read |
|---|---|---:|---|---|
| 100 | cross-method − within-bootstrap | +0.068 | [0.049, 0.084] | complementarity (significant) |
| 100 | within-bootstrap − single | −0.017 | [−0.024, −0.011] | ensembling-same hurts |
| 100 | cross-method − single | +0.051 | [0.035, 0.067] | fusion beats single |
| 10 | cross-method − single | −0.048 | [−0.090, 0.008] | tie (single best numerically) |

### Part 4 — stability selection (MB bound vs actual)

At every threshold the MB bound exceeds the actual false-positive count by a large
factor (and frequently exceeds the number selected), i.e. it is conservative but
**uninformative** at p≫n with n≈200. Precision is low (Size100 0.08–0.25). Using
selection frequency as an edge score underperforms a single tuned sparse fit.

## Verdict on the Track A thesis

On DREAM4, the original claim — that stability information makes sparse directed
inference *more reliable* — is **not supported in its strong form**. Stability
selection's finite-sample bound is too loose to be useful here, and its ranking is
worse than a single CV/theory-calibrated fit. The thesis should be retired or sharply
re-scoped: stability frequencies remain a *legitimate confidence object*, but they do
not buy reliability or accuracy over a well-penalized single model at these sample
sizes. The more useful, defensible reframing the diagnostics support is:
**directed GRN inference here is skeleton-limited and sample-complexity-limited, the
penalty is theory-predictable, and fusing complementary evidence helps in the hard
(Size100) regime.**

## General lessons (cautious, tied to results)

- Decompose before optimizing: an undirected-vs-directed split + a symmetric control
  immediately reframed the problem from "fix orientation" to "improve skeleton recovery."
- A regularization penalty that scales like √(log p / n) can replace grid/oracle tuning
  in the sample-starved regime; estimating σ is avoidable via square-root LASSO.
- Ensembling only helps with complementary errors; always include a within-method
  bootstrap control to separate complementarity from variance reduction.
- A theorem's bound (MB) can be valid yet useless; report the bound against the *actual*
  count and call it uninformative when it is.
- At n=5, report paired CIs and accept that many comparisons are ties.

## Outputs

`results/tables/`: `dream4_stability_orientation_directed_vs_undirected.csv`,
`_alpha_theory.csv`, `_fusion_control.csv`, `_stability_selection.csv`,
`_paired_tests.csv`, `_summary.csv`, `_debug_report.md`. Figures under
`results/figures/`: directed-vs-undirected AUPR; stability threshold vs MB bound vs
actual false positives.

## Run

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\17_dream4_stability_orientation_diagnostics\run_stability_orientation_diagnostics.py --quick
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\17_dream4_stability_orientation_diagnostics\run_stability_orientation_diagnostics.py --standard
```

Flags: `--skip-size100`, `--n-jobs`, `--n-subsamples`, `--random-seed`.

## Next on BEELINE

Re-ask the same three sharpened questions on real proxy networks: is the error
skeleton or orientation on ChIP/curated references; do stability-selection
probabilities stay calibrated; does cross-method complementarity survive noisy
biological references. The skeleton-vs-orientation decomposition and the symmetric
control transfer directly to single-cell.
