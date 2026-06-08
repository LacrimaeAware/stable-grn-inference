# Experiment 18 — BEELINE Curated diagnostics

## Goal

Transfer the experiment-17 diagnostic framing (skeleton-vs-orientation decomposition,
alpha-selector behavior, 3-arm fusion, formal stability selection, paired CIs) to a
**real single-cell GRN benchmark**, and test which DREAM4 conclusions survive a regime
change. The experiment reuses experiment 17's *exact* diagnostic functions (imported via
`importlib`) so the DREAM4↔BEELINE comparison is apples-to-apples by construction.

## Design choices (and why they diverge from the OpenAI prompt)

- **Curated, not scRNA-seq.** The OpenAI prompt pointed at the scRNA-seq sets. I redirected
  to BEELINE **Curated** (GSD, HSC, VSC, mCAD) because those ship **exact, directed**
  ground-truth networks, with 10 replicates each, in an **n cells ≫ p genes** regime
  (2000 × 5–19). That makes them *scorable against truth* and the cleanest possible
  **regime contrast** to DREAM4 (which is lagged time-series, p ≫ n). The 7 scRNA-seq
  sets in this download (hESC, hHep, mDC, mESC, mHSC-E/GM/L) have **no reference network**,
  so they are reported as unscorable, with the exact path to drop a reference in. Scoring
  them later would measure *reference-agreement*, not truth recovery — a different claim.
- **Static methods only.** Single-cell snapshots have no within-sample time axis, so there
  is no lagged source(t)→target(t+1) construction. Methods: static correlation (symmetric
  control), GENIE3 random forest, exclude-self CV-LASSO, and Borda fusion.
- **Cell-level subsampling** for stability (cells are ~exchangeable), unlike DREAM4's
  trajectory-respecting subsampling.
- **Paired CIs over replicates** within each model (n=5), via the experiment-17 bootstrap.

## Headline (corrected against the quick-run overstatement)

A `--quick` run on GSD alone suggested orientation "collapses to chance." **The full
4-model run does not support that as a universal statement.** The honest finding:

> On static single-cell data, orientation signal is **weaker and far more
> network-dependent** than on DREAM4's lagged time-series — not a clean universal collapse.

- Orientation-accuracy-given-skeleton, mean across models: `sparse_cv` **0.636**,
  `fusion_borda` **0.628**, `genie3_rf` **0.579**, `static_correlation` **0.500** (the
  symmetric control sits *exactly* at chance everywhere, as it must — a good sanity check).
- But the spread is large and structural: **GSD collapses** (0.32–0.47, at/below chance),
  while **VSC retains strong orientation** (sparse_cv = **1.00**, fusion = 0.90) and HSC is
  intermediate (0.62–0.74). DREAM4, by contrast, was a tight **0.88–0.96 across the board**.
  (Denominators are small: VSC and mCAD have only 5 orientable non-reciprocal edges each, HSC
  14, GSD 40, so the per-network orientation numbers — especially VSC = 1.00 — are low-power.)
- The GSD collapse has a structural cause: GSD's truth is **heavily reciprocal**
  (18 reciprocal pairs, i.e. 36 of 76 true edges, are bidirectional), so "correct orientation" is partly
  ill-posed there. Where the truth is more acyclic (VSC), orientation is recoverable even
  statically.

So the DREAM4 conclusion "**error is skeleton-detection, orientation is essentially free**"
is **regime- and network-specific**. It held on DREAM4 because lagged time-series hands you
direction via temporal precedence. On static observational single-cell data, orientation is
a real and variable problem — which is the concrete, empirical version of the identifiability
caveat, and the cleanest motivation yet for moving toward **temporal/interventional** data.

## What transfers from DREAM4, and what doesn't

| DREAM4 conclusion | BEELINE Curated verdict |
| --- | --- |
| Error is skeleton-bound; orientation ≈ free | **Partly breaks.** Orientation weaker & network-dependent (mean ~0.6, GSD ≤0.5, VSC up to 1.0). |
| LASSO penalty is sample-complexity-predictable | **Holds, intensifies.** n≫p ⇒ tiny optimal α; CV/BIC ≈ oracle (within 0.002–0.06); sqrt-LASSO sensible (beats oracle on HSC, +0.015, CI excludes 0). `density_prior` overshoots badly (HSC −0.16). |
| Fusion gain is genuine cross-method complementarity | **Regime-dependent.** Only helps the low-signal dense case GSD (cross−bootstrap +0.019, CI [0.016,0.022]); neutral on VSC; **hurts** HSC (−0.015) and mCAD (−0.017). |
| Strong stability-selection thesis not supported | **Confirmed, transfers cleanly.** MB bound still far too loose; selection-probability precision ≈ edge density (0.24–0.27 on GSD/HSC/VSC; mCAD 0.65 but density is also 0.65); ECE ~0.60. Ranking by selection frequency does not separate signal. |

## Per-model snapshot (directed AUPR / EPR / orientation-given-skeleton, best-ish method)

| model | genes | true_density | AUPR (sparse_cv) | EPR | orient.-given-skel | note |
| --- | --- | --- | --- | --- | --- | --- |
| GSD | 19 | 0.22 | 0.25 | 0.96 | 0.37 | reciprocal-heavy; orientation collapses; fusion helps here |
| HSC | 11 | 0.24 | 0.44 | 1.53 | 0.68 | sqrt-LASSO beats oracle; fusion hurts |
| VSC | 8 | 0.27 | 0.62 | 2.19 | **1.00** | most acyclic; orientation fully recoverable statically |
| mCAD | 5 | 0.65 | 0.59 | 1.04 | 0.50 | density so high EPR≈1 by construction; AUROC<0.5 (tiny graph) |

## Honest caveats

- **Tiny graphs.** 5–19 genes. AUROC on mCAD is < 0.5 (5 genes, 65% density — almost fully
  connected, so ranking is nearly meaningless). Treat mCAD as a near-degenerate edge case.
- **EPR ceilings.** At 22–65% density, EPR's headroom above 1.0 is small; modest EPR here is
  not directly comparable to a sparse benchmark's.
- **Curated ≠ scRNA-seq.** Curated truth is exact, so nothing here is contaminated by proxy
  references. The proxy-reference problem is real but lives in the *unscorable* scRNA-seq sets;
  this experiment deliberately does not pretend to score those.
- **The quick-run lesson.** The GSD-only `--quick` headline was wrong as a general claim. Logged
  as a reminder: do not generalize an orientation/fusion verdict from one network — these are
  the two most network-dependent diagnostics in the whole battery.

## Reproduce

```
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/18_beeline_diagnostics/run_beeline_diagnostics.py --quick   # GSD, 3 reps
.\.venv\Scripts\python.exe -B experiments/18_beeline_diagnostics/run_beeline_diagnostics.py           # 4 models x 5 reps
```

Artifacts (git-ignored `results/`): `beeline_diagnostics_{summary,alpha,fusion,stability,pairwise,edges}.csv`,
`beeline_diagnostics_debug_report.md`, `results/figures/beeline_diagnostics_directed_vs_undirected.png`.

## Verdict

The diagnostic battery is **portable** and the stability-selection negative result is
**robust across regimes**. The two headline DREAM4 positives — "skeleton-bound" and
"fusion complementarity" — are **regime-dependent** and do not transfer cleanly to static
single-cell. The reframed working thesis from experiment 17 (sample-complexity-limited
inference with a theory-predictable penalty) survives; the orientation and fusion claims
need to be stated per-regime. Next natural step: a benchmark where orientation is
*identifiable by design* (interventional / perturbation data), which is exactly where the
weak, network-dependent static orientation signal predicts the largest gains.
