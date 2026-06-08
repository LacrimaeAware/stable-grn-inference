# Experiment 35: non-Gaussian orientation from static data + detectability

Two candidate directions together (research_directions.md #1 and #2): get edge DIRECTION from static
data via non-Gaussianity (the LiNGAM idea, the principled realization of "the arrow is in the higher
moments"), and map per-edge detectability against a null (the SNR floor at edge resolution).

## Method

Tooling: `src/stable_grn_inference/analysis/nongaussian.py` (`pairwise_orientation`,
`nongaussian_directed_edges`, `edge_detectability`, `nongaussianity`), tested in
`tests/test_nongaussian.py`. The orientation is a tanh likelihood-ratio (Hyvarinen-Smith): for a pair,
the higher moments say which gene is the more plausible cause; the correlation skeleton mass is placed
on that direction. Detectability is the z-score of |correlation| against a per-variable permutation
null. Data: BoolODE, 6 topologies at 200 cells, exact network truth. Baselines: symmetric correlation
(cannot orient by construction) and GENIE3.

## Result

The method is correct in principle: on a planted linear non-Gaussian chain (Laplace noise, 4000
samples) it orients the chain from static data with no time axis (the unit test passes). The
data is non-Gaussian (mean absolute excess kurtosis 2.6), so the assumption is met.

But on real BoolODE it does not work (200-cell, directed AUPR vs truth):

| symmetric correlation | non-Gaussian directed | GENIE3 |
| --- | --- | --- |
| 0.358 | 0.292 | 0.413 |

By regime (LiNGAM assumes acyclicity, so it should do best on acyclic):

| regime | symmetric correlation | non-Gaussian directed | GENIE3 |
| --- | --- | --- | --- |
| acyclic | 0.397 | 0.225 | 0.428 |
| cyclic | 0.338 | 0.326 | 0.406 |

Detectability (per-edge z vs the permutation null):

| z, true edges | z, false edges | true-edge rate at z>2 |
| --- | --- | --- |
| 10.7 | 6.2 | 0.88 |

## What it says

- Non-Gaussian orientation does NOT beat plain correlation on real data, and is actually worse on the
  acyclic networks where it should excel (0.225 vs 0.397). It is below GENIE3 everywhere. This is not
  a sample-size problem: re-running at 5000 cells (25x more data) gives the same picture (non-Gaussian
  0.318 vs symmetric 0.363; acyclic 0.239 vs 0.416). So the higher-moments idea, which is provably
  correct and which we confirmed on a clean planted Laplace chain, fails on this data because the
  noise is not the LiNGAM kind the orientation measure assumes: the orientation is systematically
  anti-aligned with truth on the linear chains, so placing the correlation mass on the chosen
  direction loses to leaving it symmetric. This matches the literature finding that no advanced method
  has been shown to beat simple baselines on real GRNs.
- The detectability map works as a diagnostic: true edges sit far from the null (z=10.7) and 88%
  clear z>2, so the skeleton is clearly detectable. But false edges also sit well above the null
  (z=6.2), because transitive co-expression makes many non-edges look real. So detectability
  separates signal from pure noise but not direct edges from indirect ones, consistent with the rest
  of the project.

Net: your higher-moments intuition is correct in theory (validated on planted truth), but on real
data it does not beat correlation. The detectability tool quantifies the SNR floor per edge and shows
the skeleton is easy while direct-vs-indirect remains the wall.

## Outputs

Under `results/tables/` (git-ignored): `nongaussian_orientation_all.csv`,
`nongaussian_orientation_summary.csv`, `nongaussian_orientation_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/35_nongaussian_orientation/run_nongaussian_orientation.py
.\.venv\Scripts\python.exe -B experiments/35_nongaussian_orientation/run_nongaussian_orientation.py --cells 5000
```
