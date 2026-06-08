# Verification audit (2026-06-08)

**Verified:** 2026-06-08, ~14:40 EDT. **Scope:** experiments 01–28 and the shared library
(`src/stable_grn_inference/`), as committed in `024d85a`. This is a point-in-time record so a
future reader knows when the code and write-ups were last independently checked, without
having to re-derive it. Experiments 29–30 landed (committed in the same `024d85a`) after this
pass and are **not** covered here — see Scope below.

An independent end-to-end review of the repository. Every experiment was checked on two
axes: program correctness (does the code compute what it claims) and design validity (does
the experiment actually test its stated question, and is the design sound). Write-up numbers
were cross-checked against the saved results.

## Method

- **Library** (`src/stable_grn_inference/`): every module read in full. Correctness confirmed
  by inspection and by the test suite (158 tests, all pass).
- **DREAM4 (exp 01–14)** and **BEELINE (exp 15–18)**: raw data present locally, so
  representative scripts were re-run to confirm the headline numbers reproduce.
- **RPE1 / CausalBench (exp 19–28)**: the raw ~8.7 GB Perturb-seq `h5ad` is git-ignored and
  not present (removed after the runs). These were verified by code review plus cross-checking
  every quantitative claim against the saved `results/tables/*.csv` and `*_debug_report.md`
  (generated when the data was present); the synthetic positive controls and unit tests, which
  do not need the real data, were re-run.
- Three load-bearing integrity checks were verified directly: (a) the include-self dynamic
  models never emit a self-edge and exclude it from the candidate set, so the include-self
  gain is not gold/self leakage; (b) the split-half orientation reproducibility runs on
  provably disjoint cell halves; (c) the gold-free calibrated-confidence pipeline never reads
  gold labels outside final evaluation.

## Scope

This audit covers experiments 01–28 and the library they share. Experiments 29
(whitened asymmetry) and 30 (dynamical recovery) — the Direction A / B work from
`next_direction.md` — were created by parallel development while this audit ran; they were
not part of this review. Their tests pass at the time of writing, but they are in progress
and unaudited here.

> **TODO — verify on the next pass.** Experiments 29–30 were under active development on
> 2026-06-08 and have **not** been verified. They (and the resulting counts — the repo now
> has 30 experiments / 179 tests, while the README still reads 28 / 158) should be audited and
> reconciled once their development is finished. Until then, treat 29–30 as unverified.

## Conclusion

The shared library is correct. All 28 audited experiments are scientifically sound and
honestly reported. No gold-label leakage into the inference or model-selection path, no metric misuse,
no fabricated numbers, no fake controls, and no circular reasoning were found; every checked
figure traces to a saved table, and several headline numbers were recomputed from raw
per-edge CSVs and matched exactly. The design is consistently deliberate: genuine controls (the
self-permutation and self-residualization controls in exp 13, the symmetric-correlation 0.50
orientation baseline in exp 17/18), honest negatives (exp 10 scaling, exp 22/23/24/25/27), and
explicit self-skepticism checkpoints — exp 23 flags its own 0.999 number as a
zeros-agree-with-zeros artifact, exp 25 overturns a single-lucky-seed positive with a
multi-seed check, and exp 18 records that a `--quick` GSD-only pass over-claimed a universal
orientation collapse that the full run corrected.

## Corrections applied during the audit

- **exp 23** (`response_inverse.md`): the Part-0 synthetic table's `raw |D| AUPR` column read
  0.86 / 0.85 / 0.84 / 0.70 / 0.66 but the saved CSV (`..._synthetic_summary.csv`, column
  `aupr_raw_D`) and the auto-generated debug report both have 0.71 / 0.71 / 0.70 / 0.70 / 0.66.
  Corrected the column (and aligned the inverse column to the saved `ridge` rows). The
  conclusion is unchanged — the inverse still beats raw `|D|` at every noise level.
- **exp 18** (`beeline_diagnostics.md`): "18 of 76 true edges are in bidirectional pairs"
  mislabeled a pair count as an edge count — there are 18 reciprocal *pairs*, i.e. 36 of 76
  directed edges. Corrected, and added a small-denominator caveat (VSC and mCAD have only 5
  orientable non-reciprocal edges each, so VSC's orientation = 1.00 is low-power; verified
  against the ground-truth networks: VSC 5, mCAD 5, HSC 14, GSD 40 orientable edges).
- **exp 21** (`causalbench_response_geometry.md`): tightened one sentence that called the
  0.70 cross-split result "verified … recovers real direction." Split-half reproducibility
  shows the interventional direction is a stable, sample-independent property; it is not a
  check against a ground-truth graph (RPE1 has none), as the same section notes elsewhere.
- **README / experiment_summary / project_retrospective**: experiment count 27 → 28, test
  count 145 → 158, and exp 28 (the separability phase diagram) integrated into the experiment
  log and the cross-experiment narrative.

## Residual minor items (noted, not changed)

- **exp 22**: the polished write-up reads the dominant mode as "real biology — a cell-cycle
  program," while its auto-generated debug report's crude headline rule labels it
  "technical/abundance-linked." Same underlying correlations; the write-up's both/and reading
  (abundance-concentrated *and* a coherent cell-cycle program) is the more careful one. Worth
  reconciling the auto-headline if these are ever merged.
- **exp 28**: the `separability.py` `entanglement` docstring describes the intended behavior
  of a knob the headline sweep does not exercise (it uses entanglement 0); empirically,
  deflation does not fully collapse to the raw baseline at entanglement 1. Verify this axis
  behaves as described before building on it.
- **exp 06**: a three-way tie in the multifactorial top-3 in-hub table is presented as a single
  "best" method, resolved only by a deterministic secondary tiebreak.
- **Cosmetic**: a pandas `FutureWarning` in exp 02's aggregation; several summary CSVs mix
  per-network and mean rows under a `row_type` column (a trap for an unfiltered reader, not an
  error in any write-up).

## What could not be independently reproduced

The real RPE1 numbers could not be regenerated from source, because the raw `h5ad` is absent.
Verification of exp 19–28 real-data figures therefore rests on internal consistency across
write-up ↔ debug report ↔ summary CSV ↔ raw per-edge CSV (all mutually consistent), plus the
passing synthetic controls and unit tests that exercise every library function those
experiments depend on. Restoring the raw dataset and re-running would close this gap.
