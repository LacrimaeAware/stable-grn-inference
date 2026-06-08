# Experiment 40: identifiability pipeline (forward direction, roadmap step 1)

The first step of the math/stats-fit direction in `docs/roadmap.md`: build and validate the parameter
identifiability and inference pipeline that would later be applied to a real mechanistic adaptation
model (for example Yildirim's lac-operon delay-differential-equation model). This is tooling, not the
contribution yet; the contribution begins when a published model is reproduced and its parameters are
analyzed.

## Method and validation target

The pipeline (simulate, maximum-likelihood fit, profile likelihood, Fisher information) is validated on
the textbook mRNA -> protein cascade `dm/dt = k_m - d_m m`, `dp/dt = k_p m - d_p p`, where the answer is
known analytically: observing protein only, the transcription rate k_m and translation rate k_p are not
separately identifiable (protein depends on them only through the product k_m k_p); observing mRNA as
well makes them identifiable. The pipeline must recover this.

Tooling: `src/stable_grn_inference/dynamics/identifiability.py`, tested in
`tests/test_identifiability.py`.

## Result (verified)

- Fisher information, protein-only observation: rank 3 of 4 (rank-deficient), the k_m / k_p direction is
  the null direction. mRNA + protein: rank 4 of 4 (full). Verified directly.
- Profile likelihood (the slower confirmation): k_m is flat under protein-only observation
  (non-identifiable) and bounded when mRNA is also observed (identifiable); the product k_m k_p is
  recovered even when the individual rates are not.

So the pipeline recovers the known structural-identifiability answer. It is correct, and ready to be
pointed at a real model whose answer is not known in advance.

## Status

This is tooling validated on a known case. The next step (roadmap step 2) reproduces a published
Yildirim adaptation / lac-operon model from the paper (not from memory), matches its reported behavior,
and reports which parameters are identifiable under realistic measurements and what experiment would
resolve the unidentifiable ones. That step, not this one, is the candidate contribution.

## Outputs

Under `results/tables/` (git-ignored): `identifiability_pipeline_summary.csv`,
`identifiability_pipeline_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/40_identifiability_pipeline/run_identifiability_pipeline.py
```
