# Experiment 32: time-resolved knockout response on RENGE Perturb-seq

Direction B on real time-resolved data. RENGE (GEO GSE213069) is real single-cell CRISPR
knockout in human iPSCs across four daily timepoints (days 2 to 5), 23 knocked-out
transcription factors, non-targeting controls. Unlike the static RPE1 snapshot, it has a real
time axis, so the knockout response can be watched build over days.

## Data and loader

Standard 10x CRISPR guide-capture output per day (60,683 gene-expression features + 50 guides,
about 4,700 cells/day). The loader (`src/stable_grn_inference/data/renge.py`,
`load_renge_timecourse`) assigns each cell its dominant guide, maps the guide to its target
gene (`SOX2_1` to `SOX2`), treats AAVS1/CTRL guides as control, and builds a per-cell-normalized
response over the 23 perturbed transcription factors (the shared square block across days).
Tested offline in `tests/test_renge.py`. Download: the four daily tarballs from GSE213069 (about
360 MB total), extracted under `data/raw/renge/`.

## Result (4 days, 23 perturbed TFs)

| day | control cells | response norm ||D|| | median split-half cosine |
| --- | --- | --- | --- |
| day2 | 222 | 2.79 | 0.45 |
| day3 | 344 | 2.87 | 0.57 |
| day4 | 463 | 2.95 | 0.57 |
| day5 | 422 | 3.79 | 0.66 |

- Response magnitude grows across the time course (||D|| 2.79 at day 2 to 3.79 at day 5): the
  knockout effect builds over real time as it propagates, a cascade accumulating that a static
  snapshot cannot show.
- The directional ordering (net_out, upstream vs downstream) is reproducible across days (mean
  cross-day Spearman 0.75, adjacent-day 0.80) and stabilizes over time (day2-day3 0.67,
  day3-day4 0.83, day4-day5 0.92). The directional axis persists and sharpens as the response
  strengthens.
- Within-day reproducibility (median split-half cosine) rises from 0.45 (day 2) to 0.66
  (day 5): the response becomes more reproducible as it strengthens.

## Status and next step

This establishes the unsupervised time-resolved structure (response growth, ordering stability)
on real time-resolved Perturb-seq. The completing step is directed-edge grading against the
RENGE ChIP-seq proxy network (19 genes, binding threshold 300), which is not in the GEO download
and lives in the RENGE repository; fetching it turns this into a graded directed-recovery test
comparable to exp 30 and exp 31 on real interventional time-series data.

## Outputs

Under `results/tables/` (git-ignored): `renge_timecourse_by_day.csv`,
`renge_timecourse_netout_crossday.csv`, `renge_timecourse_summary.csv`,
`renge_timecourse_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/32_renge_timecourse/run_renge_timecourse.py
```
