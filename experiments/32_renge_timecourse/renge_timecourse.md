# Experiment 32: time-resolved knockout response on RENGE Perturb-seq

> Correction (methodology audit). The time-resolved STRUCTURE (response growth, net_out ordering
> stability) is real and reproducible. But the graded recovery is against STRING, which is
> UNDIRECTED, so this is SKELETON recovery, not the directed recovery the Direction B thesis is
> about; there is no directed validation here. The win over a hand-built observational correlation
> control is marginal (0.366 vs 0.323) and reverses on day 4 (observational 0.407). No established
> GRN method (GENIE3/GRNBoost2) was compared. exp 33 adds directed grading via TRRUST.

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

## Graded recovery vs STRING (external proxy)

The per-day interventional response is graded against the STRING functional network (58 edges
among the 23 TFs at combined score >= 0.4; skeleton chance 0.229). STRING is undirected, so this
grades the interaction skeleton.

| day | interventional AUPR | observational AUPR | chance |
| --- | --- | --- | --- |
| day2 | 0.310 | 0.283 | 0.229 |
| day3 | 0.380 | 0.268 | 0.229 |
| day4 | 0.390 | 0.407 | 0.229 |
| day5 | 0.385 | 0.333 | 0.229 |

The interventional knockout response recovers STRING functional links above chance (0.31 to 0.39
vs 0.23), recovery strengthens over the time course (0.310 at day 2 to 0.385 at day 5), and it
beats the observational control-cell correlation baseline on average (mean 0.366 vs 0.323) and on
three of four days. The win over observational is real but modest; the day-4 observational value
(0.407) is a single-day spike.

## Status and next step

The time-resolved structure (growth, ordering stability) and skeleton recovery against STRING are
graded here. The further step is DIRECTED grading against a TF-target or ChIP network (STRING is
undirected): the exact RENGE ChIP set (19 genes, threshold 300) is in the paper supplement, not
the GEO download; a downloadable directed curated TF-target database (TRRUST, DoRothEA) is the
alternative. That would test directed recovery on real interventional time-series, comparable to
exp 30 and exp 31.

## Outputs

Under `results/tables/` (git-ignored): `renge_timecourse_by_day.csv`,
`renge_timecourse_netout_crossday.csv`, `renge_timecourse_summary.csv`,
`renge_timecourse_debug_report.md`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/32_renge_timecourse/run_renge_timecourse.py
```
