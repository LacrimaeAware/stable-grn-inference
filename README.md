# stable-grn-inference

**A research log: can you recover *who regulates whom* from gene-expression data — and what it actually takes to pull structure out of messy dynamic data?**

This repo is an honest, end-to-end exploration of **gene regulatory network (GRN) inference**, run across three kinds of data — a clean simulator (DREAM4), real single-cell snapshots (BEELINE), and real CRISPR-perturbation data (CausalBench / Replogle RPE1 Perturb-seq). It documents **25 experiments** the way research actually goes: what we hoped, what happened, the mistakes in hindsight, and the lesson — negatives included.

The headline finding is unglamorous and real:

> **Whether you can recover causal *direction* is set by the kind of data you have, not by how clever the method is.**

![the regime ladder](docs/figures/fig1_regime_ladder.png)

---

## What we found (honest TL;DR)

- **Direction needs the right data, not a better model.** Time-series → direction is nearly free; static snapshots → near-impossible; interventions → recoverable. *(figure above)*
- **The original thesis — "stability selection makes inference more reliable" — does not hold.** Tested directly with the proper math; it underperforms a single well-tuned fit.
- **The one clean, transferable positive:** the right amount of regularization is **predictable from sample-complexity theory** (√(log p / n)) in *every* regime — not something you have to grid-search.
- **On real CRISPR data, simple methods barely predict interventional effects** — and the field's own benchmarks agree: even deep-learning foundation models often fail to beat simple baselines here. This is an *unsolved frontier*, not a thing we were "losing" at.
- **Every clever idea worked on clean synthetic data and dissolved on real data** — because real Perturb-seq is dominated by one convergent biological "cascade."

![clean vs real](docs/figures/fig2_clean_vs_real.png)

---

## The story in three acts

**Act I — DREAM4 (a clean simulator).** Built the whole pipeline and learned the mechanics: temporal order beats a fancier model, regularization tracks network density, fusion helps only when methods make *different* mistakes, and the stability-selection thesis quietly failed. Decent edge recovery (AUPR ≈ 0.65 vs ≈ 0.33 for plain correlation) — but it's a simulator, so "decent" was expected, not novel.

**Act II — BEELINE (real single-cell).** Ported the exact diagnostics to a real benchmark and found DREAM4's conclusions are **regime-specific**: on static snapshots, edge direction collapses toward a coin-flip and depends heavily on the network. The lesson that became the spine of the project: *the data regime decides what's knowable.*

**Act III — CausalBench / RPE1 (real CRISPR).** Direction returns under intervention and is even *reproducible* across independent cells — but the response is buried under a giant, convergent **cell-cycle cascade**, and no simple structure cleanly survives it.

![the cascade](docs/figures/fig3_cascade.png)

### Why real data is hard: the "whirlpool"

Knock out almost *any* essential gene and the cell runs the **same** emergency program — it stops dividing, moving hundreds of cell-cycle genes together at once. That convergent program is **53% of every perturbation response.** The thing we actually want — "gene A specifically controls gene B" — is a tiny signal riding on top of it, and tangled with it. That single fact explains every faint result in this repo. It's the geometry of the data, not a failure of the methods.

---

## What we got right, and wrong (honest)

**Right** — a portable diagnostic framework; honest negatives that *match the published literature*; the theory-predictable-penalty result; and a clear map of *why* the problem is hard and where the intuitions live in real mathematics.

**Wrong** — we over-narrated early results ("it worked! let's try the next dataset!") instead of reasoning about the data's geometry *first* and predicting that, e.g., a static-snapshot method can't recover direction. The corrected habit: **predict from the geometry, then test only to confirm.** Two days of this also produced a lot of "promising" framings that honest re-checks (multi-seed audits, magnitude controls) later deflated — left in the repo on purpose, because that's what the work actually looked like.

---

## Experiment log

| # | experiment | what we hoped | what happened | lesson |
|---|---|---|---|---|
| 01–04 | DREAM4 baselines, GENIE3 | a method beats correlation | correlation is a strong baseline; GENIE3 wins some | don't assume "more ML" wins |
| 07 | lagged time-series | does temporal order help? | big jump (AUPR 0.30→0.53) | **directional info in the data > clever model** |
| 08–10 | dynamic sparse, Size100 scaling | the Size10 winner scales | it collapsed at 100 genes | small-network results can be artifacts |
| 11–14 | calibration, fusion, mechanism, gold-free | explain & deploy the winners | α tracks density; fusion needs complementary errors | regularization is predictable, not magic |
| 15–18 | BEELINE adapter + diagnostics | DREAM4 results transfer | **they don't** — direction is regime-specific | identifiability is set by the data regime |
| 19 | interventional benchmark scouting | pick data where direction is identifiable | chose CausalBench (real CRISPR) | go where the evidence actually is |
| 20 | CausalBench RPE1 diagnostics | direction returns under intervention | yes (0.61 decidable) but reference is dense | interventions help; the response is broad |
| 21 | response geometry | structure in the response matrix | reproducible direction (0.70); 53% is one mode | the cascade is real and dominant |
| 22 | covariate-aware cleaning | subtract the cascade → clean core | removing it *hurts*; it's real biology | the nuisance is entangled with the signal |
| 23 | inverse / "find the stick" | recover wiring from total response | perfect on toy, useless on real | the linear model is too clean for biology |
| 24 | transferable structure | predict a held-out perturbation | no — each is individualistic | no shared low-rank code to exploit |
| 25 | counterfactual factor atlas | separate core from nuisance features | **works on planted ground truth**; doesn't transfer to genes | the idea is sound where factors are separable |

*Each experiment has its own write-up under `experiments/NN_*/` with the full numbers, the honest verdict, and (where relevant) figures.*

---

## The bottom line, and the pivot

The "novel gene-network result" path is **effectively closed with these tools** — RPE1 is a genuine unsolved frontier dominated by the cascade, and that was established honestly in two days. The durable value is the understanding.

And the project found its real question along the way: the actual interest was never genes, it was **decomposing dynamic data into mathematical components** — which is a real, named field. The intuitions here map directly onto:

- **state-space / dynamical-systems models** (`dx/dt = f(x) + u`),
- **Dynamic Mode Decomposition (DMD) / Koopman theory** — born from fluid dynamics, the rigorous version of "subtract the dominant mode, study the deviations,"
- **SINDy** — literally "recover the differential equation from movement data."

The gene cascade was one ugly, real-world instance of exactly that problem.

---

## Read more

| document | what it is |
|---|---|
| [`docs/project_retrospective.md`](docs/project_retrospective.md) | plain-language walk through everything (exp 1–22), **with the statistics explained from scratch** |
| [`docs/project_retrospective_part2.md`](docs/project_retrospective_part2.md) | the hands-on half (exp 23–25 + the theory-crafting) and the pivot |
| [`docs/regime_ladder_report.md`](docs/regime_ladder_report.md) | the paper-style consolidated writeup of the interventional arc |
| [`docs/experiment_summary.md`](docs/experiment_summary.md) | the running, detailed results log |

## Reproduce

```bash
# Python 3.13, deps in requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests          # 145 tests
.\.venv\Scripts\python.exe -B experiments/<NN_name>/run_*.py --quick # any experiment
```

Real datasets (`data/`) and generated tables (`results/`) are git-ignored; the test suite never depends on them (it uses synthetic fixtures). Committed figures are regenerated by `docs/figures/make_figures.py`.

## Layout

```text
stable-grn-inference/
├── src/stable_grn_inference/   # library: data adapters, inference, evaluation, analysis
├── experiments/                # 25 experiments, each with a write-up + script + tests
├── docs/                       # retrospectives, reports, figures
└── tests/                      # 145 tests, synthetic fixtures only
```
