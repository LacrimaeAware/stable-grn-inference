# Experiment 20 — CausalBench RPE1 interventional diagnostics

First run on **real interventional data** (Replogle/Weissman RPE1 Perturb-seq, CRISPRi).
This closes the regime ladder set up in experiments 17–19:

```
DREAM4 (lagged time-series)        -> orientation ~free (temporal precedence)
BEELINE Curated (static obs)       -> orientation weak & network-dependent
CausalBench RPE1 (interventional)  -> orientation becomes IDENTIFIABLE  <- this experiment
```

## Data + how it was loaded

The user downloaded the **raw genome-wide** Replogle RPE1 file (`rpe1_raw_singlecell_01.h5ad`,
~8.7 GB, dense **247,914 cells × 8,749 genes**) — not CausalBench's smaller preprocessed
file. A memory-efficient loader (`load_replogle_raw_h5ad`) was added that:

- reads only the **perturbed∩measured gene-column block** in row chunks (never densifies
  all 8,749 genes),
- normalizes each cell by `obs['UMI_count']` (= scanpy `normalize_per_cell` to the median
  total) then `log1p` (matching CausalBench preprocessing),
- keeps perturbations with **>100 cells** (CausalBench's 383 comes from extra
  knockdown-strength filtering not applied; >100-cell gives **651** perturbed&measured genes).

Loaded working set: **651 genes, 139,825 cells, 11,485 control (`non-targeting`) cells**,
423,150 directed candidate edges (perturbed source × target). The 11,485 controls match
CausalBench's reported RPE1 observational count exactly — a good sanity check.

## Results (full run; subsampling n_ctrl=4000, n_pert=400, 12 control-null splits)

### 1. Orientation becomes identifiable under intervention

For both-perturbed pairs, direction is read from the interventional asymmetry
(`effect(A→B) = Wasserstein(B | knockdown A, B | control)`; predict A→B if
effect(A→B) > effect(B→A), decisive when the gap beats the control-null scale).

| regime | orientation signal |
| --- | --- |
| DREAM4 (lagged) | orientation-given-skeleton 0.88–0.96 |
| BEELINE Curated (static obs) | 0.50–1.0, network-dependent (GSD collapses) |
| **CausalBench RPE1 (interventional)** | **decidability 0.606** on 211,575 both-perturbed pairs |

Observational symmetric scores are undecidable (0.5 by construction); under intervention
**60.6%** of pairs have a decisive directional asymmetry. **Direction is identifiable under
intervention** in a way it simply is not from static observational data. That is the
concrete payoff the whole regime ladder was built to test.

### 2. But observational co-expression barely predicts interventional response (deflationary)

Transfer test (non-circular): observational methods on **control cells only**, scored
against the held-out interventional reference (control-null-thresholded effects).

| method (control cells) | AUPR | **AUROC** | precision@50 |
| --- | --- | --- | --- |
| correlation | 0.864 | **0.571** | 0.98 |
| sparse (LASSO) | 0.822 | **0.506** | 0.98 |
| random baseline | 0.819 | 0.499 | 0.80 |

The reference is **dense (0.82)** — in Perturb-seq, knocking a gene down shifts a large
fraction of the transcriptome (direct + indirect + global cell-state effects), so "does
perturbing A change B at all" is true for most pairs. AUPR is therefore dominated by that
0.82 floor; **AUROC is the honest metric**, and it says observational co-expression carries
only a **weak** signal for interventional response (correlation AUROC 0.571; sparse ≈ chance
0.506). Unperturbed structure is a poor proxy for what intervention actually does.

### 3. Observational orientation is not just uninformative — it is anti-correlated

Agreement between the interventional-implied direction and the observational sparse
direction: **0.329** over 4,969 decidable+oriented pairs (< 0.5, not noise at this n).
Observational edge direction systematically **disagrees** with interventional direction.
This is the strongest statement yet that you **cannot** read causal direction off
observational single-cell data — it actively misleads.

### 4. The theory-predictable penalty transfers (with corrected wording)

At n≫p (4,000 control cells, p=651): CV-best α=0.05, BIC-best α=0.1, theory
α=1.1·√(2 ln p / n)=**0.063**. The earlier exp17/18 phrasing "tiny α" was wrong here — the
values are *moderate* — but the real transferable claim holds: **theory α lands right
between CV and BIC**, so the penalty is sample-complexity-predictable across all three
regimes (DREAM4, BEELINE, CausalBench).

## What transfers across the full regime ladder

| claim | DREAM4 | BEELINE Curated | CausalBench RPE1 |
| --- | --- | --- | --- |
| theory-predictable penalty | ✓ | ✓ | ✓ (α 0.063 ≈ CV/BIC) |
| stability-selection strong thesis | ✗ (not supported) | ✗ | (not retested; expected ✗) |
| orientation from static scores | n/a (had time) | weak/variable | **anti-correlated (0.33)** |
| orientation under intervention | — | — | **identifiable (0.61)** |
| observational→causal transfer | — | — | **weak (AUROC 0.57)** |

## Honest caveats

- **Decidability ≠ verified accuracy.** There is no exact directed ground truth, so 0.606 is
  the fraction of pairs where intervention *lets you decide* a direction, not a verified
  correctness rate. The asymmetry is directional *evidence*, not proof (indirect/downstream
  effects, compensation, off-target knockdown, cell-state shifts).
- **The interventional reference is "any measurable shift," not a sparse direct-causal
  graph** (density 0.82). It conflates direct and indirect effects. A sparser, more-direct
  reference (top-k effects per source; require strong self-knockdown) is the natural exp21
  refinement.
- **>100-cell filter, not CausalBench's strong-perturbation filter** (skipping the
  summary-stats download). Results are on a slightly larger, noisier perturbation set.
- Subsampling (4,000 control / 400 per perturbation) bounds the Wasserstein cost; seeded
  and reproducible, but estimates carry sampling noise.

## Reproduce

```
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/20_causalbench_rpe1_interventional/run_causalbench_rpe1.py --quick  # 200-gene cap
.\.venv\Scripts\python.exe -B experiments/20_causalbench_rpe1_interventional/run_causalbench_rpe1.py          # 651 genes
```

Place `rpe1_raw_singlecell_01.h5ad` (or CausalBench's `rpe1.h5ad`) under
`data/raw/causalbench/`. Tests never depend on the real file (synthetic h5ad fixtures).
Artifacts (git-ignored `results/`): `causalbench_rpe1_{summary,effect_edges,observational_scores,orientation_asymmetry,interventional_reference}.csv`,
`causalbench_rpe1_debug_report.md`, `results/figures/causalbench_rpe1_effects_and_asymmetry.png`.

## Verdict

The regime ladder is complete and the central thesis is now empirically grounded across
three regimes: **directed GRN inference is identifiability-limited, and identifiability is
set by the data regime, not the estimator.** Static observational data cannot orient edges
(and observational direction is actively anti-correlated with the truth); interventional
data makes the majority of directions decidable. Meanwhile observational co-expression is a
weak predictor of interventional response, and the LASSO penalty is theory-predictable
throughout. The most valuable next step is a **sparser, more-direct interventional
reference** (exp21) so "transfer" and "orientation accuracy" can be measured against direct
causal effects rather than the broad transcriptional response.
