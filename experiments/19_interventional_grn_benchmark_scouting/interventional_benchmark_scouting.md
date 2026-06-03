# Experiment 19 — Interventional / perturbation benchmark scouting

## Why this experiment exists

Experiments 17–18 established the regime ladder:

```
DREAM4 (lagged time-series):  orientation ~free (temporal precedence), error is skeleton-bound.
BEELINE Curated (static obs):  orientation weak & network-dependent (GSD ~0.4, VSC ~1.0).
  -> the missing regime: INTERVENTIONAL data, where direction is identifiable by design.
```

The point of going interventional is **not "better edge labels."** Real Perturb-seq
reference graphs are still biological proxies (CORUM/STRING/ChIP-seq). The point is a
different, more causal **evaluation paradigm**: held-out interventional prediction —
does perturbing a putative parent actually move the child distribution? That is what
makes direction identifiable, and it is exactly the weakness experiment 18 exposed.

This is a **scouting + de-risking** experiment: choose the target, define the adapter
shape, prove the rebuilt orientation diagnostic works on a positive control, and record
blockers — **without downloading any large dataset**.

## Candidate comparison

| candidate | source / install | data type | size (cells / genes / perturbations) | perturbation labels | held-out interventional eval | reference edges (type) | TF/regulator list | adapter fit | difficulty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **CausalBench** (Chevalley et al. 2023; Comms Bio 2025) | `pip install causalscbench`; data from Replogle et al. screen (CC-BY-4.0) | scRNA-seq CRISPRi Perturb-seq, RPE1 + K562 | RPE1 ~162.7k cells / 383 interventions / ~10–11k obs cells; K562 ~162.8k cells / 622 interventions | **yes**, per-cell knockdown target + control | **yes — first-class**: mean Wasserstein(perturbed vs control) + Mann–Whitney FOR | proxy: CORUM, STRING (physical / full), ChIP-seq (K562; HepG2 proxy for RPE1) | regulators = the perturbed gene set | **high** — maps directly onto `InterventionalDataset` | **medium** |
| Replogle 2022 raw Perturb-seq | GEO / plate-level downloads | genome-wide CRISPRi | ~2.5M cells genome-wide; hundreds of GB | yes | build-your-own | none packaged | n/a | low (huge, need heavy preprocessing) | **hard** |
| Norman 2019 (GI / dual perturb) | GEO GSE133344 | Perturb-seq, single + paired KO | ~100k cells; ~100 perturbations | yes (incl. genetic-interaction pairs) | possible | none packaged | n/a | medium (good for epistasis, not packaged) | hard |
| Frangieh 2021 (melanoma) | GEO / SCP | Perturb-seq under conditions | ~200k cells | yes | possible | none packaged | n/a | medium | hard |

Facts verified via the CausalBench paper (Nature Communications Biology 2025,
PMC11897147) and the `causalscbench` GitHub README; raw-dataset sizes are approximate.

## Decision: CausalBench first

CausalBench is the default target, and nothing in scouting argues against it:

1. **It is built around the right evaluation.** Held-out interventional scoring (mean
   Wasserstein of child under parent-knockdown vs control, plus a Mann–Whitney
   false-omission-rate) is the paradigm indicated by experiment 18. It sidesteps
   the proxy-reference problem instead of importing it.
2. **It is packaged and reproducible** (`pip install causalscbench`), unlike raw
   Replogle/Norman/Frangieh which need bespoke heavy preprocessing.
3. **It carries both regimes in one object**: ~10–11k observational cells *and*
   hundreds of single-gene interventions per cell line, so our observational diagnostics
   (skeleton, alpha selection, fusion, stability) and the new interventional orientation
   test run on the *same* data — a clean within-dataset comparison.
4. **It is the right size to be careful about**: ~163k cells × hundreds of genes per
   line. Not tiny, so it is not auto-downloaded; it is staged deliberately.

### Why it beats BEELINE Curated for directionality

BEELINE Curated is exact-labeled but static and tiny (5–19 genes), so it can rank
edges but cannot *causally* settle direction — which is precisely why GSD's orientation
collapsed. CausalBench replaces "compare two static edge scores" with "compare two
interventions," turning orientation from a near-non-identifiable observational quantity
into a measured asymmetry.

## What transfers from experiments 17–18, and what must be rebuilt

| diagnostic | transfers as-is? | notes |
| --- | --- | --- |
| skeleton vs undirected AUPR | **yes** | observational expression still gives a skeleton; score it the same way. |
| alpha selection (CV / BIC / sqrt-LASSO) | **yes** | the most robust positive; n≫p here too, so expect tiny optimal α. Run on the observational subset. |
| fusion 3-arm (cross vs within-bootstrap) | **yes** | regime-dependent per exp18 — test, don't assume. |
| stability selection (MB bound) | **yes** | expect the same negative; keep selection frequencies as confidence only. |
| **orientation-given-skeleton (observational)** | **NO — replace** | comparing static i→j vs j→i scores is uninformative (symmetric scores → 0.5). Replace with the **interventional asymmetry** test below. |
| **EPR / early precision** | partial | meaningful only against a reference; with proxy refs, report it as agreement and pair with the interventional metric. |

### The rebuilt orientation diagnostic (the key new idea)

For an unordered pair {A,B} where **both** A and B were perturbed:

```
effect(A->B) = Wasserstein( expr_B | perturb(A) ,  expr_B | control )
predict A -> B  iff  effect(A->B) > effect(B->A)
```

Implemented as `interventional_orientation_asymmetry` in
`src/stable_grn_inference/data/interventional.py`. **Limitation, stated up front:** this
requires interventions on *both* endpoints, so it only scores the subset of pairs where
both genes were perturbed. And it is **exploratory** — indirect/downstream effects,
genetic compensation, off-target knockdown, and cell-state shifts can all distort the
asymmetry. It is directional *evidence*, not a causal guarantee.

## De-risking deliverables produced here (zero download)

- **`InterventionalDataset`** dataclass + `load_interventional_frames` — the model-agnostic
  container (cells×genes, per-cell perturbation label, control mask, perturbed-gene-restricted
  candidates, optional proxy reference, densified `edge_labels`, `reference_kind`).
- **`make_synthetic_interventional`** — a tiny linear-SEM fixture (known DAG, control +
  single-gene knockdowns) matching the expected on-disk schema.
- **`interventional_effect_matrix`** and **`interventional_orientation_asymmetry`** — the
  CausalBench statistical signal and the rebuilt orientation test.
- **`run_interventional_dry_run.py`** — exercises all of the above with no download.

### Dry-run result (positive control)

On the synthetic acyclic SEM (7 genes, 120 cells/condition):

| diagnostic | orientation accuracy |
| --- | --- |
| interventional asymmetry (NEW) | **1.000** |
| observational \|correlation\| (OLD) | 0.500 (symmetric → tie by construction) |

Interventional effect separation true−false = **+5.0** (Wasserstein). This confirms the
adapter + diagnostic are correct and that the rebuilt orientation test does the thing
the observational one cannot.

## Expected real-data schema (for the ingest path)

```
expression.csv      cells (rows) x genes (cols)   # log-normalised counts
perturbations.csv   one column: per-cell target gene name, or "control"
references (proxy):  CORUM / STRING / ChIP-seq edge lists (source,target)
```

`load_interventional_frames(name, expression_df, perturbation_series, reference_edges=...)`
ingests exactly this. The CausalBench `causalscbench` loader returns expression + a
per-cell intervention label, which maps onto these two frames directly.

## Blockers / open questions

1. **Download footprint not auto-resolved.** `causalscbench` is not installed in this env
   (the dry-run reports this). Installing it and fetching RPE1/K562 is a deliberate,
   user-approved step — *not* automated, per the "no large auto-download" constraint.
2. **Proxy references.** CausalBench's database references (CORUM/STRING/ChIP-seq) are
   proxies; treat AUPR against them as agreement, and lead with the interventional metric.
3. **Orientation coverage.** The asymmetry test only scores pairs where both endpoints are
   perturbed. On CausalBench many measured genes are never intervened, so orientation will
   be reported on a subset, with explicit coverage.
4. **Compute.** ~163k cells; the observational diagnostics subsample fine, but the
   per-pair Wasserstein effect matrix over hundreds of perturbed genes needs care
   (vectorise / restrict to the perturbed × measured block).

## Exact next coding step (experiment 20)

1. User runs `pip install causalscbench` and downloads RPE1 (smaller: 383 interventions)
   into `data/raw/causalbench/` — staged, not automatic.
2. Add a thin `load_causalbench(...)` that wraps the `causalscbench` loader into
   `load_interventional_frames` (expression + per-cell label → `InterventionalDataset`).
3. Run the transferring diagnostics on the observational subset (skeleton, alpha, fusion,
   stability) **plus** the rebuilt `interventional_orientation_asymmetry` on the perturbed
   block, with the coverage caveat.
4. Question for exp20: *does orientation become identifiable under intervention
   (asymmetry accuracy ≫ 0.5 on real data), and do the exp17 positives (theory-α, skeleton
   framing) still hold in a genuinely causal regime?*

## What should be committed

- `src/stable_grn_inference/data/interventional.py` (+ `data/__init__.py` exports)
- `experiments/19_interventional_grn_benchmark_scouting/` (this doc + dry-run script)
- `tests/test_interventional_data.py` (7 tests)
- doc updates (experiment_summary, update_map, project_plan)
- `results/` stays git-ignored.

Suggested message: `Scout interventional benchmarks; choose CausalBench + interventional orientation diagnostic (exp 19)`.
