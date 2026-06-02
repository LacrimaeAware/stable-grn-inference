# Modern GRN Benchmark Adapter — Scouting & Recommendation

Goal: choose **one** modern GRN benchmark to validate the calibrated-confidence
pipeline (experiments 11–14) beyond DREAM4/GNW, and design the adapter that lets
our existing scorers/metrics run on it.

Scope guardrails for this task: scouting + interface design only. No dataset
download, no large experiment, no new models, no RL/NN. GeneNetWeaver is parked
as a possible future controlled-simulation tool.

> Verification caveat: the data-source details below are from prior knowledge,
> not a live fetch. Treat exact URLs, sizes, and license terms as "verify at
> download time." The recommendation does not depend on any single unverified
> number.

## Candidates compared

| Candidate | Data type | Directed ref edges? | Ref-edge nature | Scale (rough) | TF list | Perturbation labels | Pseudotime/time | Download/use difficulty | License/access |
|---|---|---|---|---|---|---|---|---|---|
| **BEELINE** (Pratapa et al. 2020, `Murali-group/Beeline`) | scRNA-seq (real: hESC/hHep/mESC/mDC/mHSC; synthetic: BoolODE) | **Yes** (TF→target) | synthetic = **exact**; real = **ChIP-seq / curated proxies** (cell-type ChIP, non-specific ChIP, STRING) | real ~hundreds–few thousand cells, genes filtered to ~500–1000; synthetic tiny | **Yes** (organism TF lists) | No | **Yes** (PseudoTime.csv, slingshot) | **Low–moderate** (CSV inputs, standard layout, Zenodo archive) | code GPL-3.0; data from public GEO/curated sources — research use, verify redistribution |
| **CausalBench** (Chevalley et al. 2023) | large-scale CRISPRi **Perturb-seq** (Replogle 2022: RPE1, K562) | Partial | **interventional** evaluation (does perturbing A shift B?) rather than a fixed curated edge list | **very large**: ~150k–170k+ cells/dataset, ~genome-scale genes, ~hundreds–thousands of perturbations | perturbed-gene set (not a TF list per se) | **Yes** (core) | No | higher (big downloads, framework-specific eval, compute) | code permissive; data CC-BY (verify) |
| **Raw Perturb-seq / CRISPR** (Replogle 2022, Norman 2019, Dixit 2016; via `scPerturb`) | interventional scRNA-seq | **No** (must supply) | would be **curated priors** (DoRothEA/TRRUST/RegNetwork) or interventional-derived | large | external (DoRothEA/humanTFs) | **Yes** | No | **High** (assembly required: pair expression with an external reference) | mostly public CC; per-dataset |
| Other maintained refs/sims | DoRothEA/TRRUST/RegNetwork (priors); SERGIO (synthetic SC sim); scPerturb (data hub); SCENIC+/GRETA (heavier) | varies | priors / exact (SERGIO) | varies | varies | SERGIO no; scPerturb yes | SERGIO/sim only | varies | varies |

### Reference-edge honesty (critical)

This is the single most important difference from DREAM4. DREAM4 gold standards
are the **exact** simulator networks. Modern real benchmarks do **not** have exact
gold edges:

- **BEELINE real datasets** → ChIP-seq-derived or curated/STRING references. ChIP
  binding ≠ functional regulation, so these are **noisy biological proxies**. AUPR
  here measures *agreement with a prior*, not recovery of truth. BEELINE itself
  reports **EPR (early precision ratio)** = precision in the top-k normalized by
  random, precisely because absolute AUPR against a proxy is hard to interpret.
- **BEELINE synthetic (BoolODE)** → **exact** ground truth (good for adapter
  validation and method sanity, but synthetic).
- **CausalBench** → **interventional** signal (perturb A, measure B). Arguably
  closer to causal truth than ChIP proxies, but it is a *different evaluation
  paradigm* (statistical interventional effects), not a fixed directed edge table.

Implication: report AUROC/AUPR/precision@k **and EPR**, and label the reference
kind explicitly so we never overclaim "ground-truth recovery" on proxy edges.

## What of our pipeline transfers

| Component | Transfers to single-cell? | Notes |
|---|---|---|
| Correlation ranking | **Direct** | correlation across cells; BEELINE's simplest baseline. |
| GENIE3 / tree importance | **Direct** | BEELINE's headline method family (GENIE3/GRNBoost2) — same shape. |
| Sparse **static** LASSO (exclude-self) | **Direct** | regress each gene on candidate regulators across cells. |
| Sparse **dynamic** lagged LASSO + include-self persistence | **Needs redesign** | no real time in snapshot scRNA-seq. Either drop, or build lagged pairs **along pseudotime** (order cells by PseudoTime, form adjacent pairs within a trajectory). The include-self *persistence* story is DREAM4-time-series-specific and may not map. |
| Calibrated alpha selection (CV / BIC / AIC / density-prior) | **Transfers** | CV across cells; BIC/AIC unchanged; **density-prior is especially relevant** since the candidate set and reference density differ from DREAM4. |
| Confidence / rank fusion (Borda, MRR, agreement, reciprocal penalty) | **Direct** | fuse correlation + GENIE3 + sparse; equal-weight, gold-free — exactly the calibrated-confidence claim we want to validate. |
| Topology metrics | **Mostly** | degree/hub/reciprocal fine on sparse adjacency. **FFL is O(n³)** — infeasible at thousands of genes; restrict to the TF-induced subgraph or skip. Reciprocal edges only meaningful among TF–TF pairs (candidate set is TF→gene). |
| Calibration bins / reliability / ECE | **Direct** | confidence-vs-true-rate over the candidate edges. |
| Wavelet / scattering preprocessing | **Largely N/A** | designed for time-series signals; only meaningful along pseudotime trajectories. Low priority. |

### What needs redesign (summary)

1. **Candidate edge set**: TF→gene (sources restricted to a TF list), not all-pairs.
   Changes precision@k denominators, predicted/true density, and topology semantics.
2. **Reference semantics**: noisy proxy / interventional, not exact gold → add EPR
   and a `reference_kind` label; treat AUPR as prior-agreement.
3. **Scale + preprocessing**: thousands of genes × thousands of (sparse, zero-
   inflated) cells → normalization (log1p / library-size), gene filtering
   (highly-variable + TFs), and avoiding O(n²)/O(n³) blowups.
4. **Dynamic component**: pseudotime-ordered lagged pairs if we want the temporal
   sparse model; otherwise run the static sparse + GENIE3 + correlation + fusion.

## Adapter interface design

A single dataset-agnostic container that all current scorers/metrics consume.
Illustrative sketch (not implemented in this task):

```python
@dataclass
class GrnBenchmarkDataset:
    name: str
    organism: str
    expression: pd.DataFrame            # rows = cells/samples, cols = genes (normalized)
    gene_names: list[str]
    sample_metadata: pd.DataFrame       # per-row: cell type, batch, ... (index aligns to expression)
    regulator_list: list[str]           # candidate sources (TFs); subset of gene_names
    candidate_edges: pd.DataFrame       # source, target  (source in regulator_list, target in genes, source != target)
    reference_edges: pd.DataFrame       # source, target, is_true  (directed; defined over candidate_edges)
    reference_kind: str                 # "exact" | "chip_seq_proxy" | "curated_prior" | "interventional"
    perturbation_labels: pd.Series | None   # per-row perturbed gene or control/NaN (if available)
    pseudotime: pd.DataFrame | None         # per-row x trajectory pseudotime columns (if available)
    metadata: dict                      # source, license, n_true_edges, density, k-of-interest, ...
```

Contract notes:
- `candidate_edges` is the scored universe (mirrors DREAM4's 9900 non-self edges,
  but TF→gene here). Every scorer must return one score per candidate edge.
- `reference_edges` is a left-join target keyed on (source, target); missing →
  `is_true = 0` (after `appendZeroInteractions`-style densification).
- Scorers reuse the existing `inference` API by mapping
  expression → (X predictors over `regulator_list`, Y target genes). The dynamic
  lagged path is only built when `pseudotime` is present.
- Output stays our existing **candidate directed edge table** (`source, target,
  score, rank, is_true`), so `evaluation.metrics` / `evaluation.topology` and the
  experiment-14 calibration/confidence code run unchanged.

## Recommendation: BEELINE (first modern benchmark)

**Implement BEELINE first.** Reasons it beats staying on DREAM4/GNW:

- **Lowest-friction transfer.** BEELINE is "expression matrix + directed reference
  + AUPR/precision@k/EPR" — the *same shape* as our DREAM4 pipeline. Correlation,
  GENIE3, static sparse, calibrated alpha, fusion, topology, and calibration bins
  map almost 1:1; only the candidate-set (TF→gene) and the proxy-reference framing
  need adapting.
- **Real single-cell data + real biology.** Moves us off synthetic ODE simulation
  to actual scRNA-seq, which is where the field's claims live — directly tests
  whether calibrated confidence + fusion *generalizes* beyond a simulator.
- **Mature & maintained.** Standard datasets, documented format, an established
  evaluation (EPR) and a public algorithm leaderboard to sanity-check against.
- **Honest stress test.** Noisy proxy references and TF-restricted candidates are
  exactly the regime where our "calibration / confidence / density-prior alpha"
  ideas should either help or be shown not to — a meaningful validation, not a
  victory lap.

CausalBench is the better **second** step (interventional/causal, perturbation
labels our confidence framing could later exploit) once the single-cell adapter
exists — but its scale and non-edge-list evaluation make it the wrong first move.

### First dataset choice within BEELINE

1. Validate the adapter + scorers on a **synthetic BoolODE** dataset (exact ground
   truth, tiny) — confirms correlation/GENIE3/sparse/fusion produce a sane
   candidate edge table and that AUPR/EPR/topology run.
2. Then the real modern-data target: a small BEELINE real set (e.g. **mESC** or
   **hESC**) with the **cell-type-specific ChIP-seq** reference, candidate set =
   provided TFs → all genes.

## Exact next coding step

Add `src/stable_grn_inference/data/beeline.py`:

- `load_beeline_dataset(root, name, reference="cell_type_specific") -> GrnBenchmarkDataset`
  reading the BEELINE per-dataset inputs:
  - `ExpressionData.csv` (genes × cells) → transpose to cells × genes; normalize (log1p if raw).
  - `refNetwork.csv` (columns `Gene1, Gene2[, Type]`) → directed `source→target`.
  - optional `PseudoTime.csv` → `pseudotime`.
  - organism TF list → `regulator_list`.
- Build `candidate_edges` = {(tf, g) : tf ∈ regulator_list, g ∈ gene_names, tf ≠ g},
  then left-join `refNetwork` to get `is_true` (densify zeros) → `reference_edges`.
- Add a tiny loader test (synthetic BoolODE, exact ground truth) that asserts the
  candidate/reference shapes and `source ≠ target`. Keep it skippable if the
  dataset files are absent (mirror the Size100 real-data test pattern).

Then a small experiment 16 runs the existing scorers + experiment-14
confidence/calibration on that one dataset (no new models).

## Caveats

- Re-verify BEELINE's current download location, license, and exact dataset list
  at implementation time.
- AUPR on proxy references is *prior agreement*, not truth recovery — always pair
  with EPR and the `reference_kind` label.
- Do not pull DREAM5, chase older GNW, or force GNW simulations (out of scope).
