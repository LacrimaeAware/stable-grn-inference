# BEELINE Adapter Smoke

First step of the modern-benchmark direction (experiment 15 chose BEELINE): make
the existing DREAM4-style pipeline ingest BEELINE-format single-cell GRN datasets,
and smoke-test it end-to-end. No new models; only cheap static methods.

## Adapter

`src/stable_grn_inference/data/beeline.py` provides a model-agnostic
`GrnBenchmarkDataset` and `load_beeline_dataset(root, name, reference=..., tf_list=..., log1p=True)`,
plus helpers: `read_beeline_expression`, `read_beeline_reference_edges`,
`build_tf_to_gene_candidate_edges`, `label_candidate_edges`,
`infer_expression_orientation`.

What it does:

- Reads `ExpressionData.csv` and **detects orientation** (BEELINE stores genes in
  rows) → returns expression as **cells x genes**. Applies `log1p` only when the
  values look raw/count-like.
- Reads `refNetwork.csv` as directed `source -> target` edges; drops self-edges
  and edges whose genes are absent from the expression matrix.
- Builds **candidate edges = TF -> gene** when a regulator list is available
  (from `tf_list` or a `TFs.csv` in the folder); otherwise all directed non-self
  gene pairs (less biologically realistic; flagged in metadata).
- **Densifies labels**: every candidate edge gets `is_true` in {0, 1}. Gold labels
  live only in `edge_labels` (for evaluation); the loaders/scorers never read them.
- Carries optional `pseudotime` and (always `None` for BEELINE) `perturbation_labels`,
  and rich `metadata` (reference_kind, densities, orientation, log1p_applied, ...).

## Key BEELINE caveats (vs DREAM4)

- **References are biological proxies**, not exact simulator truth (cell-type /
  non-specific ChIP-seq, or curated functional priors). AUPR here is
  *agreement-with-prior*; report the **early-precision ratio (EPR = precision@n_true
  / true density)** alongside it, and keep the `reference_kind` label.
- **Candidate set is TF -> gene**, not all-pairs — changes precision@k denominators,
  density, and topology semantics (reciprocal edges only among TF–TF pairs).
- **DREAM4 dynamic methods do not transfer directly.** Static single-cell snapshots
  have no time; the lagged LASSO and include-self persistence need time/pseudotime
  (a future pseudotime-ordered lagged path). Correlation, GENIE3/trees, static
  sparse LASSO, and rank fusion transfer most directly.

## Smoke run

No real BEELINE data was present under `data/raw/beeline/`, so the smoke used a
tiny **synthetic** BEELINE-format fixture (8 genes incl. 3 TFs, 80 cells, planted
TF→target edges) to exercise the adapter + scorers end-to-end. Methods: correlation,
GENIE3 (random forest), static LASSO, and equal-weight Borda fusion.

| method | AUROC | AUPR | EPR | P@5 | P@10 |
|---|---:|---:|---:|---:|---:|
| correlation | 1.0000 | 1.0000 | 3.5000 | 1.0000 | 0.6000 |
| genie3_random_forest | 1.0000 | 1.0000 | 3.5000 | 1.0000 | 0.6000 |
| static_lasso | 1.0000 | 1.0000 | 3.5000 | 1.0000 | 0.6000 |
| fusion_borda | 1.0000 | 1.0000 | 3.5000 | 1.0000 | 0.6000 |

(21 candidate edges, 6 true; P@10 caps at 0.6 because only 6 true edges exist.)

Interpretation: the synthetic fixture is intentionally simple, so perfect recovery
is expected. The value of this run is **plumbing validation** — that the adapter
aligns expression orientation, TF→gene candidate edges, and densified labels so the
existing scorers and `evaluation` metrics consume it unchanged. It is **not** a
difficulty or method-ranking claim. Real BEELINE references are noisy and will give
much lower, method-separating scores.

## Running on real BEELINE data

Place a dataset folder under `data/raw/beeline/<name>/` containing at least
`ExpressionData.csv` and `refNetwork.csv` (optional `TFs.csv`, `PseudoTime.csv`).
Then the smoke auto-detects and uses it:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -B experiments\16_beeline_adapter_smoke\run_beeline_adapter_smoke.py
```

For large real datasets the smoke runs correlation only (GENIE3/LASSO are gated to
small gene counts); a full benchmark experiment would restrict scorers to the
TF→gene candidate set and add EPR-focused reporting.

## What transferred / what did not

- **Transferred cleanly:** correlation, GENIE3/tree importance, static sparse LASSO,
  rank fusion (Borda), and the AUROC/AUPR/precision@k evaluation + densified labels.
- **Did not transfer:** dynamic lagged LASSO and include-self persistence (no time);
  wavelet/scattering preprocessing (time-series-specific); topology FFL at scale
  (O(n^3)); all-pairs candidate semantics (now TF→gene).
- **Needs care next:** single-cell preprocessing (normalization, gene/TF filtering),
  EPR reporting, and treating AUPR as prior-agreement.

## Outputs

If the smoke runs, under `results/tables/` (git-ignored):

- `beeline_adapter_smoke_summary.csv`
- `beeline_adapter_smoke_edges.csv`
- `beeline_adapter_smoke_debug_report.md`

## Tests

`tests/test_beeline_data.py` builds a tiny fake BEELINE dataset in a temp dir (no
download) and checks orientation correction, self-edge exclusion, TF restriction,
label correctness, missing-optional-file tolerance, gold-label separation, and that
the output is consumable by the existing correlation scorer + metrics.
