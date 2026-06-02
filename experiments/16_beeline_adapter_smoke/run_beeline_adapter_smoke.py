"""Tiny smoke benchmark for the BEELINE single-cell GRN adapter.

If a real BEELINE dataset is present under ``data/raw/beeline/<name>/`` (with an
``ExpressionData.csv``), the first one found is used. Otherwise a tiny synthetic
BEELINE-format dataset is generated in a temp directory and used, with a clear
message about where to place real data.

Only cheap, static methods are run (correlation always; GENIE3/random-forest and
static LASSO only when the gene count is small; then equal-weight Borda fusion).
DREAM4 dynamic lagged methods are NOT run: BEELINE is static single-cell, so the
lagged/include-self temporal models do not apply without time/pseudotime.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import load_beeline_dataset
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
    rank_fusion,
)

REAL_DATA_ROOT = ROOT / "data/raw/beeline"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "beeline_adapter_smoke_summary.csv"
EDGES_PATH = RESULTS_DIR / "beeline_adapter_smoke_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "beeline_adapter_smoke_debug_report.md"

TREE_GENE_LIMIT = 60  # only run RF / LASSO when small enough for a quick smoke
PRECISION_KS = (5, 10)


# --------------------------------------------------------------------------- #
# Data discovery / synthetic fixture
# --------------------------------------------------------------------------- #
def find_real_dataset(root: Path) -> tuple[Path, str] | None:
    """Return (root, name) of the first BEELINE dataset found, else None."""
    if not root.exists():
        return None
    for expr in sorted(root.glob("*/ExpressionData.csv")):
        return root, expr.parent.name
    return None


def make_synthetic_dataset(directory: Path, *, seed: int = 0) -> tuple[Path, str, list[str]]:
    """Write a tiny synthetic BEELINE dataset (planted TF->target structure)."""
    rng = np.random.default_rng(seed)
    tfs = ["G1", "G2", "G3"]
    targets = ["G4", "G5", "G6", "G7", "G8"]
    genes = tfs + targets
    n_cells = 80
    planted = [("G1", "G4"), ("G1", "G5"), ("G2", "G5"), ("G2", "G6"), ("G3", "G7"), ("G3", "G8")]

    tf_signal = {tf: rng.standard_normal(n_cells) for tf in tfs}
    columns: dict[str, np.ndarray] = {tf: tf_signal[tf].copy() for tf in tfs}
    for target in targets:
        drivers = [s for (s, t) in planted if t == target]
        signal = sum(0.7 * tf_signal[d] for d in drivers) + 1.0 * rng.standard_normal(n_cells)
        columns[target] = signal
    latent = pd.DataFrame(columns, index=[f"C{i+1}" for i in range(n_cells)])[genes]
    # turn into non-negative pseudo-counts so the log1p path is exercised
    counts = np.rint(np.expm1(np.clip(latent - latent.min(), 0, 6))).astype(int)

    base = directory / "syntheticTinyBeeline"
    base.mkdir(parents=True, exist_ok=True)
    counts.T.to_csv(base / "ExpressionData.csv")  # genes x cells (BEELINE orientation)
    pd.DataFrame(planted, columns=["Gene1", "Gene2"]).to_csv(base / "refNetwork.csv", index=False)
    pd.DataFrame({"TF": tfs}).to_csv(base / "TFs.csv", index=False)
    return directory, "syntheticTinyBeeline", tfs


# --------------------------------------------------------------------------- #
# Scoring + evaluation
# --------------------------------------------------------------------------- #
def candidate_scores(dataset, ranked_all: pd.DataFrame) -> pd.DataFrame:
    """Restrict an all-pairs ranking to the candidate edges and densify scores."""
    merged = dataset.edge_labels.merge(ranked_all, on=["source", "target"], how="left")
    merged["score"] = merged["score"].fillna(0.0)
    return merged


def evaluate(scored: pd.DataFrame, *, n_true: int, n_candidate: int) -> dict[str, float]:
    ordered = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    density = n_true / n_candidate if n_candidate else 0.0
    precision_true = precision_at_k(ordered, "is_true", max(n_true, 1))
    metrics = {
        "auroc": auroc(scored["is_true"], scored["score"]) if scored["is_true"].nunique() > 1 else float("nan"),
        "aupr": aupr(scored["is_true"], scored["score"]) if scored["is_true"].nunique() > 1 else float("nan"),
        "early_precision_ratio": (precision_true / density) if density else float("nan"),
    }
    for k in PRECISION_KS:
        metrics[f"precision_at_{k}"] = precision_at_k(ordered, "is_true", k)
    return metrics


def run_methods(dataset) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Run the cheap static methods; return summary rows, per-method scores, skips."""
    n_genes = len(dataset.genes)
    n_true = int(dataset.edge_labels["is_true"].sum())
    n_candidate = len(dataset.candidate_edges)
    rows: list[dict[str, object]] = []
    method_scores: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    method_scores["correlation"] = candidate_scores(dataset, rank_edges_by_correlation(dataset.expression))
    if n_genes <= TREE_GENE_LIMIT:
        method_scores["genie3_random_forest"] = candidate_scores(
            dataset, rank_edges_by_random_forest(dataset.expression, n_estimators=50, random_state=0)
        )
        method_scores["static_lasso"] = candidate_scores(
            dataset, rank_edges_by_lasso(dataset.expression, alpha=0.05)
        )
    else:
        skipped.append(f"genie3_random_forest, static_lasso (n_genes={n_genes} > {TREE_GENE_LIMIT}; smoke runs correlation only)")

    if len(method_scores) >= 2:
        fusion_inputs = [s[["source", "target", "score"]] for s in method_scores.values()]
        fused = rank_fusion(fusion_inputs, method="borda")
        method_scores["fusion_borda"] = dataset.edge_labels.merge(fused, on=["source", "target"], how="left").assign(
            score=lambda d: d["score"].fillna(0.0)
        )

    for method, scored in method_scores.items():
        rows.append({"method": method, **evaluate(scored, n_true=n_true, n_candidate=n_candidate)})
    return pd.DataFrame(rows), method_scores, skipped


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def build_edge_audit(dataset, method_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    audit = dataset.edge_labels.copy()
    for method, scored in method_scores.items():
        audit = audit.merge(
            scored[["source", "target", "score"]].rename(columns={"score": f"score_{method}"}),
            on=["source", "target"], how="left",
        )
    return audit


def build_report(dataset, summary: pd.DataFrame, data_kind: str, skipped: list[str]) -> str:
    m = dataset.metadata
    lines = [
        "# BEELINE Adapter Smoke Report",
        "",
        f"Data: **{data_kind}** dataset `{dataset.name}` "
        f"({m['n_samples']} cells x {m['n_genes']} genes; "
        f"reference_kind={m['reference_kind']}; tf_restricted={m['tf_restricted']}).",
        f"Candidate edges: {m['n_candidate_edges']} (TF->gene where TFs known); "
        f"true edges: {m['n_true_edges']}; true density: {m['true_edge_density']:.4f}; "
        f"log1p_applied={m['log1p_applied']}; orientation={m['expression_orientation']}.",
        "",
        "## Method results (candidate-restricted)",
        "",
        to_markdown_table(summary),
        "",
        "## Notes",
        "",
        "- References are biological proxies, not exact simulator truth, so AUPR is "
        "agreement-with-prior; the early-precision ratio (EPR = precision@n_true / density) "
        "is the more comparable score.",
        "- Candidate edge set is TF->gene when a regulator list is available; otherwise all "
        "directed non-self pairs (less biologically realistic).",
        "- Transferred directly from the DREAM4 pipeline: correlation, GENIE3/tree importance, "
        "static sparse LASSO, and equal-weight rank fusion.",
        "- NOT run here: DREAM4 dynamic lagged LASSO and include-self persistence. BEELINE is "
        "static single-cell; those need time/pseudotime (a future pseudotime-ordered lagged path).",
    ]
    if skipped:
        lines += ["", "Skipped methods: " + "; ".join(skipped) + "."]
    if data_kind == "synthetic":
        lines += [
            "",
            f"No real BEELINE data found under `{REAL_DATA_ROOT.as_posix()}`. This run used a "
            "tiny synthetic BEELINE-format fixture (planted TF->target structure) only to exercise "
            "the adapter + scorers end-to-end. Place a real dataset folder (with ExpressionData.csv, "
            "refNetwork.csv) there to benchmark on real data.",
        ]
    return "\n".join(lines)


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(c) for c in frame.columns]
    rows = [["" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r] for r in frame.to_numpy()]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(r) + " |" for r in rows]])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    found = find_real_dataset(REAL_DATA_ROOT)
    tempdir = None
    if found is not None:
        root, name = found
        data_kind = "real"
        print(f"Using real BEELINE dataset: {root / name}")
        dataset = load_beeline_dataset(root, name)
    else:
        print(f"No real BEELINE data under {REAL_DATA_ROOT.as_posix()} -> using a tiny synthetic fixture.")
        print(f"To benchmark on real data, place a dataset folder there (ExpressionData.csv, refNetwork.csv, optional TFs.csv / PseudoTime.csv).")
        tempdir = tempfile.mkdtemp(prefix="beeline_smoke_")
        root, name, _ = make_synthetic_dataset(Path(tempdir))
        data_kind = "synthetic"
        dataset = load_beeline_dataset(root, name)

    summary, method_scores, skipped = run_methods(dataset)
    summary.insert(0, "data_kind", data_kind)
    summary.insert(1, "dataset", dataset.name)

    summary.to_csv(SUMMARY_PATH, index=False)
    build_edge_audit(dataset, method_scores).to_csv(EDGES_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(build_report(dataset, summary.drop(columns=["data_kind", "dataset"]), data_kind, skipped), encoding="utf-8")

    print(f"\ndata_kind={data_kind}  dataset={dataset.name}  genes={dataset.metadata['n_genes']}  "
          f"cells={dataset.metadata['n_samples']}  candidates={dataset.metadata['n_candidate_edges']}  "
          f"true_edges={dataset.metadata['n_true_edges']}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if skipped:
        print("skipped:", "; ".join(skipped))
    for path in (SUMMARY_PATH, EDGES_PATH, DEBUG_REPORT_PATH):
        print(f"saved: {path.as_posix()}")

    if tempdir is not None:
        import shutil
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    main()
