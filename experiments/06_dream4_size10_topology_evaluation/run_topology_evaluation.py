"""Evaluate topology recovery for DREAM4 Size10 edge rankings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.evaluation import topology_metrics_for_cutoff


RESULTS_DIR = ROOT / "results/tables"
GENIE3_EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_genie3_baseline_edges.csv"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_topology_summary.csv"
DETAILS_PATH = RESULTS_DIR / "dream4_size10_topology_details.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_topology_debug_report.md"

METHODS = [
    "one_shot_correlation",
    "stability_correlation",
    "genie3_random_forest",
    "genie3_extra_trees",
]
PRIMARY_CUTOFF = "top_n_true_edges"


def load_edge_audit(path: Path) -> pd.DataFrame:
    """Load the GENIE3 edge audit used as the topology-evaluation input."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run experiments/04_dream4_genie3_baseline/run_genie3_baseline.py first."
        )
    edge_audit = pd.read_csv(path)
    required = {"data_regime", "network_id", "source", "target", "is_true"}
    missing = required - set(edge_audit.columns)
    if missing:
        raise ValueError(f"Edge audit is missing required columns: {sorted(missing)}")
    return edge_audit


def evaluate_topology(edge_audit: pd.DataFrame) -> pd.DataFrame:
    """Compute topology metrics for each regime, network, method, and cutoff."""
    rows: list[dict[str, float | int | str]] = []
    groups = edge_audit.groupby(["data_regime", "network_id"], sort=True)
    for (data_regime, network_id), network_edges in groups:
        network_edges = network_edges.copy()
        genes = sorted(set(network_edges["source"].astype(str)) | set(network_edges["target"].astype(str)))
        n_true_edges = int(network_edges["is_true"].astype(int).sum())
        cutoffs = [
            ("top5", 5),
            ("top10", 10),
            ("top20", 20),
            ("top_n_true_edges", n_true_edges),
        ]

        for method in METHODS:
            rank_column = f"rank_{method}"
            score_column = f"score_{method}"
            if rank_column not in network_edges.columns:
                continue
            for cutoff_label, cutoff in cutoffs:
                metrics = topology_metrics_for_cutoff(
                    network_edges,
                    cutoff=cutoff,
                    rank_column=rank_column,
                    genes=genes,
                )
                rows.append(
                    {
                        "data_regime": data_regime,
                        "network_id": int(network_id),
                        "method": method,
                        "cutoff": cutoff_label,
                        "numeric_cutoff": cutoff,
                        "score_column": score_column if score_column in network_edges.columns else "",
                        "rank_column": rank_column,
                        "n_genes": len(genes),
                        "n_true_edges": n_true_edges,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_details(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-network topology details into mean summary rows."""
    metric_columns = [
        column
        for column in details.columns
        if column
        not in {
            "data_regime",
            "network_id",
            "method",
            "cutoff",
            "score_column",
            "rank_column",
        }
    ]
    grouped = details.groupby(["data_regime", "method", "cutoff"], as_index=False)
    means = grouped[metric_columns].mean()
    counts = grouped.size().rename(columns={"size": "n_networks"})
    summary = means.merge(counts, on=["data_regime", "method", "cutoff"], how="left")
    summary = summary.rename(columns={column: f"mean_{column}" for column in metric_columns})
    return summary


def make_debug_report(summary: pd.DataFrame) -> str:
    """Build a human-readable topology evaluation report."""
    primary = summary[summary["cutoff"] == PRIMARY_CUTOFF].copy()
    out_hub_best = best_by_metric(primary, "mean_top3_out_hub_overlap")
    in_hub_best = best_by_metric(primary, "mean_top3_in_hub_overlap")
    out_degree_best = best_by_metric(primary, "mean_out_degree_spearman")
    in_degree_best = best_by_metric(primary, "mean_in_degree_spearman")
    stability_deltas = compare_methods(
        primary,
        baseline="one_shot_correlation",
        challenger="stability_correlation",
    )
    genie3_deltas = compare_best_genie3_to_stability(primary)
    reciprocal = reciprocal_problem_table(primary)

    lines = [
        "# DREAM4 Size10 Topology Evaluation Debug Report",
        "",
        "Topology-aware evaluation asks whether top-ranked edges recover graph structure, not only individual true edges.",
        "Here, ranked edges are thresholded into predicted directed graphs at top 5, top 10, top 20, and top N true edges.",
        "",
        "Edge AUROC/AUPR are useful, but they can miss structural failures: a method may recover many true edges while missing regulators, hubs, directionality, or motifs.",
        "",
        f"Primary interpretation below uses `{PRIMARY_CUTOFF}`, which gives each predicted graph roughly the gold-standard edge density for that network.",
        "",
        "## Best Top-3 Out-Hub Recovery",
        "",
        to_markdown_table(out_hub_best[["data_regime", "method", "mean_top3_out_hub_overlap", "mean_out_degree_spearman"]]),
        "",
        "## Best Top-3 In-Hub Recovery",
        "",
        to_markdown_table(in_hub_best[["data_regime", "method", "mean_top3_in_hub_overlap", "mean_in_degree_spearman"]]),
        "",
        "## Best Degree Spearman Recovery",
        "",
        "### Out-Degree",
        "",
        to_markdown_table(out_degree_best[["data_regime", "method", "mean_out_degree_spearman", "mean_top3_out_hub_overlap"]]),
        "",
        "### In-Degree",
        "",
        to_markdown_table(in_degree_best[["data_regime", "method", "mean_in_degree_spearman", "mean_top3_in_hub_overlap"]]),
        "",
        "## Stability Correlation Versus One-Shot Correlation",
        "",
        to_markdown_table(stability_deltas),
        "",
        "## Best GENIE3 Variant Versus Stability Correlation",
        "",
        to_markdown_table(genie3_deltas),
        "",
        "## Reciprocal-Direction False Positives",
        "",
        to_markdown_table(reciprocal),
        "",
        "## Interpretation",
        "",
        interpret_results(out_hub_best, in_hub_best, stability_deltas, genie3_deltas, reciprocal),
        "",
    ]
    return "\n".join(lines)


def best_by_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the best method per data regime for one metric."""
    rows = []
    for data_regime, regime_summary in summary.groupby("data_regime", sort=True):
        sorted_regime = regime_summary.sort_values(
            [metric, "mean_edge_precision_at_k", "method"],
            ascending=[False, False, True],
        )
        rows.append(sorted_regime.iloc[0])
    return pd.DataFrame(rows)


def compare_methods(summary: pd.DataFrame, *, baseline: str, challenger: str) -> pd.DataFrame:
    """Return topology metric deltas between two methods by data regime."""
    rows: list[dict[str, float | str | bool]] = []
    for data_regime, regime_summary in summary.groupby("data_regime", sort=True):
        indexed = regime_summary.set_index("method")
        if baseline not in indexed.index or challenger not in indexed.index:
            continue
        base = indexed.loc[baseline]
        new = indexed.loc[challenger]
        rows.append(
            {
                "data_regime": data_regime,
                "delta_edge_precision": new["mean_edge_precision_at_k"] - base["mean_edge_precision_at_k"],
                "delta_out_degree_spearman": new["mean_out_degree_spearman"] - base["mean_out_degree_spearman"],
                "delta_in_degree_spearman": new["mean_in_degree_spearman"] - base["mean_in_degree_spearman"],
                "delta_top3_out_hub_overlap": new["mean_top3_out_hub_overlap"] - base["mean_top3_out_hub_overlap"],
                "delta_top3_in_hub_overlap": new["mean_top3_in_hub_overlap"] - base["mean_top3_in_hub_overlap"],
                "delta_reciprocal_false_positive_pair_rate": (
                    new["mean_reciprocal_false_positive_pair_rate"]
                    - base["mean_reciprocal_false_positive_pair_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_best_genie3_to_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare the best GENIE3 topology row with stability correlation."""
    rows: list[dict[str, float | str | bool]] = []
    for data_regime, regime_summary in summary.groupby("data_regime", sort=True):
        stability = regime_summary[regime_summary["method"] == "stability_correlation"]
        genie3 = regime_summary[regime_summary["method"].str.startswith("genie3_")]
        if stability.empty or genie3.empty:
            continue
        best_genie3 = genie3.sort_values(
            ["mean_top3_out_hub_overlap", "mean_top3_in_hub_overlap", "mean_edge_precision_at_k"],
            ascending=[False, False, False],
        ).iloc[0]
        stability_row = stability.iloc[0]
        rows.append(
            {
                "data_regime": data_regime,
                "best_genie3_topology_method": best_genie3["method"],
                "delta_edge_precision": best_genie3["mean_edge_precision_at_k"] - stability_row["mean_edge_precision_at_k"],
                "delta_out_degree_spearman": best_genie3["mean_out_degree_spearman"] - stability_row["mean_out_degree_spearman"],
                "delta_in_degree_spearman": best_genie3["mean_in_degree_spearman"] - stability_row["mean_in_degree_spearman"],
                "delta_top3_out_hub_overlap": (
                    best_genie3["mean_top3_out_hub_overlap"] - stability_row["mean_top3_out_hub_overlap"]
                ),
                "delta_top3_in_hub_overlap": (
                    best_genie3["mean_top3_in_hub_overlap"] - stability_row["mean_top3_in_hub_overlap"]
                ),
                "genie3_improves_out_hubs": (
                    best_genie3["mean_top3_out_hub_overlap"] > stability_row["mean_top3_out_hub_overlap"]
                ),
                "genie3_improves_in_hubs": (
                    best_genie3["mean_top3_in_hub_overlap"] > stability_row["mean_top3_in_hub_overlap"]
                ),
            }
        )
    return pd.DataFrame(rows)


def reciprocal_problem_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return reciprocal false-positive rates by method and regime."""
    columns = [
        "data_regime",
        "method",
        "mean_reciprocal_pair_count",
        "mean_reciprocal_false_positive_pair_count",
        "mean_reciprocal_false_positive_pair_rate",
    ]
    return summary[columns].sort_values(["data_regime", "mean_reciprocal_false_positive_pair_rate", "method"], ascending=[True, False, True])


def interpret_results(
    out_hub_best: pd.DataFrame,
    in_hub_best: pd.DataFrame,
    stability_deltas: pd.DataFrame,
    genie3_deltas: pd.DataFrame,
    reciprocal: pd.DataFrame,
) -> str:
    """Return concise interpretation for the debug report."""
    out_winners = method_counts(out_hub_best)
    in_winners = method_counts(in_hub_best)
    stability_out_improvements = int((stability_deltas["delta_top3_out_hub_overlap"] > 0).sum())
    stability_in_improvements = int((stability_deltas["delta_top3_in_hub_overlap"] > 0).sum())
    genie3_out_improvements = int(genie3_deltas["genie3_improves_out_hubs"].sum())
    genie3_in_improvements = int(genie3_deltas["genie3_improves_in_hubs"].sum())
    correlation_reciprocal = reciprocal[reciprocal["method"] == "one_shot_correlation"]
    mean_correlation_reciprocal_rate = correlation_reciprocal[
        "mean_reciprocal_false_positive_pair_rate"
    ].mean()

    lines = [
        f"Top-3 out-hub wins by method: {out_winners}.",
        f"Top-3 in-hub wins by method: {in_winners}.",
        f"Stability correlation improves top-3 out-hub overlap in {stability_out_improvements}/4 regimes and top-3 in-hub overlap in {stability_in_improvements}/4 regimes versus one-shot correlation.",
        f"The best GENIE3 topology variant improves top-3 out-hub overlap over stability correlation in {genie3_out_improvements}/4 regimes and top-3 in-hub overlap in {genie3_in_improvements}/4 regimes.",
        f"One-shot correlation's mean reciprocal false-positive pair rate at this cutoff is {mean_correlation_reciprocal_rate:.3f}, reflecting the direction-symmetry risk of correlation scores.",
        "These topology metrics are an audit layer, not a final hidden-structure result.",
    ]
    return "\n".join(lines)


def method_counts(frame: pd.DataFrame) -> str:
    """Format method winner counts for report prose."""
    counts = frame["method"].value_counts().sort_index()
    return ", ".join(f"{method}: {count}" for method, count in counts.items())


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional packages."""
    columns = [str(column) for column in frame.columns]
    rows = [[format_cell(value) for value in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_cell(value: object) -> str:
    """Format a Markdown table cell."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_summary(summary: pd.DataFrame) -> None:
    """Print a compact summary for the primary cutoff."""
    primary = summary[summary["cutoff"] == PRIMARY_CUTOFF].copy()
    columns = [
        "data_regime",
        "method",
        "mean_edge_precision_at_k",
        "mean_out_degree_spearman",
        "mean_in_degree_spearman",
        "mean_top3_out_hub_overlap",
        "mean_top3_in_hub_overlap",
        "mean_reciprocal_false_positive_pair_rate",
    ]
    print("DREAM4 Size10 topology evaluation")
    print()
    print(f"primary_cutoff: {PRIMARY_CUTOFF}")
    print(primary[columns].to_string(index=False, float_format=format_float))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_details: {DETAILS_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def format_float(value: float) -> str:
    """Format console metric values."""
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-audit", type=Path, default=GENIE3_EDGE_AUDIT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run topology evaluation and write summary artifacts."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    edge_audit = load_edge_audit(args.edge_audit)
    details = evaluate_topology(edge_audit)
    summary = aggregate_details(details)

    details.to_csv(DETAILS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(make_debug_report(summary), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
