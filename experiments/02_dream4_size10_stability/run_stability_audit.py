"""Audit stability-aware edge rankings on DREAM4 Size10 multifactorial data."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import load_expression_matrix, load_gold_standard_edges
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_elastic_net,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
)
from stable_grn_inference.stability import (
    generate_resample_indices,
    summarize_resampled_edge_scores,
)


RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_stability_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_stability_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_network1_stability_debug_report.md"

Ranker = Callable[[pd.DataFrame], pd.DataFrame]


def expression_path(network_id: int) -> Path:
    """Return the Size10 multifactorial expression path for one network."""
    return (
        ROOT
        / f"data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_{network_id}/"
        / f"insilico_size10_{network_id}_multifactorial.tsv"
    )


def gold_standard_path(network_id: int) -> Path:
    """Return the matching Size10 gold-standard topology path."""
    return (
        ROOT
        / "data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/"
        / "DREAM4_Challenge2_GoldStandards/Size 10/"
        / f"DREAM4_GoldStandard_InSilico_Size10_{network_id}.tsv"
    )


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join edge scores to DREAM4 gold-standard truth labels and assign ranks."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        missing = scored.loc[scored["is_true"].isna(), ["source", "target"]]
        raise ValueError(f"Predicted edges missing from gold standard: {len(missing)}")

    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True])
    scored = scored.reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def evaluate_scored_edges(scored_edges: pd.DataFrame) -> dict[str, float | int]:
    """Compute edge-recovery metrics for one scored edge table."""
    return {
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": int(scored_edges["is_true"].sum()),
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
    }


def one_shot_methods(random_forest_trees: int) -> list[tuple[str, Ranker, str]]:
    """Return one-shot methods and their score definitions."""
    return [
        ("one_shot_correlation", rank_edges_by_correlation, "absolute correlation"),
        (
            "one_shot_lasso_alpha_0_1",
            lambda expression: rank_edges_by_lasso(expression, alpha=0.1, max_iter=50000),
            "absolute LASSO coefficient, alpha=0.1",
        ),
        (
            "one_shot_elastic_net_alpha_0_03_l1_0_95",
            lambda expression: rank_edges_by_elastic_net(
                expression,
                alpha=0.03,
                l1_ratio=0.95,
                max_iter=50000,
            ),
            "absolute Elastic Net coefficient, alpha=0.03, l1_ratio=0.95",
        ),
        (
            "one_shot_random_forest_importance",
            lambda expression: rank_edges_by_random_forest(
                expression,
                n_estimators=random_forest_trees,
                random_state=17,
            ),
            "random-forest feature importance",
        ),
    ]


def stability_methods(random_forest_trees: int) -> list[dict[str, object]]:
    """Return stability methods and how their stability scores are defined."""
    return [
        {
            "method": "stability_correlation",
            "ranker": rank_edges_by_correlation,
            "score_column": "mean_reciprocal_rank",
            "score_definition": "mean reciprocal rank across resamples",
        },
        {
            "method": "stability_lasso_alpha_0_1",
            "ranker": lambda expression: rank_edges_by_lasso(
                expression,
                alpha=0.1,
                max_iter=50000,
            ),
            "score_column": "selection_frequency",
            "score_definition": "fraction of resamples with nonzero coefficient",
        },
        {
            "method": "stability_elastic_net_alpha_0_03_l1_0_95",
            "ranker": lambda expression: rank_edges_by_elastic_net(
                expression,
                alpha=0.03,
                l1_ratio=0.95,
                max_iter=50000,
            ),
            "score_column": "selection_frequency",
            "score_definition": "fraction of resamples with nonzero coefficient",
        },
        {
            "method": "stability_random_forest_importance",
            "ranker": lambda expression: rank_edges_by_random_forest(
                expression,
                n_estimators=random_forest_trees,
                random_state=23,
            ),
            "score_column": "mean_score",
            "score_definition": "mean feature importance across resamples",
        },
    ]


def run_one_shot_method(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    network_id: int,
    method: str,
    ranker: Ranker,
    score_definition: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run and evaluate one one-shot method."""
    scored_edges = score_edges(ranker(expression), truth_edges)
    metric_row: dict[str, float | int | str] = {
        "row_type": "network",
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": method,
        "variant": "one_shot",
        "score_definition": score_definition,
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
    }
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges


def run_stability_method(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    network_id: int,
    method_config: dict[str, object],
    resample_indices: list,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Run and evaluate one stability-scored method."""
    method = str(method_config["method"])
    ranker = method_config["ranker"]
    score_column = str(method_config["score_column"])
    score_definition = str(method_config["score_definition"])
    if not callable(ranker):
        raise TypeError("ranker must be callable")

    stability_summary = summarize_resampled_edge_scores(
        expression,
        ranker,
        resample_indices,
        top_k=20,
        selection_threshold=0.0,
    )
    predicted = stability_summary[["source", "target", score_column]].rename(
        columns={score_column: "score"}
    )
    scored_edges = score_edges(predicted, truth_edges)

    metric_row: dict[str, float | int | str] = {
        "row_type": "network",
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": method,
        "variant": "stability",
        "score_definition": score_definition,
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
    }
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges, stability_summary


def run_network(
    network_id: int,
    *,
    n_resamples: int,
    resampling_method: str,
    sample_fraction: float,
    random_seed: int,
    random_forest_trees: int,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    """Run all one-shot and stability methods for one Size10 network."""
    expression = load_expression_matrix(expression_path(network_id))
    truth_edges = load_gold_standard_edges(gold_standard_path(network_id))
    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    metric_rows: list[dict[str, float | int | str]] = []

    for method, ranker, score_definition in one_shot_methods(random_forest_trees):
        metric_row, scored_edges = run_one_shot_method(
            expression,
            truth_edges,
            network_id,
            method,
            ranker,
            score_definition,
        )
        metric_rows.append(metric_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)

    indices = generate_resample_indices(
        len(expression),
        n_resamples,
        method=resampling_method,
        sample_fraction=sample_fraction,
        random_seed=random_seed + network_id,
    )
    for method_config in stability_methods(random_forest_trees):
        method = str(method_config["method"])
        metric_row, scored_edges, stability_summary = run_stability_method(
            expression,
            truth_edges,
            network_id,
            method_config,
            indices,
        )
        metric_row["n_resamples"] = n_resamples
        metric_row["resampling_method"] = resampling_method
        metric_rows.append(metric_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)
        edge_audit = merge_stability_columns(edge_audit, stability_summary, method)

    return metric_rows, edge_audit


def merge_score_columns(edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge score and rank columns for one method into the edge audit table."""
    method_scores = scored_edges[["source", "target", "score", "rank"]].rename(
        columns={"score": f"score_{method}", "rank": f"rank_{method}"}
    )
    return edge_audit.merge(method_scores, on=["source", "target"], how="left")


def merge_stability_columns(
    edge_audit: pd.DataFrame,
    stability_summary: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Merge detailed stability summaries into the edge audit table."""
    details = stability_summary.rename(
        columns={
            "mean_score": f"mean_score_{method}",
            "mean_reciprocal_rank": f"mean_reciprocal_rank_{method}",
            "top_k_frequency": f"top20_frequency_{method}",
            "selection_frequency": f"selection_frequency_{method}",
        }
    )
    columns = [
        "source",
        "target",
        f"mean_score_{method}",
        f"mean_reciprocal_rank_{method}",
        f"top20_frequency_{method}",
        f"selection_frequency_{method}",
    ]
    return edge_audit.merge(details[columns], on=["source", "target"], how="left")


def add_mean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Append one mean-metric row per method to the summary table."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    mean_rows = summary.groupby("method", as_index=False)[metric_columns].mean()
    mean_rows.insert(0, "row_type", "mean")
    mean_rows.insert(1, "network_id", "mean")
    mean_rows.insert(2, "network", "mean")
    mean_rows["variant"] = "mean"
    mean_rows["score_definition"] = ""

    for column in [
        "n_samples",
        "n_genes",
        "n_candidate_edges",
        "n_true_edges",
        "n_resamples",
        "resampling_method",
    ]:
        if column in summary.columns:
            mean_rows[column] = pd.NA

    columns = list(summary.columns)
    return pd.concat([summary, mean_rows[columns]], ignore_index=True)


def make_debug_report(edge_audit: pd.DataFrame) -> str:
    """Build a human-readable stability audit report for Size10 network 1."""
    method_explanations = [
        ("one-shot correlation", "absolute correlation from the full multifactorial matrix"),
        ("stability correlation", "mean reciprocal rank across bootstrap resamples"),
        ("one-shot LASSO alpha=0.1", "absolute coefficient from the full matrix"),
        ("stability LASSO alpha=0.1", "fraction of resamples where coefficient magnitude is nonzero"),
        ("one-shot random forest", "feature importance from the full matrix"),
        ("stability random forest", "mean feature importance across bootstrap resamples"),
    ]
    debug_methods = [
        ("one_shot_correlation", "one-shot correlation"),
        ("stability_correlation", "stability correlation"),
        ("one_shot_lasso_alpha_0_1", "one-shot LASSO alpha=0.1"),
        ("stability_lasso_alpha_0_1", "stability LASSO alpha=0.1"),
        ("one_shot_random_forest_importance", "one-shot random forest"),
        ("stability_random_forest_importance", "stability random forest"),
    ]

    lines = [
        "# DREAM4 Size10 Network 1 Stability Debug Report",
        "",
        "This report audits one-shot and resampled edge rankings for `insilico_size10_1`.",
        "",
        "## Stability Score Meanings",
        "",
    ]
    for label, explanation in method_explanations:
        lines.append(f"- {label}: {explanation}.")
    lines.append("")
    lines.append("Elastic Net variants are included in the CSV outputs but omitted from this debug report to keep it short.")
    lines.append("")

    for method, label in debug_methods:
        score_column = f"score_{method}"
        rank_column = f"rank_{method}"
        top_edges = edge_audit.sort_values(rank_column).head(10).copy()
        top_edges["result"] = top_edges["is_true"].map({1: "true_positive", 0: "false_positive"})
        lines.extend(
            [
                f"## Top 10 Edges By {label}",
                "",
                to_markdown_table(
                    top_edges[["source", "target", "is_true", "result", score_column, rank_column]]
                ),
                "",
            ]
        )
    return "\n".join(lines)


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


def print_summary(summary_with_means: pd.DataFrame) -> None:
    """Print mean and per-network metrics."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    mean_rows = summary_with_means[summary_with_means["row_type"] == "mean"]
    network_rows = summary_with_means[summary_with_means["row_type"] == "network"]

    print("DREAM4 Size10 stability audit")
    print()
    print("Mean metrics across networks 1-5")
    print(mean_rows[["method", *metric_columns]].to_string(index=False, float_format=format_float))
    print()
    print("Per-network metrics")
    print(
        network_rows[["network", "method", *metric_columns]].to_string(
            index=False,
            float_format=format_float,
        )
    )
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_edge_audit: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def format_float(value: float) -> str:
    """Format console metric values."""
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the stability audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--resampling-method", choices=["bootstrap", "subsample"], default="bootstrap")
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--random-forest-trees", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run the Size10 stability audit and write summary artifacts."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | int | str]] = []
    edge_tables: list[pd.DataFrame] = []
    for network_id in range(1, 6):
        network_rows, edge_audit = run_network(
            network_id,
            n_resamples=args.n_resamples,
            resampling_method=args.resampling_method,
            sample_fraction=args.sample_fraction,
            random_seed=args.random_seed,
            random_forest_trees=args.random_forest_trees,
        )
        metric_rows.extend(network_rows)
        edge_tables.append(edge_audit)

    summary = pd.DataFrame(metric_rows)
    summary_with_means = add_mean_rows(summary)
    edge_audit_all = pd.concat(edge_tables, ignore_index=True)

    summary_with_means.to_csv(SUMMARY_PATH, index=False)
    edge_audit_all.to_csv(EDGE_AUDIT_PATH, index=False)

    network_one_edges = edge_audit_all[edge_audit_all["network_id"] == 1].copy()
    DEBUG_REPORT_PATH.write_text(make_debug_report(network_one_edges), encoding="utf-8")
    print_summary(summary_with_means)


if __name__ == "__main__":
    main()
