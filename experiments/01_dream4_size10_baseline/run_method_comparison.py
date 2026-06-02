"""Compare DREAM4 Size10 multifactorial edge-ranking baselines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

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


RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_method_comparison_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_method_comparison_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_network1_debug_report.md"

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


def method_grid() -> list[tuple[str, Ranker]]:
    """Return the baseline methods for the audit comparison."""
    methods: list[tuple[str, Ranker]] = [("correlation", rank_edges_by_correlation)]

    for alpha in [0.001, 0.003, 0.01, 0.03, 0.1]:
        methods.append(
            (
                f"lasso_alpha_{format_number(alpha)}",
                lambda expression, alpha=alpha: rank_edges_by_lasso(
                    expression,
                    alpha=alpha,
                    max_iter=50000,
                ),
            )
        )

    for alpha in [0.003, 0.01, 0.03]:
        for l1_ratio in [0.3, 0.7, 0.95]:
            methods.append(
                (
                    f"elastic_net_alpha_{format_number(alpha)}_l1_{format_number(l1_ratio)}",
                    lambda expression, alpha=alpha, l1_ratio=l1_ratio: rank_edges_by_elastic_net(
                        expression,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=50000,
                    ),
                )
            )

    methods.append(
        (
            "random_forest_importance",
            lambda expression: rank_edges_by_random_forest(
                expression,
                n_estimators=100,
                random_state=17,
            ),
        )
    )
    return methods


def format_number(value: float) -> str:
    """Format a grid value for a compact method name."""
    return str(value).replace(".", "_")


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join ranked edge scores to DREAM4 gold-standard truth labels."""
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


def run_network_methods(
    network_id: int,
    methods: list[tuple[str, Ranker]],
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    """Run all methods for one Size10 network."""
    expression = load_expression_matrix(expression_path(network_id))
    truth_edges = load_gold_standard_edges(gold_standard_path(network_id))

    metric_rows: list[dict[str, float | int | str]] = []
    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)

    for method_name, ranker in methods:
        scored_edges = score_edges(ranker(expression), truth_edges)
        metric_row: dict[str, float | int | str] = {
            "row_type": "network",
            "network_id": network_id,
            "network": f"insilico_size10_{network_id}",
            "method": method_name,
            "n_samples": len(expression),
            "n_genes": expression.shape[1],
        }
        metric_row.update(evaluate_scored_edges(scored_edges))
        metric_rows.append(metric_row)

        method_scores = scored_edges[["source", "target", "score", "rank"]].rename(
            columns={"score": f"score_{method_name}", "rank": f"rank_{method_name}"}
        )
        edge_audit = edge_audit.merge(method_scores, on=["source", "target"], how="left")

    return metric_rows, edge_audit


def add_mean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Append one mean-metric row per method to the summary table."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    mean_rows = summary.groupby("method", as_index=False)[metric_columns].mean()
    mean_rows.insert(0, "row_type", "mean")
    mean_rows.insert(1, "network_id", "mean")
    mean_rows.insert(2, "network", "mean")

    for column in ["n_samples", "n_genes", "n_candidate_edges", "n_true_edges"]:
        mean_rows[column] = pd.NA

    columns = list(summary.columns)
    return pd.concat([summary, mean_rows[columns]], ignore_index=True)


def make_debug_report(
    network_id: int,
    methods: list[tuple[str, Ranker]],
    edge_audit: pd.DataFrame,
) -> str:
    """Build a human-readable audit report for one network."""
    expression = load_expression_matrix(expression_path(network_id))
    lines: list[str] = [
        "# DREAM4 Size10 Network 1 Debug Report",
        "",
        "This report audits edge rankings for `insilico_size10_1` using multifactorial expression data.",
        "",
        "## First Candidate Edges",
        "",
        to_markdown_table(edge_audit[["network_id", "source", "target", "is_true"]].head(10)),
        "",
    ]

    for method_name, _ranker in methods:
        score_column = f"score_{method_name}"
        rank_column = f"rank_{method_name}"
        top_edges = edge_audit.sort_values(rank_column).head(10).copy()
        top_edges["result"] = top_edges["is_true"].map({1: "true_positive", 0: "false_positive"})
        lines.extend(
            [
                f"## Top 10 Edges By {method_name}",
                "",
                to_markdown_table(
                    top_edges[["source", "target", "is_true", "result", score_column, rank_column]]
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Worked LASSO Example",
            "",
            "Target gene: `G1`",
            "",
            "Model: LASSO with `alpha=0.01`; predictors are all other genes. The fitted coefficient magnitude becomes the directed edge score from predictor gene to target gene.",
            "",
            to_markdown_table(lasso_worked_example(expression, target="G1", alpha=0.01)),
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


def lasso_worked_example(expression: pd.DataFrame, *, target: str, alpha: float) -> pd.DataFrame:
    """Fit one target-wise LASSO and return coefficient-to-edge-score details."""
    sources = [gene for gene in expression.columns if gene != target]
    x = expression[sources].to_numpy(dtype=float)
    y = expression[target].to_numpy(dtype=float)

    x_scaled = standardize_columns(x)
    y_scaled = standardize_vector(y)
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
    model.fit(x_scaled, y_scaled)

    return pd.DataFrame(
        {
            "source": sources,
            "target": target,
            "coefficient": model.coef_,
            "edge_score": np.abs(model.coef_),
        }
    ).sort_values("edge_score", ascending=False)


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Standardize matrix columns while tolerating constant columns."""
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (values - values.mean(axis=0)) / scale


def standardize_vector(values: np.ndarray) -> np.ndarray:
    """Standardize a vector while tolerating constant targets."""
    scale = values.std()
    if scale == 0.0:
        scale = 1.0
    return (values - values.mean()) / scale


def print_summary(summary_with_means: pd.DataFrame) -> None:
    """Print mean and per-network metrics."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    mean_rows = summary_with_means[summary_with_means["row_type"] == "mean"]
    network_rows = summary_with_means[summary_with_means["row_type"] == "network"]

    print("DREAM4 Size10 method comparison")
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


def main() -> None:
    """Run all method comparisons and write audit artifacts."""
    methods = method_grid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | int | str]] = []
    edge_tables: list[pd.DataFrame] = []
    for network_id in range(1, 6):
        network_rows, edge_audit = run_network_methods(network_id, methods)
        metric_rows.extend(network_rows)
        edge_tables.append(edge_audit)

    summary = pd.DataFrame(metric_rows)
    summary_with_means = add_mean_rows(summary)
    edge_audit_all = pd.concat(edge_tables, ignore_index=True)

    summary_with_means.to_csv(SUMMARY_PATH, index=False)
    edge_audit_all.to_csv(EDGE_AUDIT_PATH, index=False)

    network_one_edges = edge_audit_all[edge_audit_all["network_id"] == 1].copy()
    DEBUG_REPORT_PATH.write_text(
        make_debug_report(1, methods, network_one_edges),
        encoding="utf-8",
    )

    print_summary(summary_with_means)


if __name__ == "__main__":
    main()
