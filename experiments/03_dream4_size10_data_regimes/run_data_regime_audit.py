"""Audit DREAM4 Size10 one-shot and stability rankings across data regimes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import (
    SIZE10_DATA_REGIMES,
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
)
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
)
from stable_grn_inference.stability import (
    generate_resample_indices,
    summarize_resampled_edge_scores,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_data_regime_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_data_regime_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_data_regime_debug_report.md"

Ranker = Callable[[pd.DataFrame], pd.DataFrame]


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
    """Return one-shot methods and score definitions."""
    return [
        ("one_shot_correlation", rank_edges_by_correlation, "absolute correlation"),
        (
            "one_shot_lasso_alpha_0_1",
            lambda expression: rank_edges_by_lasso(expression, alpha=0.1, max_iter=50000),
            "absolute LASSO coefficient, alpha=0.1",
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


def available_regimes() -> list[str]:
    """Return Size10 data regimes that are present for all five networks."""
    regimes: list[str] = []
    for data_regime in SIZE10_DATA_REGIMES:
        paths = [
            dream4_size10_expression_path(DATA_ROOT, network_id, data_regime)
            for network_id in range(1, 6)
        ]
        if all(path.exists() for path in paths):
            regimes.append(data_regime)
    return regimes


def run_one_shot_method(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    data_regime: str,
    network_id: int,
    method: str,
    ranker: Ranker,
    score_definition: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run and evaluate one one-shot method."""
    scored_edges = score_edges(ranker(expression), truth_edges)
    metric_row: dict[str, float | int | str] = {
        "data_regime": data_regime,
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
    data_regime: str,
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
        "data_regime": data_regime,
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


def run_regime_network(
    data_regime: str,
    network_id: int,
    *,
    n_resamples: int,
    resampling_method: str,
    sample_fraction: float,
    random_seed: int,
    random_forest_trees: int,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    """Run all methods for one data regime and one Size10 network."""
    expression = load_expression_matrix(
        dream4_size10_expression_path(DATA_ROOT, network_id, data_regime),
        drop_time=True,
    )
    truth_edges = load_gold_standard_edges(dream4_size10_gold_standard_path(DATA_ROOT, network_id))
    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    edge_audit.insert(0, "data_regime", data_regime)
    metric_rows: list[dict[str, float | int | str]] = []

    for method, ranker, score_definition in one_shot_methods(random_forest_trees):
        metric_row, scored_edges = run_one_shot_method(
            expression,
            truth_edges,
            data_regime,
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
        random_seed=random_seed + (100 * network_id) + regime_seed_offset(data_regime),
    )
    for method_config in stability_methods(random_forest_trees):
        method = str(method_config["method"])
        metric_row, scored_edges, stability_summary = run_stability_method(
            expression,
            truth_edges,
            data_regime,
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


def regime_seed_offset(data_regime: str) -> int:
    """Return a deterministic seed offset for a data regime."""
    return SIZE10_DATA_REGIMES.index(data_regime) * 1000


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


def aggregate_metrics(metric_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation metrics by regime and method."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    grouped = metric_rows.groupby(["data_regime", "method"], as_index=False)
    mean_metrics = grouped[metric_columns].mean()
    std_metrics = grouped[metric_columns].std().rename(
        columns={column: f"std_{column}" for column in metric_columns}
    )
    mean_metrics = mean_metrics.rename(columns={column: f"mean_{column}" for column in metric_columns})
    summary = mean_metrics.merge(std_metrics, on=["data_regime", "method"], how="left")
    counts = grouped.size().rename(columns={"size": "n_networks"})
    return summary.merge(counts, on=["data_regime", "method"], how="left")


def make_debug_report(summary: pd.DataFrame, edge_audit: pd.DataFrame) -> str:
    """Build a human-readable data-regime audit report."""
    best_aupr = best_methods_by_metric(summary, "mean_aupr")
    best_auroc = best_methods_by_metric(summary, "mean_auroc")
    improvements = stability_improvements(summary)
    debug_regimes = choose_debug_regimes(improvements)

    lines = [
        "# DREAM4 Size10 Data-Regime Debug Report",
        "",
        "This report compares one-shot and bootstrap-stability rankings across Size10 data regimes.",
        "",
        "Time-series rows are treated as same-time expression observations after dropping `Time`; no lagged inference is used here.",
        "",
        "## Best Method By Mean AUPR",
        "",
        to_markdown_table(best_aupr[["data_regime", "method", "mean_aupr"]]),
        "",
        "## Best Method By Mean AUROC",
        "",
        to_markdown_table(best_auroc[["data_regime", "method", "mean_auroc"]]),
        "",
        "## Stability Improvements",
        "",
        to_markdown_table(improvements),
        "",
    ]

    for data_regime in debug_regimes:
        regime_edges = edge_audit[
            (edge_audit["data_regime"] == data_regime)
            & (edge_audit["network_id"] == 1)
        ].copy()
        lines.extend(
            [
                f"## Network 1 Top Edges: {data_regime}",
                "",
            ]
        )
        for method in [
            "one_shot_correlation",
            "stability_correlation",
            "one_shot_lasso_alpha_0_1",
            "stability_lasso_alpha_0_1",
            "one_shot_random_forest_importance",
            "stability_random_forest_importance",
        ]:
            score_column = f"score_{method}"
            rank_column = f"rank_{method}"
            top_edges = regime_edges.sort_values(rank_column).head(10).copy()
            top_edges["result"] = top_edges["is_true"].map({1: "true_positive", 0: "false_positive"})
            lines.extend(
                [
                    f"### Top 10 By {method}",
                    "",
                    to_markdown_table(
                        top_edges[["source", "target", "is_true", "result", score_column, rank_column]]
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def best_methods_by_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the best method per data regime for one aggregate metric."""
    idx = summary.groupby("data_regime")[metric].idxmax()
    return summary.loc[idx, ["data_regime", "method", metric]].sort_values("data_regime")


def stability_improvements(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize one-shot versus stability deltas for correlation and LASSO."""
    rows: list[dict[str, float | str | bool]] = []
    for data_regime in sorted(summary["data_regime"].unique()):
        regime_summary = summary[summary["data_regime"] == data_regime].set_index("method")
        for family, one_shot, stability in [
            ("correlation", "one_shot_correlation", "stability_correlation"),
            ("lasso_alpha_0_1", "one_shot_lasso_alpha_0_1", "stability_lasso_alpha_0_1"),
        ]:
            delta_aupr = regime_summary.loc[stability, "mean_aupr"] - regime_summary.loc[one_shot, "mean_aupr"]
            delta_auroc = regime_summary.loc[stability, "mean_auroc"] - regime_summary.loc[one_shot, "mean_auroc"]
            rows.append(
                {
                    "data_regime": data_regime,
                    "family": family,
                    "delta_mean_aupr": delta_aupr,
                    "delta_mean_auroc": delta_auroc,
                    "improved_aupr": delta_aupr > 0,
                    "improved_auroc": delta_auroc > 0,
                }
            )
    return pd.DataFrame(rows)


def choose_debug_regimes(improvements: pd.DataFrame) -> list[str]:
    """Pick multifactorial plus one informative non-multifactorial regime."""
    regimes = ["multifactorial"]
    correlation_rows = improvements[
        (improvements["family"] == "correlation")
        & (improvements["data_regime"] != "multifactorial")
    ].copy()
    if not correlation_rows.empty:
        correlation_rows["abs_delta_mean_aupr"] = correlation_rows["delta_mean_aupr"].abs()
        regimes.append(str(correlation_rows.sort_values("abs_delta_mean_aupr", ascending=False).iloc[0]["data_regime"]))
    return regimes


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
    """Print a compact aggregate summary."""
    metric_columns = ["mean_auroc", "mean_aupr", "mean_precision_at_5", "mean_precision_at_10", "mean_precision_at_20"]
    print("DREAM4 Size10 data-regime audit")
    print()
    print(summary[["data_regime", "method", *metric_columns]].to_string(index=False, float_format=format_float))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_edge_audit: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def format_float(value: float) -> str:
    """Format console metric values."""
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the data-regime audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--resampling-method", choices=["bootstrap", "subsample"], default="bootstrap")
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--random-forest-trees", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run the Size10 data-regime audit and write summary artifacts."""
    args = parse_args()
    regimes = available_regimes()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | int | str]] = []
    edge_tables: list[pd.DataFrame] = []
    for data_regime in regimes:
        for network_id in range(1, 6):
            rows, edge_audit = run_regime_network(
                data_regime,
                network_id,
                n_resamples=args.n_resamples,
                resampling_method=args.resampling_method,
                sample_fraction=args.sample_fraction,
                random_seed=args.random_seed,
                random_forest_trees=args.random_forest_trees,
            )
            metric_rows.extend(rows)
            edge_tables.append(edge_audit)

    network_metrics = pd.DataFrame(metric_rows)
    summary = aggregate_metrics(network_metrics)
    edge_audit_all = pd.concat(edge_tables, ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    edge_audit_all.to_csv(EDGE_AUDIT_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(make_debug_report(summary, edge_audit_all), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
