"""Audit lagged DREAM4 Size10 time-series edge ranking."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import (
    build_lagged_samples,
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
    split_trajectories_by_time_reset,
)
from stable_grn_inference.evaluation import (
    aupr,
    auroc,
    precision_at_k,
    topology_metrics_for_cutoff,
)
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_extra_trees,
    rank_edges_by_lagged_lasso,
    rank_edges_by_lagged_random_forest,
    rank_edges_by_lasso,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_lagged_timeseries_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_lagged_timeseries_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_lagged_timeseries_debug_report.md"

SameTimeRanker = Callable[[pd.DataFrame], pd.DataFrame]
LaggedRanker = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]

LASSO_ALPHAS = (0.01, 0.03, 0.1)
TOPOLOGY_CUTOFF = "top_n_true_edges"


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
    """Compute edge and topology metrics for one scored edge table."""
    n_true_edges = int(scored_edges["is_true"].sum())
    metrics: dict[str, float | int] = {
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": n_true_edges,
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
    }
    topology_metrics = topology_metrics_for_cutoff(
        scored_edges,
        cutoff=n_true_edges,
        rank_column="rank",
    )
    for key, value in topology_metrics.items():
        metrics[f"topology_{key}"] = value
    return metrics


def same_time_methods(
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> list[tuple[str, SameTimeRanker, str]]:
    """Return same-time reference methods for time-series rows."""
    return [
        ("same_time_correlation", rank_edges_by_correlation, "absolute same-time correlation"),
        (
            "same_time_lasso_alpha_0_1",
            lambda expression: rank_edges_by_lasso(expression, alpha=0.1, max_iter=50000),
            "same-time target-wise LASSO coefficient, alpha=0.1",
        ),
        (
            "same_time_genie3_random_forest",
            lambda expression: rank_edges_by_genie3_random_forest(
                expression,
                n_estimators=n_estimators,
                random_state=random_seed + 101,
                n_jobs=n_jobs,
            ),
            f"same-time GENIE3-style random forest feature importance, {n_estimators} trees",
        ),
        (
            "same_time_genie3_extra_trees",
            lambda expression: rank_edges_by_genie3_extra_trees(
                expression,
                n_estimators=n_estimators,
                random_state=random_seed + 202,
                n_jobs=n_jobs,
            ),
            f"same-time GENIE3-style Extra Trees feature importance, {n_estimators} trees",
        ),
    ]


def lagged_methods(
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> list[tuple[str, LaggedRanker, str]]:
    """Return lagged methods for adjacent time-pair samples."""
    methods: list[tuple[str, LaggedRanker, str]] = [
        ("lagged_correlation", rank_edges_by_lagged_correlation, "absolute correlation between source(t) and target(t+1)")
    ]
    for alpha in LASSO_ALPHAS:
        method_name = f"lagged_lasso_alpha_{format_alpha(alpha)}"
        methods.append(
            (
                method_name,
                lambda x, y, alpha=alpha: rank_edges_by_lagged_lasso(
                    x,
                    y,
                    alpha=alpha,
                    max_iter=50000,
                ),
                f"lagged target-wise LASSO coefficient, alpha={alpha}",
            )
        )
    methods.extend(
        [
            (
                "lagged_genie3_random_forest",
                lambda x, y: rank_edges_by_lagged_random_forest(
                    x,
                    y,
                    n_estimators=n_estimators,
                    random_state=random_seed + 303,
                    n_jobs=n_jobs,
                ),
                f"lagged target-wise random forest feature importance, {n_estimators} trees",
            ),
            (
                "lagged_genie3_extra_trees",
                lambda x, y: rank_edges_by_lagged_extra_trees(
                    x,
                    y,
                    n_estimators=n_estimators,
                    random_state=random_seed + 404,
                    n_jobs=n_jobs,
                ),
                f"lagged target-wise Extra Trees feature importance, {n_estimators} trees",
            ),
        ]
    )
    return methods


def format_alpha(alpha: float) -> str:
    """Format alpha values for stable method names."""
    return str(alpha).replace(".", "_")


def run_network(
    network_id: int,
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame, dict[str, int]]:
    """Run all same-time and lagged methods for one Size10 time-series network."""
    timeseries_path = dream4_size10_expression_path(DATA_ROOT, network_id, "timeseries")
    same_time_expression = load_expression_matrix(timeseries_path, drop_time=True)
    timeseries_with_time = load_expression_matrix(timeseries_path, drop_time=False)
    trajectories = split_trajectories_by_time_reset(timeseries_with_time)
    x_lagged, y_lagged, lagged_metadata = build_lagged_samples(trajectories)
    truth_edges = load_gold_standard_edges(dream4_size10_gold_standard_path(DATA_ROOT, network_id))

    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    metric_rows: list[dict[str, float | int | str]] = []
    method_seed = random_seed + (100 * network_id)

    for method, ranker, score_definition in same_time_methods(
        n_estimators=n_estimators,
        random_seed=method_seed,
        n_jobs=n_jobs,
    ):
        metric_row, scored_edges = run_same_time_method(
            same_time_expression,
            truth_edges,
            network_id,
            method,
            ranker,
            score_definition,
            len(trajectories),
            len(lagged_metadata),
        )
        metric_rows.append(metric_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)

    for method, ranker, score_definition in lagged_methods(
        n_estimators=n_estimators,
        random_seed=method_seed,
        n_jobs=n_jobs,
    ):
        metric_row, scored_edges = run_lagged_method(
            x_lagged,
            y_lagged,
            truth_edges,
            network_id,
            method,
            ranker,
            score_definition,
            len(trajectories),
            len(lagged_metadata),
        )
        metric_rows.append(metric_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)

    trajectory_info = {
        "network_id": network_id,
        "n_trajectories": len(trajectories),
        "n_timeseries_rows": len(timeseries_with_time),
        "n_lagged_samples": len(lagged_metadata),
    }
    return metric_rows, edge_audit, trajectory_info


def run_same_time_method(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    network_id: int,
    method: str,
    ranker: SameTimeRanker,
    score_definition: str,
    n_trajectories: int,
    n_lagged_samples: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run one same-time reference method."""
    scored_edges = score_edges(ranker(expression), truth_edges)
    metric_row = base_metric_row(
        network_id,
        method,
        "same_time",
        score_definition,
        n_trajectories,
        len(expression),
        n_lagged_samples,
    )
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges


def run_lagged_method(
    x_lagged: pd.DataFrame,
    y_lagged: pd.DataFrame,
    truth_edges: pd.DataFrame,
    network_id: int,
    method: str,
    ranker: LaggedRanker,
    score_definition: str,
    n_trajectories: int,
    n_lagged_samples: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run one lagged method."""
    scored_edges = score_edges(ranker(x_lagged, y_lagged), truth_edges)
    metric_row = base_metric_row(
        network_id,
        method,
        "lagged",
        score_definition,
        n_trajectories,
        len(x_lagged),
        n_lagged_samples,
    )
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges


def base_metric_row(
    network_id: int,
    method: str,
    variant: str,
    score_definition: str,
    n_trajectories: int,
    n_samples_used: int,
    n_lagged_samples: int,
) -> dict[str, int | str]:
    """Create common metric metadata."""
    return {
        "row_type": "network",
        "data_regime": "timeseries",
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": method,
        "variant": variant,
        "score_definition": score_definition,
        "n_trajectories": n_trajectories,
        "n_samples_used": n_samples_used,
        "n_lagged_samples": n_lagged_samples,
    }


def merge_score_columns(edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge score and rank columns for one method into the edge audit table."""
    method_scores = scored_edges[["source", "target", "score", "rank"]].rename(
        columns={"score": f"score_{method}", "rank": f"rank_{method}"}
    )
    return edge_audit.merge(method_scores, on=["source", "target"], how="left")


def aggregate_metrics(network_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return network rows plus mean rows across Size10 networks."""
    metric_columns = [
        column
        for column in network_metrics.columns
        if column
        not in {
            "row_type",
            "data_regime",
            "network_id",
            "network",
            "method",
            "variant",
            "score_definition",
        }
        and pd.api.types.is_numeric_dtype(network_metrics[column])
    ]
    grouped = network_metrics.groupby("method", as_index=False)
    means = grouped[metric_columns].mean()
    stds = grouped[["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]].std().rename(
        columns={
            "auroc": "std_auroc",
            "aupr": "std_aupr",
            "precision_at_5": "std_precision_at_5",
            "precision_at_10": "std_precision_at_10",
            "precision_at_20": "std_precision_at_20",
        }
    )
    counts = grouped.size().rename(columns={"size": "n_networks"})
    mean_rows = means.merge(stds, on="method", how="left").merge(counts, on="method", how="left")
    mean_rows.insert(0, "row_type", "mean")
    mean_rows["data_regime"] = "timeseries"
    mean_rows["network_id"] = pd.NA
    mean_rows["network"] = "mean_across_size10_networks"
    mean_rows["variant"] = mean_rows["method"].map(method_variant)
    mean_rows["score_definition"] = mean_rows["method"].map(method_score_definition)
    return pd.concat([network_metrics, mean_rows], ignore_index=True, sort=False)


def method_variant(method: str) -> str:
    """Return method variant for aggregate rows."""
    return "lagged" if method.startswith("lagged_") else "same_time"


def method_score_definition(method: str) -> str:
    """Return compact score definitions for aggregate rows."""
    definitions = {
        "same_time_correlation": "absolute same-time correlation",
        "same_time_lasso_alpha_0_1": "same-time target-wise LASSO coefficient, alpha=0.1",
        "same_time_genie3_random_forest": "same-time GENIE3-style random forest feature importance",
        "same_time_genie3_extra_trees": "same-time GENIE3-style Extra Trees feature importance",
        "lagged_correlation": "absolute correlation between source(t) and target(t+1)",
        "lagged_genie3_random_forest": "lagged target-wise random forest feature importance",
        "lagged_genie3_extra_trees": "lagged target-wise Extra Trees feature importance",
    }
    for alpha in LASSO_ALPHAS:
        definitions[f"lagged_lasso_alpha_{format_alpha(alpha)}"] = (
            f"lagged target-wise LASSO coefficient, alpha={alpha}"
        )
    return definitions.get(method, "")


def mean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate rows from a summary table."""
    return summary[summary["row_type"] == "mean"].copy()


def make_debug_report(summary: pd.DataFrame, trajectory_info: pd.DataFrame) -> str:
    """Build a human-readable lagged time-series debug report."""
    means = mean_rows(summary)
    best_aupr = best_method(means, "aupr")
    best_auroc = best_method(means, "auroc")
    best_out_hub = best_method(means, "topology_top3_out_hub_overlap")
    best_in_hub = best_method(means, "topology_top3_in_hub_overlap")
    comparisons = comparison_table(means)

    lines = [
        "# DREAM4 Size10 Lagged Time-Series Debug Report",
        "",
        "This audit uses temporal order in the DREAM4 Size10 time-series files. Trajectories are split when `Time` resets, then lagged samples are built only within each trajectory.",
        "",
        "The lagged design scores directed source(t) -> target(t+1) edges. Self-lag edges are excluded to match the directed non-self DREAM4 candidate edge set used in prior experiments.",
        "",
        "## Trajectory Counts",
        "",
        to_markdown_table(trajectory_info),
        "",
        "## Best Mean Edge Metrics",
        "",
        to_markdown_table(pd.DataFrame([best_aupr, best_auroc])[["metric", "method", "value"]]),
        "",
        "## Best Mean Topology Metrics",
        "",
        to_markdown_table(pd.DataFrame([best_out_hub, best_in_hub])[["metric", "method", "value"]]),
        "",
        "## Main Comparisons",
        "",
        to_markdown_table(comparisons),
        "",
        "## Mean Metrics",
        "",
        to_markdown_table(
            means[
                [
                    "method",
                    "variant",
                    "auroc",
                    "aupr",
                    "precision_at_5",
                    "precision_at_10",
                    "precision_at_20",
                    "topology_out_degree_spearman",
                    "topology_in_degree_spearman",
                    "topology_top3_out_hub_overlap",
                    "topology_top3_in_hub_overlap",
                    "topology_reciprocal_false_positive_pair_rate",
                ]
            ].sort_values("aupr", ascending=False)
        ),
        "",
        "## Interpretation",
        "",
        interpret_results(means, comparisons),
        "",
    ]
    return "\n".join(lines)


def best_method(means: pd.DataFrame, metric: str) -> dict[str, float | str]:
    """Return the best aggregate method for one metric."""
    row = means.sort_values([metric, "method"], ascending=[False, True]).iloc[0]
    return {"metric": metric, "method": row["method"], "value": float(row[metric])}


def comparison_table(means: pd.DataFrame) -> pd.DataFrame:
    """Build direct same-time versus lagged comparison rows."""
    indexed = means.set_index("method")
    rows = [
        compare_pair(indexed, "lagged_correlation", "same_time_correlation", "lagged_correlation_vs_same_time"),
        compare_pair(
            indexed,
            best_lagged_lasso_method(means),
            "same_time_lasso_alpha_0_1",
            "best_lagged_lasso_vs_same_time_lasso",
        ),
        compare_pair(
            indexed,
            best_lagged_genie3_method(means),
            best_same_time_genie3_method(means),
            "best_lagged_genie3_vs_best_same_time_genie3",
        ),
    ]
    return pd.DataFrame(rows)


def compare_pair(indexed_means: pd.DataFrame, challenger: str, baseline: str, comparison: str) -> dict[str, float | str]:
    """Compare two aggregate methods."""
    new = indexed_means.loc[challenger]
    base = indexed_means.loc[baseline]
    return {
        "comparison": comparison,
        "challenger": challenger,
        "baseline": baseline,
        "delta_aupr": new["aupr"] - base["aupr"],
        "delta_auroc": new["auroc"] - base["auroc"],
        "delta_top3_out_hub_overlap": new["topology_top3_out_hub_overlap"] - base["topology_top3_out_hub_overlap"],
        "delta_top3_in_hub_overlap": new["topology_top3_in_hub_overlap"] - base["topology_top3_in_hub_overlap"],
        "delta_reciprocal_pair_count": (
            new["topology_reciprocal_pair_count"] - base["topology_reciprocal_pair_count"]
        ),
        "delta_reciprocal_false_positive_pair_count": (
            new["topology_reciprocal_false_positive_pair_count"]
            - base["topology_reciprocal_false_positive_pair_count"]
        ),
        "delta_reciprocal_false_positive_pair_rate": (
            new["topology_reciprocal_false_positive_pair_rate"]
            - base["topology_reciprocal_false_positive_pair_rate"]
        ),
    }


def best_lagged_lasso_method(means: pd.DataFrame) -> str:
    """Return the lagged LASSO method with highest mean AUPR."""
    lasso = means[means["method"].str.startswith("lagged_lasso_")]
    return str(lasso.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]["method"])


def best_lagged_genie3_method(means: pd.DataFrame) -> str:
    """Return the lagged GENIE3-style method with highest mean AUPR."""
    tree = means[means["method"].isin(["lagged_genie3_random_forest", "lagged_genie3_extra_trees"])]
    return str(tree.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]["method"])


def best_same_time_genie3_method(means: pd.DataFrame) -> str:
    """Return the same-time GENIE3-style method with highest mean AUPR."""
    tree = means[means["method"].isin(["same_time_genie3_random_forest", "same_time_genie3_extra_trees"])]
    return str(tree.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]["method"])


def interpret_results(means: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    """Return concise cautious interpretation text."""
    best_temporal = best_method(means[means["variant"] == "lagged"], "aupr")
    lagged_corr = comparisons[comparisons["comparison"] == "lagged_correlation_vs_same_time"].iloc[0]
    lagged_lasso = comparisons[comparisons["comparison"] == "best_lagged_lasso_vs_same_time_lasso"].iloc[0]
    lagged_tree = comparisons[comparisons["comparison"] == "best_lagged_genie3_vs_best_same_time_genie3"].iloc[0]
    lines = [
        f"The strongest lagged method by mean AUPR is `{best_temporal['method']}` with AUPR {best_temporal['value']:.6f}.",
        f"Lagged correlation changes mean AUPR by {lagged_corr['delta_aupr']:.6f} versus same-time correlation.",
        f"The best lagged LASSO changes mean AUPR by {lagged_lasso['delta_aupr']:.6f} versus same-time LASSO alpha=0.1.",
        f"The best lagged GENIE3-style method changes mean AUPR by {lagged_tree['delta_aupr']:.6f} versus the best same-time GENIE3-style method.",
        "Negative reciprocal false-positive count/rate deltas indicate fewer reciprocal-direction mistakes than the same-time reference.",
        "This is a first temporal audit, not a final dynGENIE3 implementation.",
    ]
    return "\n".join(lines)


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without optional packages."""
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
    """Print compact mean metrics."""
    means = mean_rows(summary)
    columns = [
        "method",
        "variant",
        "auroc",
        "aupr",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "topology_top3_out_hub_overlap",
        "topology_top3_in_hub_overlap",
        "topology_reciprocal_false_positive_pair_rate",
    ]
    print("DREAM4 Size10 lagged time-series audit")
    print()
    print(means[columns].sort_values("aupr", ascending=False).to_string(index=False, float_format=format_float))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_edges: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def format_float(value: float) -> str:
    """Format console metric values."""
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    """Run the lagged time-series audit and write result artifacts."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | int | str]] = []
    edge_tables: list[pd.DataFrame] = []
    trajectory_rows: list[dict[str, int]] = []

    for network_id in range(1, 6):
        rows, edge_audit, trajectory_info = run_network(
            network_id,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
        metric_rows.extend(rows)
        edge_tables.append(edge_audit)
        trajectory_rows.append(trajectory_info)

    network_metrics = pd.DataFrame(metric_rows)
    summary = aggregate_metrics(network_metrics)
    edge_audit_all = pd.concat(edge_tables, ignore_index=True)
    trajectory_info = pd.DataFrame(trajectory_rows)

    summary.to_csv(SUMMARY_PATH, index=False)
    edge_audit_all.to_csv(EDGE_AUDIT_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(make_debug_report(summary, trajectory_info), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
