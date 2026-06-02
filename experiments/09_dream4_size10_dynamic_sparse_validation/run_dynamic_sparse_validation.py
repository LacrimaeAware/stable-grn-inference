"""Validate the strongest DREAM4 Size10 dynamic sparse-linear result."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import (
    build_dynamic_target,
    build_lagged_samples,
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
    split_trajectories_by_time_reset,
    trajectory_bootstrap_indices,
)
from stable_grn_inference.evaluation import (
    aggregate_per_network_metrics,
    aupr,
    auroc,
    precision_at_k,
    topology_metrics_for_cutoff,
)
from stable_grn_inference.inference import (
    build_dynamic_sparse_linear_grid,
    fit_dynamic_linear_coefficients,
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_extra_trees,
    rank_edges_by_lagged_random_forest,
    summarize_resampled_dynamic_linear_coefficients,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_dynamic_sparse_validation_summary.csv"
PER_NETWORK_PATH = RESULTS_DIR / "dream4_size10_dynamic_sparse_validation_per_network.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_dynamic_sparse_validation_edges.csv"
TOPOLOGY_PATH = RESULTS_DIR / "dream4_size10_dynamic_sparse_validation_topology.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_dynamic_sparse_validation_debug_report.md"

LASSO_ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ELASTIC_NET_ALPHAS = (0.01, 0.03, 0.1)
ELASTIC_NET_L1_RATIOS = (0.3, 0.7, 0.95)
TARGET_TYPES = ("level", "delta")


def load_network_data(network_id: int) -> dict[str, object]:
    """Load one Size10 time-series network and build lagged targets."""
    timeseries = load_expression_matrix(
        dream4_size10_expression_path(DATA_ROOT, network_id, "timeseries"),
        drop_time=False,
    )
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, metadata = build_lagged_samples(trajectories)
    targets = {
        target_type: build_dynamic_target(x_t, y_t1, metadata, target_type=target_type)
        for target_type in TARGET_TYPES
    }
    truth_edges = load_gold_standard_edges(dream4_size10_gold_standard_path(DATA_ROOT, network_id))
    return {
        "network_id": network_id,
        "timeseries": timeseries,
        "trajectories": trajectories,
        "x_t": x_t,
        "y_t1": y_t1,
        "metadata": metadata,
        "targets": targets,
        "truth_edges": truth_edges,
    }


def run_sparse_grid_for_network(
    network_data: dict[str, object],
    grid: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]], dict[str, pd.DataFrame]]:
    """Run all sparse-linear grid configurations for one network."""
    network_id = int(network_data["network_id"])
    x_t = network_data["x_t"]
    targets = network_data["targets"]
    truth_edges = network_data["truth_edges"]
    metadata = network_data["metadata"]
    trajectories = network_data["trajectories"]
    if not isinstance(x_t, pd.DataFrame) or not isinstance(targets, dict):
        raise TypeError("invalid network_data")
    if not isinstance(truth_edges, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
        raise TypeError("invalid network_data")
    if not isinstance(trajectories, list):
        raise TypeError("invalid network_data")

    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    scored_by_method: dict[str, pd.DataFrame] = {}

    for config in grid.to_dict("records"):
        target = targets[str(config["target_type"])]
        if not isinstance(target, pd.DataFrame):
            raise TypeError("target must be a DataFrame")
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            predicted_edges, self_coefficients = fit_dynamic_linear_coefficients(
                x_t,
                target,
                model_kind=str(config["model_kind"]),
                alpha=float(config["alpha"]),
                l1_ratio=float(config["l1_ratio"]) if pd.notna(config["l1_ratio"]) else None,
                self_predictor_mode=str(config["self_predictor_mode"]),
                max_iter=50000,
            )
        fit_seconds = time.perf_counter() - start
        method = str(config["method"])
        scored_edges = score_edges(predicted_edges, truth_edges)
        scored_by_method[method] = scored_edges
        diagnostics = persistence_diagnostics(scored_edges, self_coefficients)
        row, topology = evaluate_method(
            scored_edges,
            network_id=network_id,
            method=method,
            method_family="sparse_linear",
            variant="one_shot_coefficient_magnitude",
            model_kind=str(config["model_kind"]),
            target_type=str(config["target_type"]),
            self_predictor_mode=str(config["self_predictor_mode"]),
            alpha=float(config["alpha"]),
            l1_ratio=float(config["l1_ratio"]) if pd.notna(config["l1_ratio"]) else pd.NA,
            score_variant="coefficient_magnitude",
            n_trajectories=len(trajectories),
            n_lagged_samples=len(metadata),
            n_resamples=0,
            fit_seconds=fit_seconds,
            extra_metrics=diagnostics,
        )
        metric_rows.append(row)
        topology_rows.append(topology)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)
        edge_audit = merge_self_columns(edge_audit, self_coefficients, method)

    return metric_rows, edge_audit, topology_rows, scored_by_method


def run_reference_methods_for_network(
    network_data: dict[str, object],
    edge_audit: pd.DataFrame,
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    """Run lagged correlation and GENIE3-style tree references."""
    network_id = int(network_data["network_id"])
    x_t = network_data["x_t"]
    y_t1 = network_data["y_t1"]
    truth_edges = network_data["truth_edges"]
    metadata = network_data["metadata"]
    trajectories = network_data["trajectories"]
    if not isinstance(x_t, pd.DataFrame) or not isinstance(y_t1, pd.DataFrame):
        raise TypeError("invalid network_data")
    if not isinstance(truth_edges, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
        raise TypeError("invalid network_data")
    if not isinstance(trajectories, list):
        raise TypeError("invalid network_data")

    method_configs = [
        {
            "method": "lagged_correlation_reference",
            "method_family": "correlation_reference",
            "ranker": lambda: rank_edges_by_lagged_correlation(x_t, y_t1),
            "seed_offset": 0,
        },
        {
            "method": "lagged_genie3_random_forest",
            "method_family": "tree_reference",
            "ranker": lambda: rank_edges_by_lagged_random_forest(
                x_t,
                y_t1,
                n_estimators=n_estimators,
                random_state=random_seed + network_id * 100 + 11,
                n_jobs=n_jobs,
            ),
            "seed_offset": 11,
        },
        {
            "method": "lagged_genie3_extra_trees",
            "method_family": "tree_reference",
            "ranker": lambda: rank_edges_by_lagged_extra_trees(
                x_t,
                y_t1,
                n_estimators=n_estimators,
                random_state=random_seed + network_id * 100 + 22,
                n_jobs=n_jobs,
            ),
            "seed_offset": 22,
        },
    ]

    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    for config in method_configs:
        start = time.perf_counter()
        scored_edges = score_edges(config["ranker"](), truth_edges)
        fit_seconds = time.perf_counter() - start
        row, topology = evaluate_method(
            scored_edges,
            network_id=network_id,
            method=str(config["method"]),
            method_family=str(config["method_family"]),
            variant="lagged_reference",
            model_kind="tree" if str(config["method"]).startswith("lagged_genie3") else "correlation",
            target_type="level",
            self_predictor_mode="exclude_self_predictor",
            alpha=pd.NA,
            l1_ratio=pd.NA,
            score_variant="feature_importance" if str(config["method"]).startswith("lagged_genie3") else "correlation",
            n_trajectories=len(trajectories),
            n_lagged_samples=len(metadata),
            n_resamples=0,
            fit_seconds=fit_seconds,
            extra_metrics={},
        )
        metric_rows.append(row)
        topology_rows.append(topology)
        edge_audit = merge_score_columns(edge_audit, scored_edges, str(config["method"]))
    return metric_rows, edge_audit, topology_rows


def run_bootstrap_candidates(
    network_data_by_id: dict[int, dict[str, object]],
    edge_audit_by_id: dict[int, pd.DataFrame],
    candidate_configs: pd.DataFrame,
    *,
    n_resamples: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], dict[int, pd.DataFrame], list[dict[str, object]]]:
    """Run trajectory-bootstrap sparse validation for selected candidates."""
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []

    for network_id, network_data in network_data_by_id.items():
        x_t = network_data["x_t"]
        metadata = network_data["metadata"]
        targets = network_data["targets"]
        truth_edges = network_data["truth_edges"]
        trajectories = network_data["trajectories"]
        if not isinstance(x_t, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
            raise TypeError("invalid network_data")
        if not isinstance(targets, dict) or not isinstance(truth_edges, pd.DataFrame):
            raise TypeError("invalid network_data")
        if not isinstance(trajectories, list):
            raise TypeError("invalid network_data")

        resamples = trajectory_bootstrap_indices(
            metadata,
            n_resamples,
            random_seed=random_seed + network_id * 1000,
        )

        for config in candidate_configs.to_dict("records"):
            target = targets[str(config["target_type"])]
            if not isinstance(target, pd.DataFrame):
                raise TypeError("target must be a DataFrame")
            start = time.perf_counter()
            edge_summary, self_summary = summarize_resampled_dynamic_linear_coefficients(
                x_t,
                target,
                resamples,
                model_kind=str(config["model_kind"]),
                alpha=float(config["alpha"]),
                l1_ratio=float(config["l1_ratio"]) if pd.notna(config["l1_ratio"]) else None,
                self_predictor_mode=str(config["self_predictor_mode"]),
                max_iter=50000,
            )
            fit_seconds = time.perf_counter() - start
            base_method = str(config["method"])
            for score_column, score_variant in [
                ("selection_frequency", "bootstrap_selection_frequency"),
                ("mean_abs_coefficient", "bootstrap_mean_abs_coefficient"),
            ]:
                method = f"{base_method}_{score_variant}"
                predicted = edge_summary[["source", "target", score_column]].rename(
                    columns={score_column: "score"}
                )
                scored_edges = score_edges(predicted, truth_edges)
                diagnostics = persistence_diagnostics_from_bootstrap(scored_edges, self_summary)
                row, topology = evaluate_method(
                    scored_edges,
                    network_id=network_id,
                    method=method,
                    method_family="stability_sparse_linear",
                    variant="trajectory_bootstrap",
                    model_kind=str(config["model_kind"]),
                    target_type=str(config["target_type"]),
                    self_predictor_mode=str(config["self_predictor_mode"]),
                    alpha=float(config["alpha"]),
                    l1_ratio=float(config["l1_ratio"]) if pd.notna(config["l1_ratio"]) else pd.NA,
                    score_variant=score_variant,
                    n_trajectories=len(trajectories),
                    n_lagged_samples=len(metadata),
                    n_resamples=n_resamples,
                    fit_seconds=fit_seconds,
                    extra_metrics={**diagnostics, "base_method": base_method},
                )
                metric_rows.append(row)
                topology_rows.append(topology)
                edge_audit_by_id[network_id] = merge_score_columns(
                    edge_audit_by_id[network_id],
                    scored_edges,
                    method,
                )
                edge_audit_by_id[network_id] = merge_bootstrap_columns(
                    edge_audit_by_id[network_id],
                    edge_summary,
                    method,
                )
                edge_audit_by_id[network_id] = merge_bootstrap_self_columns(
                    edge_audit_by_id[network_id],
                    self_summary,
                    method,
                )

    return metric_rows, edge_audit_by_id, topology_rows


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join predicted edge scores to DREAM4 truth labels and assign ranks."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def evaluate_method(
    scored_edges: pd.DataFrame,
    *,
    network_id: int,
    method: str,
    method_family: str,
    variant: str,
    model_kind: str,
    target_type: str,
    self_predictor_mode: str,
    alpha: float | object,
    l1_ratio: float | object,
    score_variant: str,
    n_trajectories: int,
    n_lagged_samples: int,
    n_resamples: int,
    fit_seconds: float,
    extra_metrics: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute edge and topology metrics for one method/network."""
    n_true_edges = int(scored_edges["is_true"].sum())
    topology = topology_metrics_for_cutoff(scored_edges, cutoff=n_true_edges, rank_column="rank")
    base = {
        "row_type": "network",
        "data_regime": "timeseries",
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": method,
        "method_family": method_family,
        "variant": variant,
        "model_kind": model_kind,
        "target_type": target_type,
        "self_predictor_mode": self_predictor_mode,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "score_variant": score_variant,
        "n_trajectories": n_trajectories,
        "n_lagged_samples": n_lagged_samples,
        "n_resamples": n_resamples,
        "fit_seconds": fit_seconds,
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": n_true_edges,
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
        **extra_metrics,
    }
    return (
        {**base, **{f"topology_{key}": value for key, value in topology.items()}},
        {**base, **topology},
    )


def persistence_diagnostics(scored_edges: pd.DataFrame, self_coefficients: pd.DataFrame) -> dict[str, object]:
    """Summarize self-predictor strength and target-level non-self recovery."""
    mean_abs_nonself = float(scored_edges["score"].mean())
    if self_coefficients.empty:
        return {
            "mean_abs_self_coefficient": pd.NA,
            "max_abs_self_coefficient": pd.NA,
            "fraction_self_selected": pd.NA,
            "mean_abs_nonself_coefficient": mean_abs_nonself,
            "self_to_nonself_abs_ratio": pd.NA,
            "self_abs_vs_incoming_top3_precision_spearman": pd.NA,
        }

    target_precision = target_top_k_precision(scored_edges, k=3)
    merged = self_coefficients.merge(target_precision, on="target", how="left").fillna(
        {"incoming_top3_precision": 0.0}
    )
    correlation = spearman_or_zero(
        merged["self_abs_coefficient"],
        merged["incoming_top3_precision"],
    )
    return {
        "mean_abs_self_coefficient": float(self_coefficients["self_abs_coefficient"].mean()),
        "max_abs_self_coefficient": float(self_coefficients["self_abs_coefficient"].max()),
        "fraction_self_selected": float(self_coefficients["self_selected"].mean()),
        "mean_abs_nonself_coefficient": mean_abs_nonself,
        "self_to_nonself_abs_ratio": safe_ratio(
            float(self_coefficients["self_abs_coefficient"].mean()),
            mean_abs_nonself,
        ),
        "self_abs_vs_incoming_top3_precision_spearman": correlation,
    }


def persistence_diagnostics_from_bootstrap(
    scored_edges: pd.DataFrame,
    self_summary: pd.DataFrame,
) -> dict[str, object]:
    """Summarize bootstrapped self-predictor persistence."""
    mean_abs_nonself = float(scored_edges["score"].mean())
    if self_summary.empty:
        return {
            "mean_abs_self_coefficient": pd.NA,
            "max_abs_self_coefficient": pd.NA,
            "fraction_self_selected": pd.NA,
            "mean_abs_nonself_coefficient": mean_abs_nonself,
            "self_to_nonself_abs_ratio": pd.NA,
            "self_abs_vs_incoming_top3_precision_spearman": pd.NA,
        }
    target_precision = target_top_k_precision(scored_edges, k=3)
    merged = self_summary.merge(target_precision, on="target", how="left").fillna(
        {"incoming_top3_precision": 0.0}
    )
    return {
        "mean_abs_self_coefficient": float(self_summary["mean_abs_self_coefficient"].mean()),
        "max_abs_self_coefficient": float(self_summary["mean_abs_self_coefficient"].max()),
        "fraction_self_selected": float(self_summary["self_selection_frequency"].mean()),
        "mean_abs_nonself_coefficient": mean_abs_nonself,
        "self_to_nonself_abs_ratio": safe_ratio(
            float(self_summary["mean_abs_self_coefficient"].mean()),
            mean_abs_nonself,
        ),
        "self_abs_vs_incoming_top3_precision_spearman": spearman_or_zero(
            merged["mean_abs_self_coefficient"],
            merged["incoming_top3_precision"],
        ),
    }


def target_top_k_precision(scored_edges: pd.DataFrame, *, k: int) -> pd.DataFrame:
    """Compute top-k incoming-edge precision for each target gene."""
    rows = []
    for target, group in scored_edges.sort_values("rank").groupby("target"):
        rows.append({"target": target, "incoming_top3_precision": float(group.head(k)["is_true"].mean())})
    return pd.DataFrame(rows)


def spearman_or_zero(left: pd.Series, right: pd.Series) -> float:
    """Return Spearman correlation, using zero for constant or invalid inputs."""
    left = pd.to_numeric(left)
    right = pd.to_numeric(right)
    if len(left) < 2 or left.nunique() <= 1 or right.nunique() <= 1:
        return 0.0
    value = left.corr(right, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio."""
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def merge_score_columns(edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge score/rank columns for one method into the edge audit table."""
    columns = ["source", "target", "score", "rank"]
    optional = [column for column in ["coefficient", "selected"] if column in scored_edges.columns]
    method_scores = scored_edges[columns + optional].rename(
        columns={
            "score": f"score_{method}",
            "rank": f"rank_{method}",
            "coefficient": f"coefficient_{method}",
            "selected": f"selected_{method}",
        }
    )
    return edge_audit.merge(method_scores, on=["source", "target"], how="left")


def merge_self_columns(edge_audit: pd.DataFrame, self_coefficients: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge target-level self coefficients into the edge audit table."""
    if self_coefficients.empty:
        return edge_audit
    renamed = self_coefficients.rename(
        columns={
            "self_coefficient": f"self_coefficient_{method}",
            "self_abs_coefficient": f"self_abs_coefficient_{method}",
            "self_selected": f"self_selected_{method}",
        }
    )
    return edge_audit.merge(renamed, on="target", how="left")


def merge_bootstrap_columns(edge_audit: pd.DataFrame, edge_summary: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge bootstrap coefficient summaries into the edge audit table."""
    columns = [
        "source",
        "target",
        "selection_frequency",
        "mean_coefficient",
        "mean_abs_coefficient",
    ]
    renamed = edge_summary[columns].rename(
        columns={
            "selection_frequency": f"selection_frequency_{method}",
            "mean_coefficient": f"mean_coefficient_{method}",
            "mean_abs_coefficient": f"mean_abs_coefficient_{method}",
        }
    )
    return edge_audit.merge(renamed, on=["source", "target"], how="left")


def merge_bootstrap_self_columns(edge_audit: pd.DataFrame, self_summary: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge bootstrapped self-coefficient summaries into the edge audit table."""
    if self_summary.empty:
        return edge_audit
    renamed = self_summary.rename(
        columns={
            "self_selection_frequency": f"self_selection_frequency_{method}",
            "mean_self_coefficient": f"mean_self_coefficient_{method}",
            "mean_abs_self_coefficient": f"mean_abs_self_coefficient_{method}",
        }
    )
    return edge_audit.merge(renamed, on="target", how="left")


def aggregate_summary(per_network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-network validation rows across Size10 networks."""
    per_network = per_network.copy()
    if "base_method" not in per_network.columns:
        per_network["base_method"] = ""
    else:
        per_network["base_method"] = per_network["base_method"].fillna("")
    group_columns = [
        "method",
        "method_family",
        "variant",
        "model_kind",
        "target_type",
        "self_predictor_mode",
        "alpha",
        "l1_ratio",
        "score_variant",
        "base_method",
    ]
    excluded = set(group_columns) | {"row_type", "data_regime", "network_id", "network"}
    metric_columns: list[str] = []
    for column in per_network.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(per_network[column], errors="coerce")
        if numeric.notna().any():
            per_network[column] = numeric
            metric_columns.append(column)
    summary = aggregate_per_network_metrics(
        per_network,
        group_columns=group_columns,
        metric_columns=metric_columns,
    )
    summary.insert(0, "row_type", "mean")
    return summary.sort_values(["aupr", "method"], ascending=[False, True]).reset_index(drop=True)


def choose_bootstrap_candidates(summary: pd.DataFrame, grid: pd.DataFrame, *, max_candidates: int = 4) -> pd.DataFrame:
    """Choose a small sparse candidate set for trajectory-bootstrap checks."""
    sparse = summary[
        (summary["method_family"] == "sparse_linear")
        & (summary["variant"] == "one_shot_coefficient_magnitude")
    ].copy()
    selected_methods: list[str] = []
    preferred = [
        "dynamic_lasso_level_include_self_a0_03",
        best_method_name(sparse[sparse["model_kind"] == "lasso"], "aupr"),
        best_method_name(
            sparse[
                (sparse["model_kind"] == "lasso")
                & (sparse["self_predictor_mode"] == "exclude_self_predictor")
            ],
            "aupr",
        ),
        best_method_name(sparse[sparse["model_kind"] == "elastic_net"], "aupr"),
    ]
    for method in preferred:
        if method and method not in selected_methods:
            selected_methods.append(method)
        if len(selected_methods) >= max_candidates:
            break
    return grid[grid["method"].isin(selected_methods)].copy()


def best_method_name(frame: pd.DataFrame, metric: str) -> str:
    """Return the best method name from a summary frame."""
    if frame.empty:
        return ""
    return str(frame.sort_values([metric, "method"], ascending=[False, True]).iloc[0]["method"])


def build_debug_report(
    summary: pd.DataFrame,
    per_network: pd.DataFrame,
    topology: pd.DataFrame,
    trajectory_info: pd.DataFrame,
    bootstrap_candidates: pd.DataFrame,
) -> str:
    """Build a human-readable validation report."""
    best = best_summary_row(summary, "aupr")
    best_auroc = best_summary_row(summary, "auroc")
    best_sparse = best_summary_row(summary[summary["method_family"] == "sparse_linear"], "aupr")
    best_bootstrap = best_summary_row(summary[summary["variant"] == "trajectory_bootstrap"], "aupr")
    best_out_hub = best_summary_row(summary, "topology_top3_out_hub_overlap")
    best_in_hub = best_summary_row(summary, "topology_top3_in_hub_overlap")
    alpha_sensitivity = lasso_alpha_sensitivity(summary)
    per_network_best = per_network_best_methods(per_network)
    include_self = include_self_comparison(per_network)
    bootstrap = bootstrap_comparison(summary)
    reciprocal = reciprocal_comparison(summary)
    persistence = persistence_summary(summary)
    topology_winners = pd.DataFrame([best_out_hub, best_in_hub])

    lines = [
        "# DREAM4 Size10 Dynamic Sparse Validation Debug Report",
        "",
        "This audit stress-tests the strongest dynamic sparse-linear result from experiment 08. It focuses on alpha sensitivity, include-self behavior, persistence diagnostics, bootstrap selection, reciprocal errors, and topology-aware metrics.",
        "",
        "## Trajectory Summary",
        "",
        to_markdown_table(trajectory_info),
        "",
        "## Best Mean Metrics",
        "",
        to_markdown_table(pd.DataFrame([best, best_auroc, best_sparse, best_bootstrap])),
        "",
        "## Per-Network Winners By AUPR",
        "",
        to_markdown_table(per_network_best),
        "",
        "## LASSO Alpha Sensitivity",
        "",
        to_markdown_table(alpha_sensitivity),
        "",
        "## Include-Self Versus Exclude-Self",
        "",
        to_markdown_table(include_self),
        "",
        "## Bootstrap Comparisons",
        "",
        to_markdown_table(bootstrap),
        "",
        "## Persistence Diagnostics",
        "",
        to_markdown_table(persistence),
        "",
        "## Reciprocal Error Comparison",
        "",
        to_markdown_table(reciprocal),
        "",
        "## Topology Winners",
        "",
        to_markdown_table(topology_winners),
        "",
        "## Bootstrap Candidates",
        "",
        to_markdown_table(bootstrap_candidates[["method", "model_kind", "target_type", "self_predictor_mode", "alpha", "l1_ratio"]]),
        "",
        "## Interpretation",
        "",
        interpret_results(summary, per_network, bootstrap, include_self),
        "",
    ]
    return "\n".join(lines)


def best_summary_row(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    """Return a compact best-row summary."""
    if frame.empty:
        return {"metric": metric, "method": "", "value": pd.NA}
    row = frame.sort_values([metric, "method"], ascending=[False, True]).iloc[0]
    return {
        "metric": metric,
        "method": row["method"],
        "method_family": row["method_family"],
        "target_type": row["target_type"],
        "self_predictor_mode": row["self_predictor_mode"],
        "alpha": row["alpha"],
        "l1_ratio": row["l1_ratio"],
        "score_variant": row["score_variant"],
        "value": row[metric],
        "std_aupr": row.get("std_aupr", pd.NA),
    }


def lasso_alpha_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best LASSO result at each alpha."""
    lasso = summary[
        (summary["model_kind"] == "lasso")
        & (summary["variant"] == "one_shot_coefficient_magnitude")
    ].copy()
    rows = []
    for alpha, group in lasso.groupby("alpha", dropna=False):
        row = group.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "alpha": alpha,
                "best_method_at_alpha": row["method"],
                "best_aupr": row["aupr"],
                "best_auroc": row["auroc"],
                "std_aupr": row.get("std_aupr", pd.NA),
                "rank_by_aupr": 0,
            }
        )
    result = pd.DataFrame(rows).sort_values("best_aupr", ascending=False).reset_index(drop=True)
    result["rank_by_aupr"] = range(1, len(result) + 1)
    return result


def per_network_best_methods(per_network: pd.DataFrame) -> pd.DataFrame:
    """Return the best method by AUPR for each network."""
    rows = []
    for network_id, group in per_network.groupby("network_id"):
        row = group.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "network_id": network_id,
                "method": row["method"],
                "variant": row["variant"],
                "aupr": row["aupr"],
                "auroc": row["auroc"],
                "precision_at_10": row["precision_at_10"],
                "top3_out_hub_overlap": row["topology_top3_out_hub_overlap"],
                "top3_in_hub_overlap": row["topology_top3_in_hub_overlap"],
                "reciprocal_false_positive_pair_rate": row["topology_reciprocal_false_positive_pair_rate"],
            }
        )
    return pd.DataFrame(rows)


def include_self_comparison(per_network: pd.DataFrame) -> pd.DataFrame:
    """Compare LASSO include-self against exclude-self matched by target/alpha."""
    lasso = per_network[
        (per_network["model_kind"] == "lasso")
        & (per_network["variant"] == "one_shot_coefficient_magnitude")
    ].copy()
    include = lasso[lasso["self_predictor_mode"] == "include_self_predictor_no_self_edge"]
    exclude = lasso[lasso["self_predictor_mode"] == "exclude_self_predictor"]
    merged = include.merge(
        exclude,
        on=["network_id", "target_type", "alpha"],
        suffixes=("_include", "_exclude"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["delta_aupr"] = merged["aupr_include"] - merged["aupr_exclude"]
    merged["delta_auroc"] = merged["auroc_include"] - merged["auroc_exclude"]
    merged["delta_reciprocal_rate"] = (
        merged["topology_reciprocal_false_positive_pair_rate_include"]
        - merged["topology_reciprocal_false_positive_pair_rate_exclude"]
    )
    rows = []
    for (target_type, alpha), group in merged.groupby(["target_type", "alpha"], dropna=False):
        rows.append(
            {
                "target_type": target_type,
                "alpha": alpha,
                "mean_delta_aupr": group["delta_aupr"].mean(),
                "mean_delta_auroc": group["delta_auroc"].mean(),
                "include_wins_aupr": int((group["delta_aupr"] > 0).sum()),
                "exclude_wins_aupr": int((group["delta_aupr"] < 0).sum()),
                "ties_aupr": int((group["delta_aupr"] == 0).sum()),
                "mean_delta_reciprocal_rate": group["delta_reciprocal_rate"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_delta_aupr", ascending=False)


def bootstrap_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare bootstrap rankings to their one-shot base methods."""
    bootstrap = summary[summary["variant"] == "trajectory_bootstrap"].copy()
    one_shot = summary[summary["variant"] == "one_shot_coefficient_magnitude"].set_index("method")
    rows = []
    for row in bootstrap.itertuples(index=False):
        base_method = getattr(row, "base_method", "")
        if not base_method or base_method not in one_shot.index:
            continue
        base = one_shot.loc[base_method]
        rows.append(
            {
                "method": row.method,
                "base_method": base_method,
                "score_variant": row.score_variant,
                "aupr": row.aupr,
                "base_aupr": base["aupr"],
                "delta_aupr": row.aupr - base["aupr"],
                "auroc": row.auroc,
                "base_auroc": base["auroc"],
                "delta_auroc": row.auroc - base["auroc"],
            }
        )
    return pd.DataFrame(rows).sort_values("aupr", ascending=False) if rows else pd.DataFrame()


def reciprocal_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Return reciprocal false-positive metrics for important methods."""
    important = summary[
        summary["method"].isin(
            [
                best_method_name(summary, "aupr"),
                "lagged_correlation_reference",
                "lagged_genie3_random_forest",
                "lagged_genie3_extra_trees",
                "dynamic_lasso_level_include_self_a0_03",
                "dynamic_lasso_level_exclude_self_a0_03",
            ]
        )
    ].copy()
    columns = [
        "method",
        "aupr",
        "auroc",
        "topology_reciprocal_false_positive_pair_count",
        "topology_reciprocal_false_positive_pair_rate",
        "topology_reciprocal_pair_count",
    ]
    return important[columns].sort_values("aupr", ascending=False)


def persistence_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return persistence diagnostics for top include-self methods."""
    included = summary[
        summary["self_predictor_mode"].eq("include_self_predictor_no_self_edge")
        & summary["mean_abs_self_coefficient"].notna()
    ].copy()
    if included.empty:
        return pd.DataFrame()
    columns = [
        "method",
        "aupr",
        "mean_abs_self_coefficient",
        "mean_abs_nonself_coefficient",
        "self_to_nonself_abs_ratio",
        "fraction_self_selected",
        "self_abs_vs_incoming_top3_precision_spearman",
    ]
    return included[columns].sort_values("aupr", ascending=False).head(12)


def interpret_results(
    summary: pd.DataFrame,
    per_network: pd.DataFrame,
    bootstrap: pd.DataFrame,
    include_self: pd.DataFrame,
) -> str:
    """Answer the experiment questions cautiously."""
    best = best_summary_row(summary, "aupr")
    best_networks = per_network_best_methods(per_network)
    best_method_network_wins = int((best_networks["method"] == best["method"]).sum())
    lasso_alpha = lasso_alpha_sensitivity(summary)
    alpha03 = lasso_alpha[lasso_alpha["alpha"] == 0.03]
    alpha03_rank = int(alpha03["rank_by_aupr"].iloc[0]) if not alpha03.empty else 0
    include_positive = include_self[include_self["mean_delta_aupr"] > 0] if not include_self.empty else pd.DataFrame()
    bootstrap_best_delta = float(bootstrap["delta_aupr"].max()) if not bootstrap.empty else 0.0
    best_row = summary[summary["method"] == best["method"]].iloc[0]
    reciprocal_rate = best_row["topology_reciprocal_false_positive_pair_rate"]
    out_hub = best_row["topology_top3_out_hub_overlap"]
    in_hub = best_row["topology_top3_in_hub_overlap"]

    lines = [
        f"Best mean AUPR method: `{best['method']}` with AUPR {best['value']:.6f}. It is the per-network AUPR winner on {best_method_network_wins} of 5 networks.",
        f"LASSO alpha 0.03 ranks {alpha03_rank} by best mean AUPR across the LASSO alpha grid.",
        f"Include-self has positive mean AUPR deltas in {len(include_positive)} matched target/alpha comparisons; inspect the table above before treating persistence as harmless.",
        f"The best bootstrap delta AUPR versus its one-shot base is {bootstrap_best_delta:.6f}. Positive values would support bootstrap selection; negative values suggest it hurt in this audit.",
        f"The edge-metric winner has top-3 out-hub overlap {out_hub:.6f}, top-3 in-hub overlap {in_hub:.6f}, and reciprocal false-positive pair rate {reciprocal_rate:.6f}.",
        "This result is promising enough to be the current Size10 temporal sparse candidate, but it should not be treated as final until the self-predictor effect is checked on Size100 or simulation sweeps.",
        "Recommended next branch: validate this dynamic sparse-linear include-self candidate against a literature-faithful dynGENIE3 reproduction, then scale to richer data once the persistence diagnostic looks sane.",
    ]
    return "\n".join(lines)


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as Markdown without optional dependencies."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = [[format_cell(value) for value in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_cell(value: object) -> str:
    """Format values for Markdown table cells."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if pd.isna(value):
        return ""
    return str(value)


def print_summary(summary: pd.DataFrame, bootstrap_candidates: pd.DataFrame) -> None:
    """Print a compact validation summary."""
    columns = [
        "method",
        "method_family",
        "variant",
        "target_type",
        "self_predictor_mode",
        "alpha",
        "l1_ratio",
        "score_variant",
        "auroc",
        "aupr",
        "precision_at_10",
        "topology_top3_out_hub_overlap",
        "topology_top3_in_hub_overlap",
        "topology_reciprocal_false_positive_pair_rate",
    ]
    print("DREAM4 Size10 dynamic sparse validation")
    print()
    print(summary[columns].head(25).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("bootstrap candidates:")
    print(bootstrap_candidates[["method", "model_kind", "target_type", "self_predictor_mode", "alpha", "l1_ratio"]].to_string(index=False))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_per_network: {PER_NETWORK_PATH.as_posix()}")
    print(f"saved_edges: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_topology: {TOPOLOGY_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-resamples", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    """Run the validation audit and write result artifacts."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grid = build_dynamic_sparse_linear_grid(
        lasso_alphas=LASSO_ALPHAS,
        elastic_net_alphas=ELASTIC_NET_ALPHAS,
        elastic_net_l1_ratios=ELASTIC_NET_L1_RATIOS,
        target_types=TARGET_TYPES,
    )

    network_data_by_id: dict[int, dict[str, object]] = {}
    edge_audit_by_id: dict[int, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, int]] = []

    for network_id in range(1, 6):
        network_data = load_network_data(network_id)
        network_data_by_id[network_id] = network_data
        sparse_rows, edge_audit, sparse_topology, _ = run_sparse_grid_for_network(network_data, grid)
        reference_rows, edge_audit, reference_topology = run_reference_methods_for_network(
            network_data,
            edge_audit,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
        edge_audit_by_id[network_id] = edge_audit
        metric_rows.extend(sparse_rows)
        metric_rows.extend(reference_rows)
        topology_rows.extend(sparse_topology)
        topology_rows.extend(reference_topology)
        metadata = network_data["metadata"]
        trajectories = network_data["trajectories"]
        timeseries = network_data["timeseries"]
        if not isinstance(metadata, pd.DataFrame) or not isinstance(trajectories, list):
            raise TypeError("invalid network_data")
        if not isinstance(timeseries, pd.DataFrame):
            raise TypeError("invalid network_data")
        trajectory_rows.append(
            {
                "network_id": network_id,
                "n_trajectories": len(trajectories),
                "n_timeseries_rows": len(timeseries),
                "n_lagged_samples": len(metadata),
            }
        )

    one_shot_per_network = pd.DataFrame(metric_rows)
    one_shot_summary = aggregate_summary(one_shot_per_network)
    bootstrap_candidates = choose_bootstrap_candidates(one_shot_summary, grid)
    bootstrap_rows, edge_audit_by_id, bootstrap_topology = run_bootstrap_candidates(
        network_data_by_id,
        edge_audit_by_id,
        bootstrap_candidates,
        n_resamples=args.n_resamples,
        random_seed=args.random_seed,
    )
    metric_rows.extend(bootstrap_rows)
    topology_rows.extend(bootstrap_topology)

    per_network = pd.DataFrame(metric_rows)
    topology = pd.DataFrame(topology_rows)
    summary = aggregate_summary(per_network)
    edge_audit = pd.concat([edge_audit_by_id[network_id] for network_id in sorted(edge_audit_by_id)], ignore_index=True)
    trajectory_info = pd.DataFrame(trajectory_rows)

    summary.to_csv(SUMMARY_PATH, index=False)
    per_network.to_csv(PER_NETWORK_PATH, index=False)
    edge_audit.to_csv(EDGE_AUDIT_PATH, index=False)
    topology.to_csv(TOPOLOGY_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(
        build_debug_report(summary, per_network, topology, trajectory_info, bootstrap_candidates),
        encoding="utf-8",
    )
    print_summary(summary, bootstrap_candidates)


if __name__ == "__main__":
    main()
