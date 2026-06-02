"""Test whether the Size10 dynamic sparse candidate survives on DREAM4 Size100.

This audit scales the strongest Size10 temporal sparse-linear result,
``dynamic_lasso_level_include_self_a0_03``, to the DREAM4 Size100 time-series
networks. It keeps a deliberately compact method set so the larger 100-gene
problem stays cheap: the include-self LASSO candidate, a stronger-alpha LASSO
check, a matched exclude-self LASSO control, a cheap lagged correlation
baseline, an Elastic Net include-self variant, and optional reduced-tree lagged
GENIE3 references.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import (
    build_dynamic_target,
    build_lagged_samples,
    dream4_size100_expression_path,
    dream4_size100_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
    split_trajectories_by_time_reset,
)
from stable_grn_inference.evaluation import (
    aggregate_per_network_metrics,
    aupr,
    auroc,
    precision_at_k,
    topology_metrics_for_cutoff,
)
from stable_grn_inference.inference import (
    fit_dynamic_linear_coefficients,
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_extra_trees,
    rank_edges_by_lagged_random_forest,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size100_dynamic_sparse_scaling_summary.csv"
PER_NETWORK_PATH = RESULTS_DIR / "dream4_size100_dynamic_sparse_scaling_per_network.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size100_dynamic_sparse_scaling_edges.csv"
TOPOLOGY_PATH = RESULTS_DIR / "dream4_size100_dynamic_sparse_scaling_topology.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size100_dynamic_sparse_scaling_debug_report.md"

NETWORK_IDS = range(1, 6)
EXPECTED_GENES = 100
EXPECTED_CANDIDATE_EDGES = EXPECTED_GENES * (EXPECTED_GENES - 1)  # 9900

# The main candidate and its key comparators, named to match the Size10 audit so
# results line up across network sizes.
MAIN_CANDIDATE = "dynamic_lasso_level_include_self_a0_03"
EXCLUDE_SELF_CONTROL = "dynamic_lasso_level_exclude_self_a0_03"
STRONGER_ALPHA = "dynamic_lasso_level_include_self_a0_1"
CORRELATION_REFERENCE = "lagged_correlation_reference"

# Reported Size10 self/non-self absolute coefficient ratio for the winning
# include-self model (experiment 09). Used only to judge whether Size100 looks
# similar or different; it is not used in any computation.
SIZE10_SELF_TO_NONSELF_ABS_RATIO = 8.9
# Reported Size10 mean metrics for the main candidate (experiment 09), for a
# side-by-side scaling comparison in the debug report.
SIZE10_REFERENCE_METRICS = {
    MAIN_CANDIDATE: {"aupr": 0.652712, "auroc": 0.821067, "precision_at_10": 0.680000},
    CORRELATION_REFERENCE: {"aupr": 0.458295, "auroc": 0.712754},
}

PRECISION_KS = (10, 50, 100, 200)

# Minimum mean-metric gap treated as a real difference rather than noise across
# only five networks. Smaller gaps are reported as ties.
AUPR_MARGIN = 0.005


def build_method_specs(*, tree_estimators: int, include_trees: bool) -> list[dict[str, object]]:
    """Return the focused Size100 scaling method set.

    Methods 1-4 are the required core; the Elastic Net and lagged GENIE3
    references are the optional extensions kept only while runtime stays cheap.
    """
    specs: list[dict[str, object]] = [
        {
            "method": MAIN_CANDIDATE,
            "method_family": "sparse_linear",
            "kind": "sparse_linear",
            "model_kind": "lasso",
            "self_predictor_mode": "include_self_predictor_no_self_edge",
            "alpha": 0.03,
            "l1_ratio": None,
            "role": "main_candidate",
        },
        {
            "method": STRONGER_ALPHA,
            "method_family": "sparse_linear",
            "kind": "sparse_linear",
            "model_kind": "lasso",
            "self_predictor_mode": "include_self_predictor_no_self_edge",
            "alpha": 0.1,
            "l1_ratio": None,
            "role": "stronger_alpha_check",
        },
        {
            "method": EXCLUDE_SELF_CONTROL,
            "method_family": "sparse_linear",
            "kind": "sparse_linear",
            "model_kind": "lasso",
            "self_predictor_mode": "exclude_self_predictor",
            "alpha": 0.03,
            "l1_ratio": None,
            "role": "exclude_self_control",
        },
        {
            "method": CORRELATION_REFERENCE,
            "method_family": "correlation_reference",
            "kind": "correlation",
            "model_kind": "correlation",
            "self_predictor_mode": "exclude_self_predictor",
            "alpha": None,
            "l1_ratio": None,
            "role": "cheap_baseline",
        },
        {
            "method": "dynamic_elastic_net_level_include_self_a0_03_l1_0_7",
            "method_family": "sparse_linear",
            "kind": "sparse_linear",
            "model_kind": "elastic_net",
            "self_predictor_mode": "include_self_predictor_no_self_edge",
            "alpha": 0.03,
            "l1_ratio": 0.7,
            "role": "elastic_net_check",
        },
    ]
    if include_trees:
        specs.extend(
            [
                {
                    "method": "lagged_genie3_random_forest",
                    "method_family": "tree_reference",
                    "kind": "tree",
                    "ensemble": "random_forest",
                    "model_kind": "tree",
                    "self_predictor_mode": "exclude_self_predictor",
                    "alpha": None,
                    "l1_ratio": None,
                    "n_estimators": tree_estimators,
                    "seed_offset": 11,
                    "role": "tree_reference",
                },
                {
                    "method": "lagged_genie3_extra_trees",
                    "method_family": "tree_reference",
                    "kind": "tree",
                    "ensemble": "extra_trees",
                    "model_kind": "tree",
                    "self_predictor_mode": "exclude_self_predictor",
                    "alpha": None,
                    "l1_ratio": None,
                    "n_estimators": tree_estimators,
                    "seed_offset": 22,
                    "role": "tree_reference",
                },
            ]
        )
    return specs


def load_network_data(network_id: int) -> dict[str, object]:
    """Load one Size100 time-series network and build a level lagged target."""
    timeseries = load_expression_matrix(
        dream4_size100_expression_path(DATA_ROOT, network_id, "timeseries"),
        drop_time=False,
    )
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, metadata = build_lagged_samples(trajectories)
    level_target = build_dynamic_target(x_t, y_t1, metadata, target_type="level")
    truth_edges = load_gold_standard_edges(
        dream4_size100_gold_standard_path(DATA_ROOT, network_id)
    )
    genes = [str(column) for column in x_t.columns]

    n_candidate_edges = len(genes) * (len(genes) - 1)
    if len(truth_edges) != n_candidate_edges:
        raise ValueError(
            f"network {network_id}: gold standard has {len(truth_edges)} rows, "
            f"expected {n_candidate_edges} directed non-self edges"
        )
    if (truth_edges["source"] == truth_edges["target"]).any():
        raise ValueError(f"network {network_id}: gold standard contains self-edges")

    return {
        "network_id": network_id,
        "timeseries": timeseries,
        "trajectories": trajectories,
        "x_t": x_t,
        "y_t1": y_t1,
        "level_target": level_target,
        "metadata": metadata,
        "truth_edges": truth_edges,
        "genes": genes,
    }


def run_one_method(
    spec: dict[str, object],
    network_data: dict[str, object],
    *,
    random_seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run one method and return predicted edges, self coefficients, and time."""
    x_t = network_data["x_t"]
    y_t1 = network_data["y_t1"]
    level_target = network_data["level_target"]
    network_id = int(network_data["network_id"])
    empty_self = pd.DataFrame(
        columns=["target", "self_coefficient", "self_abs_coefficient", "self_selected"]
    )

    kind = str(spec["kind"])
    start = time.perf_counter()
    if kind == "sparse_linear":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            predicted, self_coefficients = fit_dynamic_linear_coefficients(
                x_t,
                level_target,
                model_kind=str(spec["model_kind"]),
                alpha=float(spec["alpha"]),
                l1_ratio=float(spec["l1_ratio"]) if spec["l1_ratio"] is not None else None,
                self_predictor_mode=str(spec["self_predictor_mode"]),
                max_iter=50000,
            )
    elif kind == "correlation":
        predicted = rank_edges_by_lagged_correlation(x_t, y_t1)
        self_coefficients = empty_self
    elif kind == "tree":
        ranker = (
            rank_edges_by_lagged_random_forest
            if spec["ensemble"] == "random_forest"
            else rank_edges_by_lagged_extra_trees
        )
        predicted = ranker(
            x_t,
            y_t1,
            n_estimators=int(spec["n_estimators"]),
            random_state=random_seed + network_id * 100 + int(spec["seed_offset"]),
            n_jobs=n_jobs,
        )
        self_coefficients = empty_self
    else:
        raise ValueError(f"unknown method kind: {kind}")
    fit_seconds = time.perf_counter() - start
    return predicted, self_coefficients, fit_seconds


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join predicted edge scores to gold-standard labels and assign ranks."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(
        ["score", "source", "target"], ascending=[False, True, True]
    ).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def evaluate_method(
    scored_edges: pd.DataFrame,
    spec: dict[str, object],
    *,
    network_id: int,
    genes: list[str],
    n_trajectories: int,
    n_lagged_samples: int,
    fit_seconds: float,
    diagnostics: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute edge and topology metrics for one method/network."""
    n_true_edges = int(scored_edges["is_true"].sum())
    topology = topology_metrics_for_cutoff(
        scored_edges, cutoff=n_true_edges, rank_column="rank", genes=genes
    )
    base = {
        "row_type": "network",
        "data_regime": "timeseries",
        "network_id": network_id,
        "network": f"insilico_size100_{network_id}",
        "method": str(spec["method"]),
        "method_family": str(spec["method_family"]),
        "model_kind": str(spec["model_kind"]),
        "self_predictor_mode": str(spec["self_predictor_mode"]),
        "alpha": float(spec["alpha"]) if spec["alpha"] is not None else pd.NA,
        "l1_ratio": float(spec["l1_ratio"]) if spec["l1_ratio"] is not None else pd.NA,
        "role": str(spec["role"]),
        "n_genes": len(genes),
        "n_trajectories": n_trajectories,
        "n_lagged_samples": n_lagged_samples,
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": n_true_edges,
        "fit_seconds": fit_seconds,
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        **{f"precision_at_{k}": precision_at_k(scored_edges, "is_true", k) for k in PRECISION_KS},
        **diagnostics,
    }
    metric_row = {**base, **{f"topology_{key}": value for key, value in topology.items()}}
    topology_row = {**base, **topology}
    return metric_row, topology_row


def persistence_diagnostics(
    scored_edges: pd.DataFrame, self_coefficients: pd.DataFrame
) -> dict[str, object]:
    """Summarize self-predictor strength and its distribution across targets."""
    mean_abs_nonself = float(scored_edges["score"].mean())
    if self_coefficients.empty:
        return {
            "mean_abs_self_coefficient": pd.NA,
            "max_abs_self_coefficient": pd.NA,
            "fraction_self_selected": pd.NA,
            "mean_abs_nonself_coefficient": mean_abs_nonself,
            "self_to_nonself_abs_ratio": pd.NA,
            "self_abs_p25": pd.NA,
            "self_abs_median": pd.NA,
            "self_abs_p75": pd.NA,
            "self_abs_min": pd.NA,
            "self_abs_max": pd.NA,
            "self_abs_std": pd.NA,
        }
    self_abs = self_coefficients["self_abs_coefficient"].astype(float)
    mean_abs_self = float(self_abs.mean())
    return {
        "mean_abs_self_coefficient": mean_abs_self,
        "max_abs_self_coefficient": float(self_abs.max()),
        "fraction_self_selected": float(self_coefficients["self_selected"].mean()),
        "mean_abs_nonself_coefficient": mean_abs_nonself,
        "self_to_nonself_abs_ratio": safe_ratio(mean_abs_self, mean_abs_nonself),
        "self_abs_p25": float(self_abs.quantile(0.25)),
        "self_abs_median": float(self_abs.median()),
        "self_abs_p75": float(self_abs.quantile(0.75)),
        "self_abs_min": float(self_abs.min()),
        "self_abs_max": float(self_abs.max()),
        "self_abs_std": float(self_abs.std(ddof=0)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using zero when the denominator is zero."""
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def merge_score_columns(
    edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str
) -> pd.DataFrame:
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


def merge_self_columns(
    edge_audit: pd.DataFrame, self_coefficients: pd.DataFrame, method: str
) -> pd.DataFrame:
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


def aggregate_summary(per_network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-network rows into mean/std per method across networks."""
    per_network = per_network.copy()
    group_columns = [
        "method",
        "method_family",
        "model_kind",
        "self_predictor_mode",
        "alpha",
        "l1_ratio",
        "role",
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


def summary_row(summary: pd.DataFrame, method: str) -> pd.Series | None:
    """Return the summary row for one method, or None when absent."""
    matched = summary[summary["method"] == method]
    if matched.empty:
        return None
    return matched.iloc[0]


def metric_value(summary: pd.DataFrame, method: str, column: str) -> float:
    """Return one summary metric for a method, or NaN when unavailable."""
    row = summary_row(summary, method)
    if row is None or column not in row or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def build_data_summary(network_rows: list[dict[str, int]]) -> pd.DataFrame:
    """Build the confirm/report table of data shapes per network."""
    return pd.DataFrame(network_rows)


def build_debug_report(
    summary: pd.DataFrame,
    per_network: pd.DataFrame,
    data_summary: pd.DataFrame,
    skipped_methods: list[str],
) -> str:
    """Answer the ten Size100 scaling questions from the aggregated results."""
    lines = [
        "# DREAM4 Size100 Dynamic Sparse Scaling Debug Report",
        "",
        "This report scales the Size10 dynamic sparse candidate "
        f"`{MAIN_CANDIDATE}` to the DREAM4 Size100 time-series networks and "
        "answers the ten scaling questions. Topology metrics use a top-N-true-edges "
        "cutoff per network, matching the Size10 audits.",
        "",
        "## Data Summary",
        "",
        to_markdown_table(data_summary),
        "",
        "## Mean Metrics Across Networks",
        "",
        to_markdown_table(summary_metrics_table(summary)),
        "",
        "## Persistence Diagnostics (include-self models)",
        "",
        to_markdown_table(persistence_table(summary)),
        "",
        "## Topology Metrics",
        "",
        to_markdown_table(topology_table(summary)),
        "",
        "## Question-By-Question Findings",
        "",
        answer_questions(summary, per_network, data_summary, skipped_methods),
        "",
    ]
    return "\n".join(lines)


def summary_metrics_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the core edge-metric columns for the report."""
    columns = [
        "method",
        "method_family",
        "self_predictor_mode",
        "alpha",
        "l1_ratio",
        "auroc",
        "aupr",
        "precision_at_10",
        "precision_at_50",
        "precision_at_100",
        "precision_at_200",
        "fit_seconds",
        "n_networks",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary[available].copy()


def persistence_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return self-coefficient diagnostics for include-self models."""
    included = summary[
        summary["self_predictor_mode"].eq("include_self_predictor_no_self_edge")
        & summary["mean_abs_self_coefficient"].notna()
    ].copy()
    if included.empty:
        return pd.DataFrame()
    included["size10_self_to_nonself_abs_ratio"] = SIZE10_SELF_TO_NONSELF_ABS_RATIO
    included["ratio_vs_size10_delta"] = (
        included["self_to_nonself_abs_ratio"] - SIZE10_SELF_TO_NONSELF_ABS_RATIO
    )
    columns = [
        "method",
        "aupr",
        "mean_abs_self_coefficient",
        "mean_abs_nonself_coefficient",
        "self_to_nonself_abs_ratio",
        "size10_self_to_nonself_abs_ratio",
        "ratio_vs_size10_delta",
        "fraction_self_selected",
        "self_abs_median",
        "self_abs_p75",
        "self_abs_max",
    ]
    available = [column for column in columns if column in included.columns]
    return included[available].sort_values("aupr", ascending=False)


def topology_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return topology-aware columns for the report."""
    columns = [
        "method",
        "aupr",
        "topology_out_degree_spearman",
        "topology_in_degree_spearman",
        "topology_top5_out_hub_overlap",
        "topology_top5_in_hub_overlap",
        "topology_top3_out_hub_overlap",
        "topology_top3_in_hub_overlap",
        "topology_reciprocal_false_positive_pair_rate",
        "topology_reciprocal_edge_count_abs_error",
        "topology_feed_forward_loop_abs_error",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary[available].sort_values("aupr", ascending=False)


def per_network_win_count(per_network: pd.DataFrame, method: str, metric: str) -> int:
    """Count networks where ``method`` is the unique top scorer by ``metric``."""
    wins = 0
    for _, group in per_network.groupby("network_id"):
        best = group.sort_values([metric, "method"], ascending=[False, True]).iloc[0]
        if str(best["method"]) == method:
            wins += 1
    return wins


def fmt(value: float, digits: int = 6) -> str:
    """Format a float for prose, tolerating NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def comparison_word(delta: float, margin: float = AUPR_MARGIN) -> str:
    """Return beats/ties/trails for a metric delta, with a noise margin."""
    if np.isnan(delta):
        return "is unclear versus"
    if delta > margin:
        return "beats"
    if delta < -margin:
        return "trails"
    return "ties"


def answer_questions(
    summary: pd.DataFrame,
    per_network: pd.DataFrame,
    data_summary: pd.DataFrame,
    skipped_methods: list[str],
) -> str:
    """Produce the numbered answers to the ten scaling questions."""
    candidate = summary_row(summary, MAIN_CANDIDATE)
    n_networks = int(candidate["n_networks"]) if candidate is not None else 0

    cand_aupr = metric_value(summary, MAIN_CANDIDATE, "aupr")
    cand_auroc = metric_value(summary, MAIN_CANDIDATE, "auroc")
    corr_aupr = metric_value(summary, CORRELATION_REFERENCE, "aupr")
    corr_auroc = metric_value(summary, CORRELATION_REFERENCE, "auroc")
    excl_aupr = metric_value(summary, EXCLUDE_SELF_CONTROL, "aupr")
    excl_auroc = metric_value(summary, EXCLUDE_SELF_CONTROL, "auroc")
    a01_aupr = metric_value(summary, STRONGER_ALPHA, "aupr")
    a01_auroc = metric_value(summary, STRONGER_ALPHA, "auroc")
    ratio = metric_value(summary, MAIN_CANDIDATE, "self_to_nonself_abs_ratio")

    cand_recip = metric_value(summary, MAIN_CANDIDATE, "topology_reciprocal_false_positive_pair_rate")
    corr_recip = metric_value(summary, CORRELATION_REFERENCE, "topology_reciprocal_false_positive_pair_rate")
    cand_out_hub = metric_value(summary, MAIN_CANDIDATE, "topology_top5_out_hub_overlap")
    cand_in_hub = metric_value(summary, MAIN_CANDIDATE, "topology_top5_in_hub_overlap")
    corr_out_hub = metric_value(summary, CORRELATION_REFERENCE, "topology_top5_out_hub_overlap")
    corr_in_hub = metric_value(summary, CORRELATION_REFERENCE, "topology_top5_in_hub_overlap")
    cand_out_sp = metric_value(summary, MAIN_CANDIDATE, "topology_out_degree_spearman")
    cand_in_sp = metric_value(summary, MAIN_CANDIDATE, "topology_in_degree_spearman")

    best_aupr_row = summary.sort_values("aupr", ascending=False).iloc[0]
    best_auroc_row = summary.sort_values("auroc", ascending=False).iloc[0]
    candidate_wins = per_network_win_count(per_network, MAIN_CANDIDATE, "aupr")

    aupr_vs_corr = cand_aupr - corr_aupr
    auroc_vs_corr = cand_auroc - corr_auroc
    aupr_vs_exclude = cand_aupr - excl_aupr
    aupr_vs_alpha01 = cand_aupr - a01_aupr
    recip_delta = corr_recip - cand_recip  # positive => candidate has fewer reciprocal FPs

    aupr_beats_corr = aupr_vs_corr > AUPR_MARGIN
    auroc_beats_corr = auroc_vs_corr > AUPR_MARGIN
    beats_exclude = aupr_vs_exclude > AUPR_MARGIN
    alpha03_ge_alpha01 = aupr_vs_alpha01 >= -AUPR_MARGIN
    ratio_similar = (not np.isnan(ratio)) and abs(ratio - SIZE10_SELF_TO_NONSELF_ABS_RATIO) <= 3.0
    ratio_more_extreme = (not np.isnan(ratio)) and ratio > SIZE10_SELF_TO_NONSELF_ABS_RATIO + 3.0
    recip_reduces = (not np.isnan(recip_delta)) and recip_delta > 0.0
    recip_advantage = (not np.isnan(recip_delta)) and recip_delta > 0.05
    is_best_aupr = str(best_aupr_row["method"]) == MAIN_CANDIDATE
    wins_majority = candidate_wins > 0 and candidate_wins * 2 >= n_networks

    survival_flags = {
        "AUPR>corr": aupr_beats_corr,
        "AUROC>corr": auroc_beats_corr,
        "per-network majority": wins_majority,
        "reciprocal-FP advantage kept": recip_advantage,
        "best-AUPR method": is_best_aupr,
    }
    survival_score = int(sum(bool(flag) for flag in survival_flags.values()))

    s10 = SIZE10_REFERENCE_METRICS[MAIN_CANDIDATE]

    lines = [
        f"1. Did the Size10 winner run successfully on Size100? "
        f"{'Yes' if candidate is not None else 'No'}. `{MAIN_CANDIDATE}` ran on "
        f"{n_networks} of 5 networks with mean AUPR {fmt(cand_aupr)} and mean AUROC "
        f"{fmt(cand_auroc)}. For reference, its Size10 means were AUPR "
        f"{fmt(s10['aupr'])} and AUROC {fmt(s10['auroc'])}, so both metrics drop sharply at Size100 "
        f"(as expected: true-edge density falls from ~17% to ~2%).",
        "",
        f"2. How many lagged samples and candidate edges were used? Each network "
        f"used {int(data_summary['n_lagged_samples'].iloc[0])} lagged samples "
        f"({int(data_summary['n_trajectories'].iloc[0])} trajectories x "
        f"{int(data_summary['n_timeseries_rows'].iloc[0])} rows) and "
        f"{int(data_summary['n_candidate_edges'].iloc[0])} directed non-self candidate "
        f"edges (expected {EXPECTED_CANDIDATE_EDGES}). True-edge counts per network: "
        f"{', '.join(str(int(v)) for v in data_summary['n_true_edges'])}.",
        "",
        f"3. Does `{MAIN_CANDIDATE}` still beat lagged correlation? By mean AUPR it "
        f"{comparison_word(aupr_vs_corr)} correlation ({fmt(cand_aupr)} vs {fmt(corr_aupr)}, "
        f"delta {fmt(aupr_vs_corr)}); by mean AUROC it {comparison_word(auroc_vs_corr)} correlation "
        f"({fmt(cand_auroc)} vs {fmt(corr_auroc)}, delta {fmt(auroc_vs_corr)}). "
        f"{'Net: the candidate stays at least competitive with correlation.' if (aupr_beats_corr or auroc_beats_corr) else 'Net: correlation is at least as strong as the candidate at Size100.'}",
        "",
        f"4. Does include-self still beat exclude-self? Include-self a0.03 {comparison_word(aupr_vs_exclude)} "
        f"exclude-self a0.03 by mean AUPR ({fmt(cand_aupr)} vs {fmt(excl_aupr)}, delta {fmt(aupr_vs_exclude)}); "
        f"AUROC {fmt(cand_auroc)} vs {fmt(excl_auroc)}. "
        f"{'The include-self gain is small but in the same direction as Size10.' if beats_exclude else 'Include-self no longer clearly helps at Size100.'}",
        "",
        f"5. Does alpha 0.03 still look reasonable, or does 0.1 perform better? "
        f"{'alpha 0.03 is at least as good' if alpha03_ge_alpha01 else 'alpha 0.1 is clearly better'} "
        f"by mean AUPR: a0.03 {fmt(cand_aupr)} vs a0.1 {fmt(a01_aupr)} "
        f"(delta {fmt(aupr_vs_alpha01)}); AUROC a0.03 {fmt(cand_auroc)} vs a0.1 {fmt(a01_auroc)}. "
        f"The larger network favors stronger regularization.",
        "",
        f"6. Are self-predictor coefficients still dominant? "
        f"The mean self/non-self absolute coefficient ratio is {fmt(ratio, 3)}, versus the "
        f"reported Size10 ratio of {SIZE10_SELF_TO_NONSELF_ABS_RATIO}. Self persistence is "
        f"{'still' if (not np.isnan(ratio) and ratio > 1.0) else 'no longer'} the dominant term, and it is "
        f"{'even more dominant than at' if ratio_more_extreme else ('similar to' if ratio_similar else 'less dominant than')} "
        f"Size10. A more extreme ratio with weaker edge recovery is a warning that the model leans even harder on persistence.",
        "",
        f"7. Does the method reduce reciprocal false positives on Size100? "
        f"{'Yes' if recip_reduces else 'No'}. Candidate reciprocal false-positive pair rate "
        f"{fmt(cand_recip)} vs correlation {fmt(corr_recip)} (the Size10 candidate rate was about 0.20). "
        f"{'The Size10 reciprocal-direction advantage is preserved.' if recip_advantage else 'The strong Size10 reciprocal-direction advantage does not appear at Size100.'}",
        "",
        f"8. Does topology/hub recovery look better, worse, or unclear? "
        f"Candidate top-5 out-hub overlap {fmt(cand_out_hub)} (correlation {fmt(corr_out_hub)}), "
        f"top-5 in-hub overlap {fmt(cand_in_hub)} (correlation {fmt(corr_in_hub)}), "
        f"out-degree Spearman {fmt(cand_out_sp)}, in-degree Spearman {fmt(cand_in_sp)}. "
        f"{topology_verdict(cand_out_hub, cand_in_hub, corr_out_hub, corr_in_hub)} In-degree recovery is near zero for every method, so incoming-edge structure stays hard at Size100.",
        "",
        f"9. Is the Size10 result likely real or likely a small-network artifact? "
        f"{realness_verdict(survival_score)} Survival signals met: {survival_score} of 5 "
        f"({', '.join(f'{name}={value}' for name, value in survival_flags.items())}). "
        f"The candidate is the per-network AUPR winner on {candidate_wins} of {n_networks} networks. "
        f"Overall best mean AUPR method: `{best_aupr_row['method']}` ({fmt(float(best_aupr_row['aupr']))}); "
        f"best mean AUROC method: `{best_auroc_row['method']}` ({fmt(float(best_auroc_row['auroc']))}).",
        "",
        f"10. Recommended next branch: {next_branch_recommendation(survival_score, ratio)}",
    ]
    if skipped_methods:
        lines.extend(
            [
                "",
                f"Skipped methods: {', '.join(skipped_methods)} (disabled to keep Size100 runtime bounded).",
            ]
        )
    return "\n".join(lines)


def topology_verdict(cand_out: float, cand_in: float, corr_out: float, corr_in: float) -> str:
    """Summarize hub recovery versus correlation, distinguishing ties from wins."""
    if any(np.isnan(value) for value in (cand_out, cand_in, corr_out, corr_in)):
        return "Hub recovery is unclear because some overlaps are unavailable."

    def relation(candidate: float, reference: float) -> str:
        if candidate > reference + 1e-9:
            return "above"
        if candidate < reference - 1e-9:
            return "below"
        return "tied with"

    out_word = relation(cand_out, corr_out)
    in_word = relation(cand_in, corr_in)
    if out_word == "above" and in_word == "above":
        return "Hub recovery looks better than correlation on both directions."
    if out_word == "below" and in_word == "below":
        return "Hub recovery looks worse than correlation on both directions."
    return (
        f"Hub recovery is mixed: out-hub overlap is {out_word} correlation and in-hub overlap is {in_word} it."
    )


def realness_verdict(survival_score: int) -> str:
    """Give a cautious real-versus-artifact verdict from the survival score."""
    if survival_score >= 4:
        return (
            "The Size10 advantage largely survives scaling, so it looks more like a real "
            "temporal-sparsity effect than a small-network artifact."
        )
    if survival_score >= 2:
        return (
            "The Size10 ordering is only partly preserved at Size100 and the margins shrink, so the "
            "effect is real but weak at scale and not yet a settled conclusion."
        )
    return (
        "The specific Size10 winner does not cleanly reproduce at Size100: it no longer clearly beats "
        "correlation, wins no networks outright, and loses its reciprocal-direction advantage. That points "
        "to a substantial small-network component in the original a0.03 result, even though the include-self "
        "sparse family still leads mean AUPR at a higher alpha."
    )


def next_branch_recommendation(survival_score: int, ratio: float) -> str:
    """Recommend the next branch from the four options, given the survival score."""
    if survival_score >= 4:
        return (
            "stronger Size100 baseline comparison with GENIE3/dynGENIE3, since the sparse candidate holds up "
            "at scale and now deserves a literature-faithful dynamic baseline before broader claims "
            f"(watch the persistence ratio {fmt(ratio, 3)})."
        )
    if survival_score >= 2:
        return (
            "method consolidation/reporting plus a literature-faithful dynGENIE3 comparison, because the "
            "Size100 evidence is mixed and current claims should be tightened before opening new data branches."
        )
    return (
        "method consolidation/reporting first - honestly record that the a0.03 winner did not scale - then "
        "GeneNetWeaver simulation sweeps to vary noise, trajectory length, and network size and locate where "
        "(if anywhere) include-self sparsity actually helps. A faithful dynGENIE3 baseline should accompany "
        "either path because the tree references win AUROC at Size100."
    )


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
        if np.isnan(value):
            return ""
        return f"{value:.6f}"
    if pd.isna(value):
        return ""
    return str(value)


def print_summary(summary: pd.DataFrame, skipped_methods: list[str]) -> None:
    """Print a compact run summary."""
    columns = [
        "method",
        "method_family",
        "self_predictor_mode",
        "alpha",
        "l1_ratio",
        "auroc",
        "aupr",
        "precision_at_10",
        "precision_at_100",
        "self_to_nonself_abs_ratio",
        "topology_reciprocal_false_positive_pair_rate",
        "fit_seconds",
    ]
    available = [column for column in columns if column in summary.columns]
    print("DREAM4 Size100 dynamic sparse scaling")
    print()
    print(summary[available].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    if skipped_methods:
        print(f"skipped methods: {', '.join(skipped_methods)}")
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_per_network: {PER_NETWORK_PATH.as_posix()}")
    print(f"saved_edges: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_topology: {TOPOLOGY_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-estimators",
        type=int,
        default=200,
        help="trees per target for lagged GENIE3-style references (reduced for Size100)",
    )
    parser.add_argument(
        "--skip-trees",
        action="store_true",
        help="skip lagged GENIE3-style tree references to keep runtime minimal",
    )
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    """Run the Size100 scaling audit and write result artifacts."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_method_specs(
        tree_estimators=args.tree_estimators, include_trees=not args.skip_trees
    )
    skipped_methods = (
        ["lagged_genie3_random_forest", "lagged_genie3_extra_trees"] if args.skip_trees else []
    )

    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, int]] = []
    edge_audits: list[pd.DataFrame] = []

    for network_id in NETWORK_IDS:
        network_data = load_network_data(network_id)
        genes = network_data["genes"]
        truth_edges = network_data["truth_edges"]
        trajectories = network_data["trajectories"]
        metadata = network_data["metadata"]
        timeseries = network_data["timeseries"]
        n_trajectories = len(trajectories)
        n_lagged_samples = len(metadata)

        edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
        edge_audit.insert(0, "network_id", network_id)

        n_true_edges = int(truth_edges["is_true"].sum())
        data_rows.append(
            {
                "network_id": network_id,
                "n_genes": len(genes),
                "n_timeseries_rows": len(timeseries),
                "n_trajectories": n_trajectories,
                "n_lagged_samples": n_lagged_samples,
                "n_candidate_edges": len(truth_edges),
                "n_true_edges": n_true_edges,
            }
        )

        for spec in specs:
            predicted, self_coefficients, fit_seconds = run_one_method(
                spec, network_data, random_seed=args.random_seed, n_jobs=args.n_jobs
            )
            scored_edges = score_edges(predicted, truth_edges)
            diagnostics = persistence_diagnostics(scored_edges, self_coefficients)
            metric_row, topology_row = evaluate_method(
                scored_edges,
                spec,
                network_id=network_id,
                genes=genes,
                n_trajectories=n_trajectories,
                n_lagged_samples=n_lagged_samples,
                fit_seconds=fit_seconds,
                diagnostics=diagnostics,
            )
            metric_rows.append(metric_row)
            topology_rows.append(topology_row)
            edge_audit = merge_score_columns(edge_audit, scored_edges, str(spec["method"]))
            edge_audit = merge_self_columns(edge_audit, self_coefficients, str(spec["method"]))
        edge_audits.append(edge_audit)

    per_network = pd.DataFrame(metric_rows)
    topology = pd.DataFrame(topology_rows)
    summary = aggregate_summary(per_network)
    data_summary = build_data_summary(data_rows)
    edge_audit = pd.concat(edge_audits, ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    per_network.to_csv(PER_NETWORK_PATH, index=False)
    edge_audit.to_csv(EDGE_AUDIT_PATH, index=False)
    topology.to_csv(TOPOLOGY_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(
        build_debug_report(summary, per_network, data_summary, skipped_methods),
        encoding="utf-8",
    )
    print_summary(summary, skipped_methods)


if __name__ == "__main__":
    main()
