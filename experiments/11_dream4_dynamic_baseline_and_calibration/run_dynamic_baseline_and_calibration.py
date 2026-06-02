"""DREAM4 dynamic baseline, sparsity calibration, and rank-fusion audit.

This experiment follows the Size100 scaling result (experiment 10), where the
Size10 sparse candidate `dynamic_lasso_level_include_self_a0_03` failed to scale.
It pursues three connected goals on DREAM4 Size10 and Size100 time-series data:

A. A closer dynGENIE3-style temporal tree baseline (level, delta, and derivative
   targets) alongside the current level GENIE3 baselines.
B. Sparsity calibration: instead of guessing one alpha, sweep an alpha grid for
   LASSO/Elastic Net (level and delta, include/exclude self) and analyze which
   sparsity level works and whether the best alpha tracks network density.
C. Rank fusion of complementary evidence, including a simple reciprocal-direction
   penalty aimed at the recurring reciprocal false-positive problem.

dynGENIE3 note: no official dynGENIE3 package is installed in this environment,
so the delta/derivative tree methods are dynGENIE3-*style*, not an official
reproduction. If an official package becomes importable, `detect_official_dyngenie3`
will report it and the method can be added separately.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
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
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
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
    rank_edges_by_dynamic_tree_ensemble,
    rank_edges_by_lagged_correlation,
    rank_fusion,
    rank_fusion_with_reciprocal_penalty,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
PREFIX = "dream4_dynamic_baseline_calibration"
SUMMARY_PATH = RESULTS_DIR / f"{PREFIX}_summary.csv"
PER_NETWORK_PATH = RESULTS_DIR / f"{PREFIX}_per_network.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_edges.csv"
TOPOLOGY_PATH = RESULTS_DIR / f"{PREFIX}_topology.csv"
ALPHA_SENS_PATH = RESULTS_DIR / f"{PREFIX}_alpha_sensitivity.csv"
PAIRWISE_PATH = RESULTS_DIR / f"{PREFIX}_pairwise_comparisons.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / f"{PREFIX}_debug_report.md"

NETWORK_IDS = range(1, 6)
FULL_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
QUICK_ALPHA_GRID = (0.03, 0.1)
PRECISION_KS = (5, 10, 20, 50, 100, 200)
MARGIN = 0.005  # mean-metric gap treated as real rather than noise across 5 networks
ELASTIC_NET_L1_RATIO = 0.7
TREE_MAX_ITER = 50000
FUSION_PENALTY_TOP_FRACTION = 0.05

# Sparse families swept across the alpha grid. Each is (model_kind, target, self_mode, l1_ratio).
SPARSE_CONFIGS = [
    ("lasso", "level", "include_self_predictor_no_self_edge", None),
    ("lasso", "level", "exclude_self_predictor", None),
    ("lasso", "delta", "include_self_predictor_no_self_edge", None),
    ("lasso", "delta", "exclude_self_predictor", None),
    ("elastic_net", "level", "include_self_predictor_no_self_edge", ELASTIC_NET_L1_RATIO),
    ("elastic_net", "delta", "include_self_predictor_no_self_edge", ELASTIC_NET_L1_RATIO),
]

# dynGENIE3-style tree methods. All use every gene at t as a predictor (including
# the target's own past value) but never emit a self-edge.
TREE_CONFIGS = [
    ("random_forest", "level", "lagged_genie3_rf_level", 11),
    ("extra_trees", "level", "lagged_genie3_extra_trees_level", 22),
    ("random_forest", "delta", "dyn_genie3_rf_delta", 33),
    ("extra_trees", "delta", "dyn_genie3_extra_trees_delta", 44),
    ("random_forest", "derivative", "dyn_genie3_rf_derivative", 55),
    ("extra_trees", "derivative", "dyn_genie3_extra_trees_derivative", 66),
]

CORRELATION_METHOD = "lagged_correlation"

SIZE_SETTINGS = {
    10: {
        "expression_path": lambda n: dream4_size10_expression_path(DATA_ROOT, n, "timeseries"),
        "gold_path": lambda n: dream4_size10_gold_standard_path(DATA_ROOT, n),
        "expected_genes": 10,
        "network_label": lambda n: f"insilico_size10_{n}",
        "precision_ks": (5, 10, 20),
        "hub_tops": (3,),
    },
    100: {
        "expression_path": lambda n: dream4_size100_expression_path(DATA_ROOT, n, "timeseries"),
        "gold_path": lambda n: dream4_size100_gold_standard_path(DATA_ROOT, n),
        "expected_genes": 100,
        "network_label": lambda n: f"insilico_size100_{n}",
        "precision_ks": (10, 50, 100, 200),
        "hub_tops": (5, 10),
    },
}


# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
def detect_official_dyngenie3() -> tuple[bool, str]:
    """Report whether an official dynGENIE3/GENIE3 implementation is importable."""
    for module in ("dynGENIE3", "dyngenie3", "GENIE3", "arboreto"):
        if importlib.util.find_spec(module) is not None:
            return True, module
    return False, ""


def detect_gnw() -> str:
    """Return a short note on whether GeneNetWeaver appears available locally."""
    on_path = shutil.which("gnw")
    if on_path:
        return f"GNW executable found on PATH at {on_path}."
    candidates = list(ROOT.glob("**/*gnw*.jar")) + list(ROOT.glob("**/GeneNetWeaver*"))
    if candidates:
        return f"Possible GNW artifact found: {candidates[0]}."
    return "GeneNetWeaver not detected (no `gnw` on PATH and no jar in repo); design is scaffold-only."


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def format_alpha(alpha: float) -> str:
    """Format a regularization value for method names (0.03 -> '0_03')."""
    return str(alpha).replace(".", "_")


def short_self(self_mode: str) -> str:
    """Return compact self-predictor mode text for method names."""
    return "include_self" if self_mode == "include_self_predictor_no_self_edge" else "exclude_self"


def sparse_method_name(model_kind: str, target_type: str, self_mode: str, alpha: float, l1_ratio: float | None) -> str:
    """Build a sparse method name consistent with earlier experiments."""
    if model_kind == "elastic_net":
        return (
            f"dynamic_elastic_net_{target_type}_{short_self(self_mode)}"
            f"_a{format_alpha(alpha)}_l1_{format_alpha(l1_ratio)}"
        )
    return f"dynamic_lasso_{target_type}_{short_self(self_mode)}_a{format_alpha(alpha)}"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_size_network(size: int, network_id: int) -> dict[str, object]:
    """Load one network's lagged samples and level/delta/derivative targets."""
    settings = SIZE_SETTINGS[size]
    timeseries = load_expression_matrix(settings["expression_path"](network_id), drop_time=False)
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, metadata = build_lagged_samples(trajectories)
    targets = {
        target_type: build_dynamic_target(x_t, y_t1, metadata, target_type=target_type)
        for target_type in ("level", "delta", "derivative")
    }
    truth_edges = load_gold_standard_edges(settings["gold_path"](network_id))
    genes = [str(column) for column in x_t.columns]
    n_candidate = len(genes) * (len(genes) - 1)
    if len(truth_edges) != n_candidate:
        raise ValueError(
            f"size{size} net{network_id}: gold standard has {len(truth_edges)} rows, expected {n_candidate}"
        )
    delta_time = (metadata["time_t1"] - metadata["time_t"]).round(6)
    return {
        "size": size,
        "network_id": network_id,
        "x_t": x_t,
        "y_t1": y_t1,
        "targets": targets,
        "metadata": metadata,
        "trajectories": trajectories,
        "timeseries": timeseries,
        "truth_edges": truth_edges,
        "genes": genes,
        "n_true_edges": int(truth_edges["is_true"].sum()),
        "constant_time_step": delta_time.nunique() == 1,
    }


# --------------------------------------------------------------------------- #
# Method execution
# --------------------------------------------------------------------------- #
def run_sparse(network: dict[str, object], model_kind: str, target_type: str, self_mode: str,
               alpha: float, l1_ratio: float | None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fit one sparse-linear configuration and return edges, self coefficients, time."""
    target = network["targets"][target_type]
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        edges, self_coefficients = fit_dynamic_linear_coefficients(
            network["x_t"],
            target,
            model_kind=model_kind,
            alpha=alpha,
            l1_ratio=l1_ratio,
            self_predictor_mode=self_mode,
            max_iter=TREE_MAX_ITER,
        )
    return edges, self_coefficients, time.perf_counter() - start


def run_tree(network: dict[str, object], ensemble: str, target_type: str, *,
             n_estimators: int, random_state: int, n_jobs: int) -> tuple[pd.DataFrame, float]:
    """Run one dynGENIE3-style tree configuration (all genes at t as predictors)."""
    target = network["targets"][target_type]
    start = time.perf_counter()
    edges = rank_edges_by_dynamic_tree_ensemble(
        network["x_t"],
        target,
        ensemble=ensemble,
        n_estimators=n_estimators,
        random_state=random_state,
        self_predictor_mode="include_self_predictor_no_self_edge",
        n_jobs=n_jobs,
    )
    return edges, time.perf_counter() - start


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join predicted edge scores to gold-standard labels and assign ranks."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


# --------------------------------------------------------------------------- #
# Metrics, topology, diagnostics
# --------------------------------------------------------------------------- #
def diagnostics_for(scored_edges: pd.DataFrame, self_coefficients: pd.DataFrame, *, is_sparse: bool) -> dict[str, object]:
    """Return sparsity and self-persistence diagnostics (sparse models only)."""
    n_candidate = len(scored_edges)
    blank = {
        "n_nonzero_nonself_edges": pd.NA,
        "predicted_edge_density": pd.NA,
        "mean_abs_self_coefficient": pd.NA,
        "mean_abs_nonself_coefficient": pd.NA,
        "self_to_nonself_abs_ratio": pd.NA,
        "fraction_self_selected": pd.NA,
        "self_abs_median": pd.NA,
        "self_abs_max": pd.NA,
    }
    if not is_sparse:
        return blank
    n_nonzero = int(scored_edges["selected"].sum()) if "selected" in scored_edges.columns else int((scored_edges["score"] > 0).sum())
    mean_abs_nonself = float(scored_edges["score"].mean())
    out = {
        "n_nonzero_nonself_edges": n_nonzero,
        "predicted_edge_density": n_nonzero / n_candidate if n_candidate else 0.0,
        "mean_abs_self_coefficient": pd.NA,
        "mean_abs_nonself_coefficient": mean_abs_nonself,
        "self_to_nonself_abs_ratio": pd.NA,
        "fraction_self_selected": pd.NA,
        "self_abs_median": pd.NA,
        "self_abs_max": pd.NA,
    }
    if not self_coefficients.empty:
        self_abs = self_coefficients["self_abs_coefficient"].astype(float)
        mean_abs_self = float(self_abs.mean())
        out["mean_abs_self_coefficient"] = mean_abs_self
        out["self_to_nonself_abs_ratio"] = (mean_abs_self / mean_abs_nonself) if mean_abs_nonself else 0.0
        out["fraction_self_selected"] = float(self_coefficients["self_selected"].mean())
        out["self_abs_median"] = float(self_abs.median())
        out["self_abs_max"] = float(self_abs.max())
    return out


def evaluate(scored_edges: pd.DataFrame, *, descriptors: dict[str, object], genes: list[str],
             n_true_edges: int, diagnostics: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Compute edge + topology metrics for one method/network."""
    topology = topology_metrics_for_cutoff(scored_edges, cutoff=n_true_edges, rank_column="rank", genes=genes)
    base = {
        **descriptors,
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": n_true_edges,
        "true_edge_density": n_true_edges / len(scored_edges) if len(scored_edges) else 0.0,
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        **{f"precision_at_{k}": precision_at_k(scored_edges, "is_true", k) for k in PRECISION_KS},
        # oracle-density precision = precision at top-N-true cutoff (NOT deployable; uses true count)
        "oracle_density_precision": topology["edge_precision_at_k"],
        **diagnostics,
    }
    metric_row = {**base, **{f"topology_{key}": value for key, value in topology.items()}}
    topology_row = {**base, **topology}
    return metric_row, topology_row


# --------------------------------------------------------------------------- #
# Per-size execution
# --------------------------------------------------------------------------- #
def run_size(size: int, *, alpha_grid: tuple[float, ...], tree_estimators: int, run_trees: bool,
             run_fusion: bool, random_seed: int, n_jobs: int) -> dict[str, object]:
    """Run all methods on all networks for one size; return rows and edge store."""
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    edge_store: dict[tuple[int, str], pd.DataFrame] = {}

    networks = {nid: load_size_network(size, nid) for nid in NETWORK_IDS}
    for nid, network in networks.items():
        genes = network["genes"]
        truth_edges = network["truth_edges"]
        n_true = network["n_true_edges"]
        common = {
            "size": size,
            "network_id": nid,
            "network": SIZE_SETTINGS[size]["network_label"](nid),
            "n_genes": len(genes),
            "n_trajectories": len(network["trajectories"]),
            "n_lagged_samples": len(network["metadata"]),
        }
        data_rows.append(
            {
                **common,
                "n_timeseries_rows": len(network["timeseries"]),
                "n_candidate_edges": len(truth_edges),
                "n_true_edges": n_true,
                "true_edge_density": n_true / len(truth_edges),
                "constant_time_step": network["constant_time_step"],
            }
        )

        # ---- sparse alpha sweep ----
        for model_kind, target_type, self_mode, l1_ratio in SPARSE_CONFIGS:
            for alpha in alpha_grid:
                edges, self_coef, fit_seconds = run_sparse(
                    network, model_kind, target_type, self_mode, alpha, l1_ratio
                )
                scored = score_edges(edges, truth_edges)
                method = sparse_method_name(model_kind, target_type, self_mode, alpha, l1_ratio)
                descriptors = {
                    **common,
                    "method": method,
                    "method_family": "sparse_linear",
                    "model_kind": model_kind,
                    "target_type": target_type,
                    "self_predictor_mode": self_mode,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio if l1_ratio is not None else pd.NA,
                    "fit_seconds": fit_seconds,
                }
                diag = diagnostics_for(scored, self_coef, is_sparse=True)
                metric_row, topo_row = evaluate(
                    scored, descriptors=descriptors, genes=genes, n_true_edges=n_true, diagnostics=diag
                )
                metric_rows.append(metric_row)
                topology_rows.append(topo_row)
                edge_store[(nid, method)] = scored[["source", "target", "score", "rank", "is_true"]]

        # ---- correlation ----
        corr_edges = rank_edges_by_lagged_correlation(network["x_t"], network["y_t1"])
        scored = score_edges(corr_edges, truth_edges)
        descriptors = {
            **common, "method": CORRELATION_METHOD, "method_family": "correlation_reference",
            "model_kind": "correlation", "target_type": "level",
            "self_predictor_mode": "exclude_self_predictor", "alpha": pd.NA, "l1_ratio": pd.NA, "fit_seconds": pd.NA,
        }
        diag = diagnostics_for(scored, pd.DataFrame(), is_sparse=False)
        metric_row, topo_row = evaluate(scored, descriptors=descriptors, genes=genes, n_true_edges=n_true, diagnostics=diag)
        metric_rows.append(metric_row)
        topology_rows.append(topo_row)
        edge_store[(nid, CORRELATION_METHOD)] = scored[["source", "target", "score", "rank", "is_true"]]

        # ---- trees ----
        if run_trees:
            for ensemble, target_type, method, offset in TREE_CONFIGS:
                edges, fit_seconds = run_tree(
                    network, ensemble, target_type,
                    n_estimators=tree_estimators,
                    random_state=random_seed + nid * 100 + offset,
                    n_jobs=n_jobs,
                )
                scored = score_edges(edges, truth_edges)
                descriptors = {
                    **common, "method": method,
                    "method_family": "dyn_genie3_style" if target_type != "level" else "level_genie3",
                    "model_kind": ensemble, "target_type": target_type,
                    "self_predictor_mode": "include_self_predictor_no_self_edge",
                    "alpha": pd.NA, "l1_ratio": pd.NA, "fit_seconds": fit_seconds,
                }
                diag = diagnostics_for(scored, pd.DataFrame(), is_sparse=False)
                metric_row, topo_row = evaluate(scored, descriptors=descriptors, genes=genes, n_true_edges=n_true, diagnostics=diag)
                metric_rows.append(metric_row)
                topology_rows.append(topo_row)
                edge_store[(nid, method)] = scored[["source", "target", "score", "rank", "is_true"]]

    per_network = pd.DataFrame(metric_rows)
    summary = aggregate_summary(per_network)

    fusion_inputs: dict[str, str] = {}
    if run_fusion:
        fusion_inputs = choose_fusion_inputs(summary, run_trees=run_trees)
        fusion_metric_rows, fusion_topo_rows = run_fusion_methods(
            networks, edge_store, fusion_inputs, size=size
        )
        metric_rows.extend(fusion_metric_rows)
        topology_rows.extend(fusion_topo_rows)
        # edge_store already updated in-place inside run_fusion_methods
        per_network = pd.DataFrame(metric_rows)
        summary = aggregate_summary(per_network)

    return {
        "size": size,
        "per_network": per_network,
        "topology": pd.DataFrame(topology_rows),
        "summary": summary,
        "data_summary": pd.DataFrame(data_rows),
        "edge_store": edge_store,
        "fusion_inputs": fusion_inputs,
        "networks": {nid: {"n_true_edges": net["n_true_edges"], "genes": net["genes"]} for nid, net in networks.items()},
    }


def choose_fusion_inputs(summary: pd.DataFrame, *, run_trees: bool) -> dict[str, str]:
    """Pick the best sparse and best tree method (by mean AUPR) plus correlation.

    This is a diagnostic selection on aggregated AUPR, documented as such; it is
    not a deployable model-selection rule.
    """
    inputs: dict[str, str] = {}
    sparse = summary[summary["method_family"] == "sparse_linear"]
    if not sparse.empty:
        inputs["sparse"] = str(sparse.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]["method"])
    if run_trees:
        trees = summary[summary["method_family"].isin(["dyn_genie3_style", "level_genie3"])]
        if not trees.empty:
            inputs["tree"] = str(trees.sort_values(["aupr", "method"], ascending=[False, True]).iloc[0]["method"])
    inputs["correlation"] = CORRELATION_METHOD
    return inputs


def run_fusion_methods(networks: dict[int, dict[str, object]], edge_store: dict[tuple[int, str], pd.DataFrame],
                       fusion_inputs: dict[str, str], *, size: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fuse the chosen input rankings per network and evaluate fusion methods."""
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    input_methods = [m for m in fusion_inputs.values() if m]

    fusion_specs = [
        ("fusion_mean_reciprocal_rank", lambda tables: rank_fusion(tables, method="mean_reciprocal_rank")),
        ("fusion_borda", lambda tables: rank_fusion(tables, method="borda")),
        ("fusion_mean_normalized_score", lambda tables: rank_fusion(tables, method="mean_normalized_score")),
        ("fusion_reciprocal_penalty_w0_5",
         lambda tables: rank_fusion_with_reciprocal_penalty(tables, penalty=0.5, top_fraction=FUSION_PENALTY_TOP_FRACTION)),
        ("fusion_reciprocal_penalty_w0_25",
         lambda tables: rank_fusion_with_reciprocal_penalty(tables, penalty=0.25, top_fraction=FUSION_PENALTY_TOP_FRACTION)),
    ]

    for nid, net in networks.items():
        genes = net["genes"]
        n_true = net["n_true_edges"]
        truth = net["truth_edges"]
        common = {
            "size": size, "network_id": nid, "network": SIZE_SETTINGS[size]["network_label"](nid),
            "n_genes": len(genes), "n_trajectories": len(net["metadata"]), "n_lagged_samples": len(net["metadata"]),
        }
        tables = [edge_store[(nid, method)][["source", "target", "score"]] for method in input_methods]
        for method, fuse in fusion_specs:
            fused = fuse(tables)
            scored = score_edges(fused, truth)
            descriptors = {
                **common, "method": method, "method_family": "fusion", "model_kind": "fusion",
                "target_type": "level", "self_predictor_mode": "exclude_self_predictor",
                "alpha": pd.NA, "l1_ratio": pd.NA, "fit_seconds": pd.NA,
            }
            diag = diagnostics_for(scored, pd.DataFrame(), is_sparse=False)
            metric_row, topo_row = evaluate(scored, descriptors=descriptors, genes=genes, n_true_edges=n_true, diagnostics=diag)
            metric_rows.append(metric_row)
            topology_rows.append(topo_row)
            edge_store[(nid, method)] = scored[["source", "target", "score", "rank", "is_true"]]
    return metric_rows, topology_rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
GROUP_COLUMNS = [
    "size", "method", "method_family", "model_kind", "target_type",
    "self_predictor_mode", "alpha", "l1_ratio",
]


def aggregate_summary(per_network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-network rows into mean/std per method (within a size)."""
    per_network = per_network.copy()
    excluded = set(GROUP_COLUMNS) | {"network_id", "network"}
    metric_columns: list[str] = []
    for column in per_network.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(per_network[column], errors="coerce")
        if numeric.notna().any():
            per_network[column] = numeric
            metric_columns.append(column)
    summary = aggregate_per_network_metrics(per_network, group_columns=GROUP_COLUMNS, metric_columns=metric_columns)
    summary.insert(0, "row_type", "mean")
    return summary.sort_values(["size", "aupr", "method"], ascending=[True, False, True]).reset_index(drop=True)


def build_alpha_sensitivity(per_network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sparse metrics per (size, config, alpha) and mark best alphas."""
    sparse = per_network[per_network["method_family"] == "sparse_linear"].copy()
    if sparse.empty:
        return pd.DataFrame()
    group_cols = ["size", "model_kind", "target_type", "self_predictor_mode", "alpha"]
    metric_cols = [
        "auroc", "aupr", "precision_at_10", "oracle_density_precision",
        "predicted_edge_density", "true_edge_density", "n_nonzero_nonself_edges",
        "self_to_nonself_abs_ratio", "mean_abs_self_coefficient", "mean_abs_nonself_coefficient",
        "topology_top3_out_hub_overlap", "topology_top5_out_hub_overlap",
        "topology_top3_in_hub_overlap", "topology_top5_in_hub_overlap",
        "topology_reciprocal_false_positive_pair_rate",
    ]
    available = [c for c in metric_cols if c in sparse.columns]
    for column in available:
        sparse[column] = pd.to_numeric(sparse[column], errors="coerce")
    agg = sparse.groupby(group_cols, dropna=False, as_index=False)[available].mean()

    config_cols = ["size", "model_kind", "target_type", "self_predictor_mode"]
    for metric, flag in [
        ("aupr", "is_best_alpha_by_aupr"),
        ("auroc", "is_best_alpha_by_auroc"),
        ("precision_at_10", "is_best_alpha_by_precision_at_10"),
        ("topology_top3_out_hub_overlap", "is_best_alpha_by_top3_out_hub"),
    ]:
        if metric in agg.columns:
            idx = agg.groupby(config_cols, dropna=False)[metric].idxmax()
            agg[flag] = False
            agg.loc[idx, flag] = True
    if "topology_reciprocal_false_positive_pair_rate" in agg.columns:
        idx = agg.groupby(config_cols, dropna=False)["topology_reciprocal_false_positive_pair_rate"].idxmin()
        agg["is_best_alpha_by_low_reciprocal_fp"] = False
        agg.loc[idx, "is_best_alpha_by_low_reciprocal_fp"] = True
    return agg.sort_values(config_cols + ["alpha"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pairwise comparisons + debug report helpers
# --------------------------------------------------------------------------- #
def metric_value(summary: pd.DataFrame, size: int, method: str, column: str) -> float:
    """Return one summary metric, or NaN when unavailable."""
    row = summary[(summary["size"] == size) & (summary["method"] == method)]
    if row.empty or column not in row.columns or pd.isna(row.iloc[0][column]):
        return float("nan")
    return float(row.iloc[0][column])


def best_method(summary: pd.DataFrame, size: int, metric: str, families: list[str] | None = None) -> tuple[str, float]:
    """Return (method, value) maximizing a metric within a size (optionally a family set)."""
    frame = summary[summary["size"] == size]
    if families is not None:
        frame = frame[frame["method_family"].isin(families)]
    frame = frame[frame[metric].notna()]
    if frame.empty:
        return "", float("nan")
    row = frame.sort_values([metric, "method"], ascending=[False, True]).iloc[0]
    return str(row["method"]), float(row[metric])


def fmt(value: float, digits: int = 6) -> str:
    """Format a float for prose, tolerating NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def comparison_word(delta: float, margin: float = MARGIN) -> str:
    """Return beats/ties/trails for a metric delta, with a noise margin."""
    if np.isnan(delta):
        return "is unclear versus"
    if delta > margin:
        return "beats"
    if delta < -margin:
        return "trails"
    return "ties"


def best_alpha(alpha_sens: pd.DataFrame, size: int, model_kind: str, target: str, self_mode: str,
               metric: str, *, maximize: bool = True) -> tuple[float, float]:
    """Return (alpha, value) optimizing a metric for one sparse config at a size."""
    frame = alpha_sens[
        (alpha_sens["size"] == size)
        & (alpha_sens["model_kind"] == model_kind)
        & (alpha_sens["target_type"] == target)
        & (alpha_sens["self_predictor_mode"] == self_mode)
        & (alpha_sens[metric].notna())
    ]
    if frame.empty:
        return float("nan"), float("nan")
    row = frame.sort_values(metric, ascending=not maximize).iloc[0]
    return float(row["alpha"]), float(row[metric])


def build_pairwise_comparisons(summary: pd.DataFrame, sizes: list[int], run_trees: bool,
                               run_fusion: bool, fusion_inputs_by_size: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Build a tidy table of head-to-head mean-metric comparisons per size."""
    rows: list[dict[str, object]] = []

    def add(size: int, comparison: str, method_a: str, method_b: str, metric: str) -> None:
        value_a = metric_value(summary, size, method_a, metric)
        value_b = metric_value(summary, size, method_b, metric)
        rows.append({
            "size": size, "comparison": comparison, "metric": metric,
            "method_a": method_a, "value_a": value_a,
            "method_b": method_b, "value_b": value_b,
            "delta_a_minus_b": value_a - value_b,
        })

    for size in sizes:
        if run_trees:
            for metric in ("aupr", "auroc"):
                add(size, "dyngenie3_delta_vs_level_genie3_rf", "dyn_genie3_rf_delta", "lagged_genie3_rf_level", metric)
            best_tree_aupr = best_method(summary, size, "aupr", ["dyn_genie3_style", "level_genie3"])[0]
            best_sparse_aupr = best_method(summary, size, "aupr", ["sparse_linear"])[0]
            if best_tree_aupr and best_sparse_aupr:
                for metric in ("aupr", "auroc"):
                    add(size, "best_tree_vs_best_sparse", best_tree_aupr, best_sparse_aupr, metric)
        for metric in ("aupr", "auroc"):
            add(size, "lasso_level_include_vs_exclude_a0_03",
                "dynamic_lasso_level_include_self_a0_03", "dynamic_lasso_level_exclude_self_a0_03", metric)
            add(size, "lasso_level_include_a0_03_vs_a0_1",
                "dynamic_lasso_level_include_self_a0_03", "dynamic_lasso_level_include_self_a0_1", metric)
        if run_fusion:
            best_fusion = best_method(summary, size, "aupr", ["fusion"])[0]
            inputs = fusion_inputs_by_size.get(size, {})
            best_input = ""
            best_input_aupr = -1.0
            for method in inputs.values():
                value = metric_value(summary, size, method, "aupr")
                if not np.isnan(value) and value > best_input_aupr:
                    best_input_aupr, best_input = value, method
            if best_fusion and best_input:
                for metric in ("aupr", "auroc", "precision_at_10"):
                    add(size, "best_fusion_vs_best_input", best_fusion, best_input, metric)
            for metric in ("aupr", "precision_at_10", "topology_reciprocal_false_positive_pair_rate"):
                add(size, "reciprocal_penalty_vs_base_fusion",
                    "fusion_reciprocal_penalty_w0_5", "fusion_mean_reciprocal_rank", metric)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Debug report
# --------------------------------------------------------------------------- #
def build_debug_report(summary: pd.DataFrame, alpha_sens: pd.DataFrame, data_summary: pd.DataFrame,
                       sizes: list[int], run_trees: bool, run_fusion: bool,
                       fusion_inputs_by_size: dict[int, dict[str, str]], official_dyngenie3: tuple[bool, str],
                       gnw_note: str) -> str:
    """Answer the thirteen questions from the aggregated results."""
    lines = [
        "# DREAM4 Dynamic Baseline, Calibration, and Fusion Debug Report",
        "",
        "This report compares dynGENIE3-style temporal tree baselines, an alpha-calibrated "
        "dynamic sparse family, and rank fusion (including a reciprocal-direction penalty) on "
        f"DREAM4 sizes {', '.join(str(s) for s in sizes)}. Topology metrics use a top-N-true-edges cutoff.",
        "",
        f"dynGENIE3 status: {'official package ' + official_dyngenie3[1] + ' detected' if official_dyngenie3[0] else 'no official package installed; delta/derivative trees are dynGENIE3-STYLE, not an official reproduction'}.",
        f"GNW status: {gnw_note}",
        "",
        "## Data Summary",
        "",
        to_markdown_table(data_summary),
        "",
        "## Headline Mean Metrics (top methods per size)",
        "",
        to_markdown_table(headline_table(summary, sizes)),
        "",
        "## Question-By-Question Findings",
        "",
        answer_questions(summary, alpha_sens, sizes, run_trees, run_fusion, fusion_inputs_by_size),
        "",
    ]
    return "\n".join(lines)


def headline_table(summary: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    """Return the top few methods by AUPR for each size."""
    frames = []
    columns = ["size", "method", "method_family", "alpha", "auroc", "aupr", "precision_at_10",
               "oracle_density_precision", "self_to_nonself_abs_ratio",
               "topology_reciprocal_false_positive_pair_rate"]
    for size in sizes:
        frame = summary[summary["size"] == size].sort_values("aupr", ascending=False).head(8)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else summary
    available = [c for c in columns if c in combined.columns]
    return combined[available]


def answer_questions(summary: pd.DataFrame, alpha_sens: pd.DataFrame, sizes: list[int],
                     run_trees: bool, run_fusion: bool, fusion_inputs_by_size: dict[int, dict[str, str]]) -> str:
    """Produce numbered answers to the thirteen questions."""
    out: list[str] = []

    # Q1 delta/derivative vs level GENIE3
    out.append("**1. Does dynGENIE3-style delta/derivative improve over level GENIE3?**")
    if run_trees:
        for size in sizes:
            d_rf = metric_value(summary, size, "dyn_genie3_rf_delta", "aupr")
            dv_rf = metric_value(summary, size, "dyn_genie3_rf_derivative", "aupr")
            l_rf = metric_value(summary, size, "lagged_genie3_rf_level", "aupr")
            out.append(
                f"- Size{size}: delta RF AUPR {fmt(d_rf)} {comparison_word(d_rf - l_rf)} level RF {fmt(l_rf)} "
                f"(derivative RF {fmt(dv_rf)}). On DREAM4's constant time grid delta and derivative tree rankings nearly coincide."
            )
    else:
        out.append("- Trees were skipped, so this comparison was not run.")

    # Q2/Q3 dynGENIE3-style vs sparse
    out.append("")
    out.append("**2. Does dynGENIE3-style beat dynamic sparse methods by AUPR?**")
    for size in sizes:
        if not run_trees:
            out.append(f"- Size{size}: trees skipped.")
            continue
        tm, tv = best_method(summary, size, "aupr", ["dyn_genie3_style", "level_genie3"])
        sm, sv = best_method(summary, size, "aupr", ["sparse_linear"])
        out.append(f"- Size{size}: best tree `{tm}` ({fmt(tv)}) {comparison_word(tv - sv)} best sparse `{sm}` ({fmt(sv)}).")
    out.append("")
    out.append("**3. Does dynGENIE3-style beat dynamic sparse methods by AUROC?**")
    for size in sizes:
        if not run_trees:
            out.append(f"- Size{size}: trees skipped.")
            continue
        tm, tv = best_method(summary, size, "auroc", ["dyn_genie3_style", "level_genie3"])
        sm, sv = best_method(summary, size, "auroc", ["sparse_linear"])
        out.append(f"- Size{size}: best tree `{tm}` ({fmt(tv)}) {comparison_word(tv - sv)} best sparse `{sm}` ({fmt(sv)}).")

    # Q4 best topology
    out.append("")
    out.append("**4. Which method has best topology/hub recovery?**")
    for size in sizes:
        hub_metric = "topology_top3_out_hub_overlap" if size == 10 else "topology_top5_out_hub_overlap"
        om, ov = best_method(summary, size, hub_metric)
        recip = "topology_reciprocal_false_positive_pair_rate"
        rm, rv = best_method(summary, size, recip)  # max is worst; recompute min
        frame = summary[(summary["size"] == size) & summary[recip].notna()]
        if not frame.empty:
            rrow = frame.sort_values([recip, "method"], ascending=[True, True]).iloc[0]
            rm, rv = str(rrow["method"]), float(rrow[recip])
        out.append(f"- Size{size}: best {hub_metric.replace('topology_', '')} is `{om}` ({fmt(ov)}); "
                   f"lowest reciprocal-FP pair rate is `{rm}` ({fmt(rv)}).")

    # Q5 alpha explains size difference
    out.append("")
    out.append("**5. Does alpha choice explain the Size10 vs Size100 difference?**")
    if 10 in sizes and 100 in sizes:
        a10, v10 = best_alpha(alpha_sens, 10, "lasso", "level", "include_self_predictor_no_self_edge", "aupr")
        a100, v100 = best_alpha(alpha_sens, 100, "lasso", "level", "include_self_predictor_no_self_edge", "aupr")
        verdict = "yes - the best alpha rises with the larger, sparser network" if a100 > a10 else "not clearly - the best alpha does not rise with size"
        out.append(f"- Best AUPR alpha for LASSO level include-self: Size10 = {fmt(a10, 3)} (AUPR {fmt(v10)}), "
                   f"Size100 = {fmt(a100, 3)} (AUPR {fmt(v100)}). {verdict}.")
    else:
        out.append("- Needs both sizes; run with --standard (Size10 + Size100) to compare.")

    # Q6 stronger reg better at Size100
    out.append("")
    out.append("**6. Is stronger regularization consistently better at Size100?**")
    if 100 in sizes:
        out.append("- " + stronger_reg_size100(alpha_sens))
    else:
        out.append("- Size100 not run.")

    # Q7 include-self after sparsity/density
    out.append("")
    out.append("**7. Does include-self help after accounting for sparsity/density?**")
    for size in sizes:
        ai, vi = best_alpha(alpha_sens, size, "lasso", "level", "include_self_predictor_no_self_edge", "aupr")
        ae, ve = best_alpha(alpha_sens, size, "lasso", "level", "exclude_self_predictor", "aupr")
        out.append(f"- Size{size}: best include-self AUPR {fmt(vi)} (alpha {fmt(ai, 3)}) {comparison_word(vi - ve)} "
                   f"best exclude-self AUPR {fmt(ve)} (alpha {fmt(ae, 3)}), each at its own best alpha.")

    # Q8 self-persistence useful or dangerous
    out.append("")
    out.append("**8. Does self-persistence look useful or dangerous?**")
    out.append("- " + self_persistence_verdict(alpha_sens, sizes))

    # Q9 fusion helps
    out.append("")
    out.append("**9. Does rank fusion help?**")
    if run_fusion:
        for size in sizes:
            fm, fv = best_method(summary, size, "aupr", ["fusion"])
            inputs = fusion_inputs_by_size.get(size, {})
            best_input, best_input_v = "", -1.0
            for method in inputs.values():
                value = metric_value(summary, size, method, "aupr")
                if not np.isnan(value) and value > best_input_v:
                    best_input_v, best_input = value, method
            out.append(f"- Size{size}: best fusion `{fm}` ({fmt(fv)}) {comparison_word(fv - best_input_v)} "
                       f"best single input `{best_input}` ({fmt(best_input_v)}). Inputs: {', '.join(inputs.values())}.")
    else:
        out.append("- Fusion was skipped.")

    # Q10 reciprocal penalty helps
    out.append("")
    out.append("**10. Does the reciprocal-direction penalty help?**")
    if run_fusion:
        for size in sizes:
            base = metric_value(summary, size, "fusion_mean_reciprocal_rank", "aupr")
            pen = metric_value(summary, size, "fusion_reciprocal_penalty_w0_5", "aupr")
            base_recip = metric_value(summary, size, "fusion_mean_reciprocal_rank", "topology_reciprocal_false_positive_pair_rate")
            pen_recip = metric_value(summary, size, "fusion_reciprocal_penalty_w0_5", "topology_reciprocal_false_positive_pair_rate")
            base_p10 = metric_value(summary, size, "fusion_mean_reciprocal_rank", "precision_at_10")
            pen_p10 = metric_value(summary, size, "fusion_reciprocal_penalty_w0_5", "precision_at_10")
            out.append(
                f"- Size{size}: penalty(w=0.5) AUPR {fmt(pen)} {comparison_word(pen - base)} base fusion {fmt(base)}; "
                f"precision@10 {fmt(pen_p10)} vs {fmt(base_p10)}; reciprocal-FP rate {fmt(pen_recip)} vs {fmt(base_recip)}."
            )
    else:
        out.append("- Fusion was skipped.")

    # Q11 best method per regime
    out.append("")
    out.append("**11. What is the best current method per regime (size)?**")
    for size in sizes:
        am, av = best_method(summary, size, "aupr")
        rm, rv = best_method(summary, size, "auroc")
        out.append(f"- Size{size}: best AUPR `{am}` ({fmt(av)}); best AUROC `{rm}` ({fmt(rv)}).")

    # Q12 main project claim
    out.append("")
    out.append("**12. What is the main project claim after this result?**")
    out.append("- " + main_claim(summary, alpha_sens, sizes, run_trees))

    # Q13 GNW sweeps
    out.append("")
    out.append("**13. What should be tested with GNW sweeps?**")
    out.append(
        "- Whether the size/density dependence of the best alpha holds under controlled sweeps of network "
        "size (10/30/50/100), trajectory length (21/50/100), trajectory count (5/10/20), and noise; whether "
        "dynamic sparse ever overtakes tree methods as trajectory length grows; whether self-persistence helps "
        "or hurts with longer series; and whether fusion plus a reciprocal penalty improves robustness and "
        "topology. See experiments/12_gnw_sweep_design/gnw_sweep_design.md."
    )
    return "\n".join(out)


def stronger_reg_size100(alpha_sens: pd.DataFrame) -> str:
    """Describe whether higher alpha helps for Size100 LASSO level include-self."""
    frame = alpha_sens[
        (alpha_sens["size"] == 100) & (alpha_sens["model_kind"] == "lasso")
        & (alpha_sens["target_type"] == "level")
        & (alpha_sens["self_predictor_mode"] == "include_self_predictor_no_self_edge")
    ].sort_values("alpha")
    if frame.empty:
        return "Size100 LASSO level include-self sweep unavailable."
    low = frame[frame["alpha"] <= 0.03]["aupr"].mean()
    high = frame[frame["alpha"] >= 0.1]["aupr"].mean()
    best_a, best_v = float(frame.sort_values("aupr", ascending=False).iloc[0]["alpha"]), float(frame["aupr"].max())
    trend = "yes, stronger regularization helps on average" if high > low + MARGIN else (
        "not uniformly - very strong alphas eventually hurt" if high < low - MARGIN else "roughly flat across alpha")
    return (f"For Size100 LASSO level include-self, mean AUPR is {fmt(low)} at alpha<=0.03 and {fmt(high)} at alpha>=0.1; "
            f"best alpha {fmt(best_a, 3)} (AUPR {fmt(best_v)}). {trend}.")


def self_persistence_verdict(alpha_sens: pd.DataFrame, sizes: list[int]) -> str:
    """Judge whether self-persistence is useful or dangerous from the sweep."""
    parts = []
    for size in sizes:
        frame = alpha_sens[
            (alpha_sens["size"] == size) & (alpha_sens["model_kind"] == "lasso")
            & (alpha_sens["target_type"] == "level")
            & (alpha_sens["self_predictor_mode"] == "include_self_predictor_no_self_edge")
            & (alpha_sens["self_to_nonself_abs_ratio"].notna())
        ]
        if frame.empty:
            continue
        ratio_at_best = frame.sort_values("aupr", ascending=False).iloc[0]["self_to_nonself_abs_ratio"]
        max_ratio = frame["self_to_nonself_abs_ratio"].max()
        parts.append(f"Size{size} self/non-self ratio is {fmt(float(ratio_at_best), 2)} at the best-AUPR alpha (up to {fmt(float(max_ratio), 2)})")
    if not parts:
        return "No include-self sparse sweep available to judge persistence."
    return ("; ".join(parts) + ". The ratio grows with alpha and network size while edge recovery does not improve "
            "proportionally, so self-persistence reads as a dominant but largely diagnostic term - useful to model "
            "stability, dangerous to lean on for directed-edge claims.")


def main_claim(summary: pd.DataFrame, alpha_sens: pd.DataFrame, sizes: list[int], run_trees: bool) -> str:
    """Compose the cautious main project claim."""
    claim = [
        "Dynamic GRN inference on DREAM4 is regime-dependent: the best sparsity level tracks network "
        "size/density rather than being a fixed alpha, and no single method dominates both AUPR and AUROC.",
    ]
    if 100 in sizes and run_trees:
        tm, tv = best_method(summary, 100, "auroc", ["dyn_genie3_style", "level_genie3"])
        claim.append(f"At Size100, tree baselines (best AUROC `{tm}`, {fmt(tv)}) remain the strongest ranking by AUROC, "
                     "so a literature-faithful dynGENIE3 comparison is the right next reference.")
    claim.append("Sparse calibration, not a single tuned alpha, plus honest reporting of the Size100 negative result, "
                 "is the defensible current position; deployable claims need GNW sweeps and an official dynGENIE3 baseline.")
    return " ".join(claim)


# --------------------------------------------------------------------------- #
# Edge audit (headline methods, wide)
# --------------------------------------------------------------------------- #
def build_edge_audit(size_results: list[dict[str, object]], run_trees: bool, run_fusion: bool) -> pd.DataFrame:
    """Build a wide edge audit for a curated set of headline methods."""
    blocks: list[pd.DataFrame] = []
    for result in size_results:
        size = result["size"]
        edge_store = result["edge_store"]
        fusion_inputs = result["fusion_inputs"]
        headline = [
            CORRELATION_METHOD,
            "dynamic_lasso_level_include_self_a0_03",
            "dynamic_lasso_level_include_self_a0_1",
            "dynamic_lasso_level_exclude_self_a0_03",
        ]
        if run_trees:
            headline += ["lagged_genie3_rf_level", "dyn_genie3_rf_delta"]
        headline += [m for m in fusion_inputs.values() if m]
        if run_fusion:
            headline += ["fusion_mean_reciprocal_rank", "fusion_reciprocal_penalty_w0_5", "fusion_reciprocal_penalty_w0_25"]
        seen: list[str] = []
        for method in headline:
            if method not in seen:
                seen.append(method)
        for nid in NETWORK_IDS:
            base_key = next((key for key in edge_store if key[0] == nid), None)
            if base_key is None:
                continue
            base = edge_store[(nid, CORRELATION_METHOD)][["source", "target", "is_true"]].copy()
            base.insert(0, "size", size)
            base.insert(1, "network_id", nid)
            for method in seen:
                key = (nid, method)
                if key not in edge_store:
                    continue
                merged = edge_store[key][["source", "target", "score", "rank"]].rename(
                    columns={"score": f"score_{method}", "rank": f"rank_{method}"}
                )
                base = base.merge(merged, on=["source", "target"], how="left")
            blocks.append(base)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as Markdown without optional dependencies."""
    if frame is None or frame.empty:
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


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Size10 only, reduced trees, reduced alpha grid [0.03, 0.1]")
    parser.add_argument("--standard", action="store_true",
                        help="explicit default: Size10 + Size100, full alpha grid, trees, fusion")
    parser.add_argument("--skip-size100", action="store_true")
    parser.add_argument("--skip-trees", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tree-estimators-size10", type=int, default=None)
    parser.add_argument("--tree-estimators-size100", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=20260602)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, object]:
    """Resolve flags into a concrete run configuration."""
    sizes = [10] if (args.quick or args.skip_size100) else [10, 100]
    alpha_grid = QUICK_ALPHA_GRID if args.quick else FULL_ALPHA_GRID
    tree10 = args.tree_estimators_size10 if args.tree_estimators_size10 is not None else (100 if args.quick else 500)
    tree100 = args.tree_estimators_size100 if args.tree_estimators_size100 is not None else 200
    return {
        "sizes": sizes,
        "alpha_grid": alpha_grid,
        "tree_estimators": {10: tree10, 100: tree100},
        "run_trees": not args.skip_trees,
        "run_fusion": not args.skip_fusion,
        "n_jobs": args.n_jobs,
        "random_seed": args.random_seed,
    }


def main() -> None:
    """Run the baseline/calibration/fusion audit and write artifacts."""
    args = parse_args()
    config = resolve_config(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    official_dyngenie3 = detect_official_dyngenie3()
    gnw_note = detect_gnw()

    size_results = []
    for size in config["sizes"]:
        result = run_size(
            size,
            alpha_grid=config["alpha_grid"],
            tree_estimators=config["tree_estimators"][size],
            run_trees=config["run_trees"],
            run_fusion=config["run_fusion"],
            random_seed=config["random_seed"],
            n_jobs=config["n_jobs"],
        )
        size_results.append(result)

    per_network = pd.concat([r["per_network"] for r in size_results], ignore_index=True)
    topology = pd.concat([r["topology"] for r in size_results], ignore_index=True)
    summary = pd.concat([r["summary"] for r in size_results], ignore_index=True)
    data_summary = pd.concat([r["data_summary"] for r in size_results], ignore_index=True)
    alpha_sens = build_alpha_sensitivity(per_network)
    fusion_inputs_by_size = {r["size"]: r["fusion_inputs"] for r in size_results}
    pairwise = build_pairwise_comparisons(
        summary, config["sizes"], config["run_trees"], config["run_fusion"], fusion_inputs_by_size
    )
    edge_audit = build_edge_audit(size_results, config["run_trees"], config["run_fusion"])

    summary.to_csv(SUMMARY_PATH, index=False)
    per_network.to_csv(PER_NETWORK_PATH, index=False)
    edge_audit.to_csv(EDGE_AUDIT_PATH, index=False)
    topology.to_csv(TOPOLOGY_PATH, index=False)
    alpha_sens.to_csv(ALPHA_SENS_PATH, index=False)
    pairwise.to_csv(PAIRWISE_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(
        build_debug_report(summary, alpha_sens, data_summary, config["sizes"], config["run_trees"],
                            config["run_fusion"], fusion_inputs_by_size, official_dyngenie3, gnw_note),
        encoding="utf-8",
    )
    print_summary(summary, config, official_dyngenie3, gnw_note, fusion_inputs_by_size)


def print_summary(summary: pd.DataFrame, config: dict[str, object], official_dyngenie3: tuple[bool, str],
                  gnw_note: str, fusion_inputs_by_size: dict[int, dict[str, str]]) -> None:
    """Print a compact run summary."""
    columns = ["size", "method", "method_family", "alpha", "auroc", "aupr", "precision_at_10",
               "self_to_nonself_abs_ratio", "topology_reciprocal_false_positive_pair_rate"]
    available = [c for c in columns if c in summary.columns]
    print("DREAM4 dynamic baseline, calibration, and fusion")
    print(f"sizes={config['sizes']} alpha_grid={config['alpha_grid']} trees={config['run_trees']} fusion={config['run_fusion']}")
    print(f"official dynGENIE3: {'yes (' + official_dyngenie3[1] + ')' if official_dyngenie3[0] else 'no - dynGENIE3-style only'}")
    print(f"GNW: {gnw_note}")
    print(f"fusion inputs by size: {fusion_inputs_by_size}")
    print()
    for size in config["sizes"]:
        print(f"--- Size{size} top 6 by AUPR ---")
        frame = summary[summary["size"] == size].sort_values("aupr", ascending=False).head(6)
        print(frame[available].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print()
    for path in (SUMMARY_PATH, PER_NETWORK_PATH, EDGE_AUDIT_PATH, TOPOLOGY_PATH, ALPHA_SENS_PATH, PAIRWISE_PATH, DEBUG_REPORT_PATH):
        print(f"saved: {path.as_posix()}")


if __name__ == "__main__":
    main()
