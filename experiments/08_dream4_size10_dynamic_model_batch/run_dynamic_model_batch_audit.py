"""Broad dynamic-model audit for DREAM4 Size10 time-series data."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
    moving_average_smooth_trajectories,
    split_trajectories_by_time_reset,
    trajectory_bootstrap_indices,
)
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k, topology_metrics_for_cutoff
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_dynamic_correlation,
    rank_edges_by_dynamic_elastic_net,
    rank_edges_by_dynamic_lasso,
    rank_edges_by_dynamic_mlp_permutation,
    rank_edges_by_dynamic_tree_ensemble,
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
    rank_edges_by_lasso,
    rank_fusion,
    summarize_resampled_dynamic_scores,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_dynamic_model_batch_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_size10_dynamic_model_batch_edges.csv"
TOPOLOGY_PATH = RESULTS_DIR / "dream4_size10_dynamic_model_batch_topology.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_size10_dynamic_model_batch_debug_report.md"

TARGET_TYPES = ("level", "delta", "derivative")
SELF_MODES = ("exclude_self_predictor", "include_self_predictor_no_self_edge")
LASSO_ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3)
MLP_ALPHAS = (0.001, 0.01, 0.1)
PRIMARY_TOPOLOGY_CUTOFF = "top_n_true_edges"

Ranker = Callable[[], pd.DataFrame]


def run_network(
    network_id: int,
    *,
    n_estimators: int,
    stability_estimators: int,
    n_resamples: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Run all dynamic model batch methods for one Size10 network."""
    truth_edges = load_gold_standard_edges(dream4_size10_gold_standard_path(DATA_ROOT, network_id))
    raw_timeseries = load_expression_matrix(
        dream4_size10_expression_path(DATA_ROOT, network_id, "timeseries"),
        drop_time=False,
    )
    raw_trajectories = split_trajectories_by_time_reset(raw_timeseries)
    x_raw, y_raw, metadata = build_lagged_samples(raw_trajectories)
    targets = {
        target_type: build_dynamic_target(x_raw, y_raw, metadata, target_type=target_type)
        for target_type in TARGET_TYPES
    }

    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    scored_by_method: dict[str, pd.DataFrame] = {}
    score_tables_for_fusion: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    seed = random_seed + network_id * 1000

    same_time_expression = raw_timeseries.drop(columns=["Time"])
    same_time_configs = same_time_method_configs(same_time_expression, n_estimators, seed, n_jobs)
    for config in same_time_configs:
        row, scored_edges, topology = run_method_config(
            config,
            truth_edges,
            network_id,
            n_trajectories=len(raw_trajectories),
            n_lagged_samples=len(metadata),
        )
        metric_rows.append(row)
        topology_rows.append(topology)
        edge_audit = merge_score_columns(edge_audit, scored_edges, str(config["method"]))
        scored_by_method[str(config["method"])] = scored_edges

    for config in dynamic_method_configs(x_raw, targets, n_estimators, seed, n_jobs):
        row, scored_edges, topology = run_method_config(
            config,
            truth_edges,
            network_id,
            n_trajectories=len(raw_trajectories),
            n_lagged_samples=len(metadata),
        )
        metric_rows.append(row)
        topology_rows.append(topology)
        method = str(config["method"])
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)
        scored_by_method[method] = scored_edges
        if method in fusion_candidate_method_names():
            score_tables_for_fusion[method] = scored_edges[["source", "target", "score"]].copy()

    stability_rows, stability_topology, edge_audit, stability_score_tables = run_stability_subset(
        x_raw,
        targets,
        metadata,
        truth_edges,
        edge_audit,
        network_id,
        n_trajectories=len(raw_trajectories),
        n_lagged_samples=len(metadata),
        n_resamples=n_resamples,
        stability_estimators=stability_estimators,
        random_seed=seed,
        n_jobs=n_jobs,
    )
    metric_rows.extend(stability_rows)
    topology_rows.extend(stability_topology)
    score_tables_for_fusion.update(stability_score_tables)

    fusion_rows, fusion_topology, edge_audit = run_rank_fusions(
        score_tables_for_fusion,
        truth_edges,
        edge_audit,
        network_id,
        n_trajectories=len(raw_trajectories),
        n_lagged_samples=len(metadata),
    )
    metric_rows.extend(fusion_rows)
    topology_rows.extend(fusion_topology)

    preprocessing_rows, preprocessing_topology, edge_audit, preprocessing_skips = run_preprocessing_ablations(
        raw_trajectories,
        truth_edges,
        edge_audit,
        network_id,
        n_estimators=n_estimators,
        random_seed=seed,
        n_jobs=n_jobs,
    )
    metric_rows.extend(preprocessing_rows)
    topology_rows.extend(preprocessing_topology)
    skipped.extend(preprocessing_skips)

    trajectory_info = {
        "network_id": network_id,
        "n_trajectories": len(raw_trajectories),
        "n_timeseries_rows": len(raw_timeseries),
        "n_lagged_samples": len(metadata),
        "skipped": "; ".join(sorted(set(skipped))),
    }
    return metric_rows, edge_audit, topology_rows, trajectory_info


def same_time_method_configs(
    expression: pd.DataFrame,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    """Return same-time reference method configs."""
    return [
        make_config(
            method="same_time_correlation",
            method_family="correlation",
            variant="same_time_reference",
            target_type="same_time",
            self_predictor_mode="not_applicable",
            preprocessing="raw",
            ranker=lambda: rank_edges_by_correlation(expression),
        ),
        make_config(
            method="same_time_lasso_alpha_0_1",
            method_family="sparse_linear",
            variant="same_time_reference",
            target_type="same_time",
            self_predictor_mode="exclude_self_predictor",
            preprocessing="raw",
            ranker=lambda: rank_edges_by_lasso(expression, alpha=0.1, max_iter=50000),
        ),
        make_config(
            method="same_time_genie3_random_forest",
            method_family="tree",
            variant="same_time_reference",
            target_type="same_time",
            self_predictor_mode="exclude_self_predictor",
            preprocessing="raw",
            ranker=lambda: rank_edges_by_genie3_random_forest(
                expression,
                n_estimators=n_estimators,
                random_state=random_seed + 11,
                n_jobs=n_jobs,
            ),
        ),
        make_config(
            method="same_time_genie3_extra_trees",
            method_family="tree",
            variant="same_time_reference",
            target_type="same_time",
            self_predictor_mode="exclude_self_predictor",
            preprocessing="raw",
            ranker=lambda: rank_edges_by_genie3_extra_trees(
                expression,
                n_estimators=n_estimators,
                random_state=random_seed + 22,
                n_jobs=n_jobs,
            ),
        ),
    ]


def dynamic_method_configs(
    x: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    """Return dynamic method configs for the broad batch."""
    configs: list[dict[str, object]] = []

    for target_type, target in targets.items():
        configs.append(
            make_config(
                method=f"dynamic_correlation_{target_type}_raw",
                method_family="correlation",
                variant="dynamic",
                target_type=target_type,
                self_predictor_mode="not_applicable",
                preprocessing="raw",
                ranker=lambda target=target: rank_edges_by_dynamic_correlation(x, target),
            )
        )
        for self_mode in SELF_MODES:
            short_self = short_self_mode(self_mode)
            for ensemble in ["random_forest", "extra_trees"]:
                method = f"dynamic_{ensemble}_{target_type}_{short_self}_raw"
                configs.append(
                    make_config(
                        method=method,
                        method_family="tree",
                        variant="dynamic",
                        target_type=target_type,
                        self_predictor_mode=self_mode,
                        preprocessing="raw",
                        ranker=lambda target=target, ensemble=ensemble, target_type=target_type, self_mode=self_mode: rank_edges_by_dynamic_tree_ensemble(
                            x,
                            target,
                            ensemble=ensemble,
                            n_estimators=n_estimators,
                            random_state=random_seed + method_seed_offset(ensemble, target_type, self_mode),
                            self_predictor_mode=self_mode,
                            n_jobs=n_jobs,
                        ),
                    )
                )
            for alpha in LASSO_ALPHAS:
                method = f"dynamic_lasso_a{format_alpha(alpha)}_{target_type}_{short_self}_raw"
                configs.append(
                    make_config(
                        method=method,
                        method_family="sparse_linear",
                        variant="dynamic",
                        target_type=target_type,
                        self_predictor_mode=self_mode,
                        preprocessing="raw",
                        ranker=lambda target=target, alpha=alpha, self_mode=self_mode: rank_edges_by_dynamic_lasso(
                            x,
                            target,
                            alpha=alpha,
                            self_predictor_mode=self_mode,
                            max_iter=50000,
                        ),
                    )
                )
            method = f"dynamic_elastic_net_a0_03_l1_0_7_{target_type}_{short_self}_raw"
            configs.append(
                make_config(
                    method=method,
                    method_family="sparse_linear",
                    variant="dynamic",
                    target_type=target_type,
                    self_predictor_mode=self_mode,
                    preprocessing="raw",
                    ranker=lambda target=target, self_mode=self_mode: rank_edges_by_dynamic_elastic_net(
                        x,
                        target,
                        alpha=0.03,
                        l1_ratio=0.7,
                        self_predictor_mode=self_mode,
                        max_iter=50000,
                    ),
                )
            )

        for alpha in MLP_ALPHAS:
            method = f"dynamic_mlp_a{format_alpha(alpha)}_{target_type}_exclude_self_raw"
            configs.append(
                make_config(
                    method=method,
                    method_family="neural_mlp",
                    variant="dynamic",
                    target_type=target_type,
                    self_predictor_mode="exclude_self_predictor",
                    preprocessing="raw",
                    ranker=lambda target=target, alpha=alpha: rank_edges_by_dynamic_mlp_permutation(
                        x,
                        target,
                        hidden_layer_sizes=(16,),
                        alpha=alpha,
                        random_state=random_seed + int(alpha * 100000),
                        self_predictor_mode="exclude_self_predictor",
                        max_iter=300,
                        n_repeats=2,
                    ),
                )
            )
    return configs


def make_config(
    *,
    method: str,
    method_family: str,
    variant: str,
    target_type: str,
    self_predictor_mode: str,
    preprocessing: str,
    ranker: Ranker,
    stability_enabled: bool = False,
    fusion_enabled: bool = False,
) -> dict[str, object]:
    """Create a method config dictionary."""
    return {
        "method": method,
        "method_family": method_family,
        "variant": variant,
        "target_type": target_type,
        "self_predictor_mode": self_predictor_mode,
        "preprocessing": preprocessing,
        "stability_enabled": stability_enabled,
        "fusion_enabled": fusion_enabled,
        "ranker": ranker,
    }


def run_method_config(
    config: dict[str, object],
    truth_edges: pd.DataFrame,
    network_id: int,
    *,
    n_trajectories: int,
    n_lagged_samples: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """Run and evaluate one configured method."""
    ranker = config["ranker"]
    if not callable(ranker):
        raise TypeError("ranker must be callable")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        predicted_edges = ranker()
    fit_seconds = time.perf_counter() - start
    scored_edges = score_edges(predicted_edges, truth_edges)
    metric_row, topology_row = evaluate_scored_edges(
        scored_edges,
        config,
        network_id,
        n_trajectories=n_trajectories,
        n_lagged_samples=n_lagged_samples,
        fit_seconds=fit_seconds,
    )
    return metric_row, scored_edges, topology_row


def run_stability_subset(
    x: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    truth_edges: pd.DataFrame,
    edge_audit: pd.DataFrame,
    network_id: int,
    *,
    n_trajectories: int,
    n_lagged_samples: int,
    n_resamples: int,
    stability_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run trajectory-bootstrap stability variants for a practical subset."""
    resample_indices = trajectory_bootstrap_indices(metadata, n_resamples, random_seed=random_seed + 500)
    stability_configs = [
        {
            "method": "stability_dynamic_rf_level_exclude_self_raw",
            "method_family": "stability_tree",
            "target_type": "level",
            "self_predictor_mode": "exclude_self_predictor",
            "score_column": "mean_score",
            "ranker": lambda sample_x, sample_y: rank_edges_by_dynamic_tree_ensemble(
                sample_x,
                sample_y,
                ensemble="random_forest",
                n_estimators=stability_estimators,
                random_state=random_seed + 600,
                self_predictor_mode="exclude_self_predictor",
                n_jobs=n_jobs,
            ),
        },
        {
            "method": "stability_dynamic_lasso_a0_1_level_exclude_self_raw",
            "method_family": "stability_sparse_linear",
            "target_type": "level",
            "self_predictor_mode": "exclude_self_predictor",
            "score_column": "selection_frequency",
            "ranker": lambda sample_x, sample_y: rank_edges_by_dynamic_lasso(
                sample_x,
                sample_y,
                alpha=0.1,
                self_predictor_mode="exclude_self_predictor",
                max_iter=50000,
            ),
        },
    ]
    metric_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    score_tables: dict[str, pd.DataFrame] = {}

    for config in stability_configs:
        start = time.perf_counter()
        stability_summary = summarize_resampled_dynamic_scores(
            x,
            targets[str(config["target_type"])],
            config["ranker"],
            resample_indices,
            top_k=20,
            selection_threshold=0.0,
        )
        fit_seconds = time.perf_counter() - start
        predicted = stability_summary[["source", "target", str(config["score_column"])]].rename(
            columns={str(config["score_column"]): "score"}
        )
        scored_edges = score_edges(predicted, truth_edges)
        method_config = make_config(
            method=str(config["method"]),
            method_family=str(config["method_family"]),
            variant="stability",
            target_type=str(config["target_type"]),
            self_predictor_mode=str(config["self_predictor_mode"]),
            preprocessing="raw",
            ranker=lambda: predicted,
            stability_enabled=True,
        )
        metric_row, topology_row = evaluate_scored_edges(
            scored_edges,
            method_config,
            network_id,
            n_trajectories=n_trajectories,
            n_lagged_samples=n_lagged_samples,
            fit_seconds=fit_seconds,
            n_resamples=n_resamples,
        )
        metric_rows.append(metric_row)
        topology_rows.append(topology_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, str(config["method"]))
        score_tables[str(config["method"])] = scored_edges[["source", "target", "score"]].copy()
    return metric_rows, topology_rows, edge_audit, score_tables


def run_rank_fusions(
    score_tables: dict[str, pd.DataFrame],
    truth_edges: pd.DataFrame,
    edge_audit: pd.DataFrame,
    network_id: int,
    *,
    n_trajectories: int,
    n_lagged_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    """Run equal-weight rank-fusion ensembles over fixed candidate inputs."""
    candidate_names = [name for name in fusion_candidate_method_names() if name in score_tables]
    if len(candidate_names) < 2:
        return [], [], edge_audit
    candidate_tables = [score_tables[name] for name in candidate_names]
    rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    for fusion_method in ["mean_normalized_score", "mean_reciprocal_rank", "borda"]:
        method = f"fusion_{fusion_method}"
        config = make_config(
            method=method,
            method_family="rank_fusion",
            variant="fusion",
            target_type="mixed",
            self_predictor_mode="mixed",
            preprocessing="raw",
            ranker=lambda fusion_method=fusion_method: rank_fusion(candidate_tables, method=fusion_method),
            fusion_enabled=True,
        )
        metric_row, scored_edges, topology = run_method_config(
            config,
            truth_edges,
            network_id,
            n_trajectories=n_trajectories,
            n_lagged_samples=n_lagged_samples,
        )
        metric_row["fusion_inputs"] = ";".join(candidate_names)
        rows.append(metric_row)
        topology_rows.append(topology)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)
    return rows, topology_rows, edge_audit


def fusion_candidate_method_names() -> list[str]:
    """Return fixed, non-gold-standard-tuned fusion candidate method names."""
    return [
        "dynamic_correlation_level_raw",
        "dynamic_random_forest_level_exclude_self_raw",
        "dynamic_extra_trees_level_exclude_self_raw",
        "dynamic_lasso_a0_1_level_exclude_self_raw",
        "dynamic_mlp_a0_01_level_exclude_self_raw",
        "stability_dynamic_rf_level_exclude_self_raw",
    ]


def run_preprocessing_ablations(
    raw_trajectories: list[pd.DataFrame],
    truth_edges: pd.DataFrame,
    edge_audit: pd.DataFrame,
    network_id: int,
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame, list[str]]:
    """Run light preprocessing ablations for the strongest practical model."""
    rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    skipped: list[str] = []

    preprocessing_sets = [
        ("moving_average3", moving_average_smooth_trajectories(raw_trajectories, window=3)),
    ]
    wavelet_trajectories, wavelet_skip = maybe_wavelet_denoise_trajectories(raw_trajectories)
    if wavelet_trajectories is not None:
        preprocessing_sets.append(("wavelet_denoise", wavelet_trajectories))
    else:
        skipped.append(wavelet_skip)

    for preprocessing, trajectories in preprocessing_sets:
        x, y, metadata = build_lagged_samples(trajectories)
        target = build_dynamic_target(x, y, metadata, target_type="level")
        method = f"preproc_{preprocessing}_dynamic_rf_level_exclude_self"
        config = make_config(
            method=method,
            method_family="preprocessing_tree",
            variant="preprocessing_ablation",
            target_type="level",
            self_predictor_mode="exclude_self_predictor",
            preprocessing=preprocessing,
            ranker=lambda x=x, target=target, preprocessing=preprocessing: rank_edges_by_dynamic_tree_ensemble(
                x,
                target,
                ensemble="random_forest",
                n_estimators=n_estimators,
                random_state=random_seed + (700 if preprocessing == "moving_average3" else 800),
                self_predictor_mode="exclude_self_predictor",
                n_jobs=n_jobs,
            ),
        )
        metric_row, scored_edges, topology = run_method_config(
            config,
            truth_edges,
            network_id,
            n_trajectories=len(trajectories),
            n_lagged_samples=len(metadata),
        )
        rows.append(metric_row)
        topology_rows.append(topology)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)
    return rows, topology_rows, edge_audit, skipped


def maybe_wavelet_denoise_trajectories(
    trajectories: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame] | None, str]:
    """Optionally denoise trajectories with PyWavelets if it is installed."""
    try:
        import pywt  # type: ignore
    except ImportError:
        return None, "wavelet_denoise skipped: PyWavelets is not installed"

    denoised: list[pd.DataFrame] = []
    for trajectory in trajectories:
        result = trajectory.copy()
        gene_columns = [column for column in trajectory.columns if column != "Time"]
        for column in gene_columns:
            # copy=True: PyWavelets' Cython transform needs a writable buffer,
            # and a bare to_numpy() can return a read-only view.
            values = trajectory[column].to_numpy(dtype=float, copy=True)
            coeffs = pywt.wavedec(values, wavelet="db1", mode="symmetric", level=1)
            detail = coeffs[-1]
            threshold = 0.5 * float(pd.Series(detail).abs().median())
            coeffs[-1] = pywt.threshold(detail, threshold, mode="soft")
            reconstructed = pywt.waverec(coeffs, wavelet="db1", mode="symmetric")[: len(values)]
            result[column] = reconstructed
        denoised.append(result)
    return denoised, ""


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join predicted edge scores to DREAM4 truth labels."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def evaluate_scored_edges(
    scored_edges: pd.DataFrame,
    config: dict[str, object],
    network_id: int,
    *,
    n_trajectories: int,
    n_lagged_samples: int,
    fit_seconds: float,
    n_resamples: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute edge and topology metrics for one scored edge table."""
    n_true_edges = int(scored_edges["is_true"].sum())
    topology = topology_metrics_for_cutoff(scored_edges, cutoff=n_true_edges, rank_column="rank")
    base = {
        "row_type": "network",
        "data_regime": "timeseries",
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": config["method"],
        "method_family": config["method_family"],
        "variant": config["variant"],
        "target_type": config["target_type"],
        "self_predictor_mode": config["self_predictor_mode"],
        "preprocessing": config["preprocessing"],
        "stability_enabled": bool(config["stability_enabled"]),
        "fusion_enabled": bool(config["fusion_enabled"]),
        "n_trajectories": n_trajectories,
        "n_lagged_samples": n_lagged_samples,
        "n_resamples": n_resamples or 0,
        "fit_seconds": fit_seconds,
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": n_true_edges,
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
    }
    topology_row = {**base, **topology}
    metric_row = {**base, **{f"topology_{key}": value for key, value in topology.items()}}
    return metric_row, topology_row


def merge_score_columns(edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge method score and rank columns into the edge audit table."""
    method_scores = scored_edges[["source", "target", "score", "rank"]].rename(
        columns={"score": f"score_{method}", "rank": f"rank_{method}"}
    )
    return edge_audit.merge(method_scores, on=["source", "target"], how="left")


def aggregate_metrics(network_metrics: pd.DataFrame) -> pd.DataFrame:
    """Append mean rows across the five Size10 networks."""
    metadata_columns = [
        "method",
        "method_family",
        "variant",
        "target_type",
        "self_predictor_mode",
        "preprocessing",
        "stability_enabled",
        "fusion_enabled",
    ]
    metric_columns = [
        column
        for column in network_metrics.columns
        if column
        not in {
            "row_type",
            "data_regime",
            "network_id",
            "network",
            *metadata_columns,
            "fusion_inputs",
        }
        and pd.api.types.is_numeric_dtype(network_metrics[column])
    ]
    grouped = network_metrics.groupby(metadata_columns, dropna=False, as_index=False)
    mean_rows = grouped[metric_columns].mean()
    std_rows = grouped[["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]].std().rename(
        columns={
            "auroc": "std_auroc",
            "aupr": "std_aupr",
            "precision_at_5": "std_precision_at_5",
            "precision_at_10": "std_precision_at_10",
            "precision_at_20": "std_precision_at_20",
        }
    )
    counts = grouped.size().rename(columns={"size": "n_networks"})
    mean_rows = mean_rows.merge(std_rows, on=metadata_columns, how="left").merge(counts, on=metadata_columns)
    mean_rows.insert(0, "row_type", "mean")
    mean_rows["data_regime"] = "timeseries"
    mean_rows["network_id"] = pd.NA
    mean_rows["network"] = "mean_across_size10_networks"
    return pd.concat([network_metrics, mean_rows], ignore_index=True, sort=False)


def mean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate rows."""
    return summary[summary["row_type"] == "mean"].copy()


def make_debug_report(
    summary: pd.DataFrame,
    topology: pd.DataFrame,
    trajectory_info: pd.DataFrame,
    skipped: list[str],
) -> str:
    """Build a human-readable dynamic batch debug report."""
    means = mean_rows(summary)
    best_aupr = best_row(means, "aupr")
    best_auroc = best_row(means, "auroc")
    best_p5 = best_row(means, "precision_at_5")
    best_p10 = best_row(means, "precision_at_10")
    best_p20 = best_row(means, "precision_at_20")
    best_out_hub = best_row(means, "topology_top3_out_hub_overlap")
    best_in_hub = best_row(means, "topology_top3_in_hub_overlap")
    target_summary = best_by_group(means[means["variant"] == "dynamic"], "target_type", "aupr")
    self_summary = best_by_group(means[means["variant"] == "dynamic"], "self_predictor_mode", "aupr")
    family_summary = best_by_group(means, "method_family", "aupr")
    fusion_summary = means[means["fusion_enabled"]].sort_values("aupr", ascending=False)
    stability_summary = means[means["stability_enabled"]].sort_values("aupr", ascending=False)
    preprocessing_summary = means[means["variant"] == "preprocessing_ablation"].sort_values("aupr", ascending=False)
    mlp_summary = means[means["method_family"] == "neural_mlp"].sort_values("aupr", ascending=False)

    lines = [
        "# DREAM4 Size10 Dynamic Model Batch Debug Report",
        "",
        "This broad benchmark compares temporal modeling, tree-based conditional prediction, sparse linear prediction, a small neural MLP baseline, trajectory-bootstrap stability, equal-weight rank fusion, and light signal preprocessing.",
        "",
        "All dynamic methods use within-trajectory lag pairs only. Reinforcement learning/GFlowNet graph search is intentionally not implemented here because it needs a separate graph-search formulation.",
        "",
        "## Trajectory Summary",
        "",
        to_markdown_table(trajectory_info),
        "",
        "## Best Overall Metrics",
        "",
        to_markdown_table(pd.DataFrame([best_aupr, best_auroc, best_p5, best_p10, best_p20, best_out_hub, best_in_hub])),
        "",
        "## Best Mean AUPR By Model Family",
        "",
        to_markdown_table(family_summary[["method_family", "method", "aupr", "auroc", "precision_at_10"]]),
        "",
        "## Best Mean AUPR By Target Type",
        "",
        to_markdown_table(target_summary[["target_type", "method", "aupr", "auroc"]]),
        "",
        "## Best Mean AUPR By Self-Predictor Mode",
        "",
        to_markdown_table(self_summary[["self_predictor_mode", "method", "aupr", "auroc"]]),
        "",
        "## Stability Variants",
        "",
        to_markdown_table(stability_summary[["method", "aupr", "auroc", "precision_at_10", "topology_top3_out_hub_overlap", "topology_top3_in_hub_overlap"]]),
        "",
        "## Rank Fusion Variants",
        "",
        to_markdown_table(fusion_summary[["method", "aupr", "auroc", "precision_at_10", "topology_top3_out_hub_overlap", "topology_top3_in_hub_overlap"]]),
        "",
        "## Neural MLP Variants",
        "",
        to_markdown_table(mlp_summary.head(10)[["method", "target_type", "aupr", "auroc", "precision_at_10"]]),
        "",
        "## Preprocessing Ablations",
        "",
        to_markdown_table(preprocessing_summary[["method", "preprocessing", "aupr", "auroc", "precision_at_10", "topology_reciprocal_false_positive_pair_rate"]]),
        "",
        "## Interpretation",
        "",
        interpret_results(means),
        "",
        "## Skipped Or Deferred",
        "",
        skipped_text(skipped),
        "",
    ]
    return "\n".join(lines)


def best_row(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    """Return best method row for one metric."""
    row = frame.sort_values([metric, "method"], ascending=[False, True]).iloc[0]
    return {
        "metric": metric,
        "method": row["method"],
        "method_family": row["method_family"],
        "value": row[metric],
    }


def best_by_group(frame: pd.DataFrame, group_column: str, metric: str) -> pd.DataFrame:
    """Return best row per group by a metric."""
    if frame.empty:
        return pd.DataFrame(columns=[group_column, "method", metric])
    rows = []
    for group_value, group in frame.groupby(group_column, dropna=False):
        rows.append(group.sort_values([metric, "method"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(rows).sort_values(metric, ascending=False)


def interpret_results(means: pd.DataFrame) -> str:
    """Return concise cautious interpretation."""
    best = best_row(means, "aupr")
    best_single = best_row(means[~means["fusion_enabled"]], "aupr")
    best_fusion = best_row(means[means["fusion_enabled"]], "aupr") if not means[means["fusion_enabled"]].empty else None
    best_stability = best_row(means[means["stability_enabled"]], "aupr") if not means[means["stability_enabled"]].empty else None
    best_mlp = best_row(means[means["method_family"] == "neural_mlp"], "aupr") if not means[means["method_family"] == "neural_mlp"].empty else None
    best_preproc = best_row(means[means["variant"] == "preprocessing_ablation"], "aupr") if not means[means["variant"] == "preprocessing_ablation"].empty else None
    raw_rf = means[means["method"] == "dynamic_random_forest_level_exclude_self_raw"]

    lines = [
        f"Best mean AUPR method: `{best['method']}` ({best['value']:.6f}).",
        f"Best non-fusion mean AUPR method: `{best_single['method']}` ({best_single['value']:.6f}).",
    ]
    if best_fusion is not None:
        lines.append(
            f"Best fusion method: `{best_fusion['method']}` ({best_fusion['value']:.6f}); compare this to the best single model before treating fusion as useful."
        )
    if best_stability is not None:
        lines.append(f"Best stability method: `{best_stability['method']}` ({best_stability['value']:.6f}).")
    if best_mlp is not None:
        lines.append(f"Best MLP method: `{best_mlp['method']}` ({best_mlp['value']:.6f}); this is a sanity baseline, not a neural-model claim.")
    if best_preproc is not None:
        baseline_value = float(raw_rf["aupr"].iloc[0]) if not raw_rf.empty else float("nan")
        lines.append(
            f"Best preprocessing ablation: `{best_preproc['method']}` ({best_preproc['value']:.6f}); raw RF level/exclude-self AUPR was {baseline_value:.6f}."
        )
    lines.append("This is a broad benchmark batch; apparent winners should be checked for stability across networks and topology metrics before scaling.")
    return "\n".join(lines)


def skipped_text(skipped: list[str]) -> str:
    """Return skipped/deferred notes."""
    unique = sorted({item for item in skipped if item})
    if not unique:
        return "- No optional preprocessing dependency was skipped."
    return "\n".join(f"- {item}" for item in unique)


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = [[format_cell(value) for value in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_cell(value: object) -> str:
    """Format a markdown cell."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_summary(summary: pd.DataFrame) -> None:
    """Print compact mean rows."""
    means = mean_rows(summary).sort_values("aupr", ascending=False)
    columns = [
        "method",
        "method_family",
        "variant",
        "target_type",
        "self_predictor_mode",
        "preprocessing",
        "auroc",
        "aupr",
        "precision_at_10",
        "topology_top3_out_hub_overlap",
        "topology_top3_in_hub_overlap",
        "topology_reciprocal_false_positive_pair_rate",
    ]
    print("DREAM4 Size10 dynamic model batch audit")
    print()
    print(means[columns].head(25).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_edges: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_topology: {TOPOLOGY_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def short_self_mode(self_mode: str) -> str:
    """Return compact self-predictor mode text for method names."""
    return "exclude_self" if self_mode == "exclude_self_predictor" else "include_self"


def format_alpha(alpha: float) -> str:
    """Format alpha for method names."""
    return str(alpha).replace(".", "_")


def method_seed_offset(*parts: str) -> int:
    """Return a deterministic small seed offset for method config strings."""
    return sum(ord(character) for part in parts for character in part)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--stability-estimators", type=int, default=50)
    parser.add_argument("--n-resamples", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    """Run the dynamic model batch audit."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[dict[str, object]] = []
    all_edge_tables: list[pd.DataFrame] = []
    all_topology_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for network_id in range(1, 6):
        metric_rows, edge_audit, topology_rows, trajectory_info = run_network(
            network_id,
            n_estimators=args.n_estimators,
            stability_estimators=args.stability_estimators,
            n_resamples=args.n_resamples,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
        all_metric_rows.extend(metric_rows)
        all_edge_tables.append(edge_audit)
        all_topology_rows.extend(topology_rows)
        trajectory_rows.append(trajectory_info)
        if trajectory_info["skipped"]:
            skipped.append(str(trajectory_info["skipped"]))

    network_metrics = pd.DataFrame(all_metric_rows)
    topology = pd.DataFrame(all_topology_rows)
    summary = aggregate_metrics(network_metrics)
    edge_audit = pd.concat(all_edge_tables, ignore_index=True)
    trajectory_info = pd.DataFrame(trajectory_rows)

    summary.to_csv(SUMMARY_PATH, index=False)
    edge_audit.to_csv(EDGE_AUDIT_PATH, index=False)
    topology.to_csv(TOPOLOGY_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(make_debug_report(summary, topology, trajectory_info, skipped), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
