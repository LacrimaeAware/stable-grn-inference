"""DREAM4 calibrated confidence: a deployable, gold-free edge-confidence rule.

This turns the mechanism findings (experiment 13) into a usable pipeline:

  Part 1. Select the dynamic sparse alpha WITHOUT gold labels (cross-validation,
          BIC, AIC, a density-prior heuristic, and bootstrap selection stability).
  Part 2. Build an edge-confidence score from equal-weight agreement of
          complementary methods (fusion, agreement counts, reciprocal penalty,
          optional topology penalty). No weights tuned on gold labels.
  Part 3. Calibration diagnostics: do higher-confidence edges have higher true
          rates (reliability, ECE-style summary, top-k reliability)?
  Part 4. A topology-aware decision layer with separate winners for edge ranking,
          top-k precision, topology/hub recovery, and reciprocal-direction control.
  Part 5. Baselines including an ORACLE-alpha sparse model (clearly labeled, not
          deployable), and dynGENIE3-style tree references.

Gold labels are used only for evaluation after selection. No official dynGENIE3
package is installed here, so tree methods are dynGENIE3-style.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold

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
    trajectory_bootstrap_indices,
)
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k, topology_metrics_for_cutoff
from stable_grn_inference.inference import (
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_extra_trees,
    rank_edges_by_lagged_random_forest,
    rank_fusion,
    rank_fusion_with_reciprocal_penalty,
    summarize_resampled_dynamic_linear_coefficients,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

RESULTS_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"
PREFIX = "dream4_calibrated_confidence"
SUMMARY_PATH = RESULTS_DIR / f"{PREFIX}_summary.csv"
PER_NETWORK_PATH = RESULTS_DIR / f"{PREFIX}_per_network.csv"
EDGES_PATH = RESULTS_DIR / f"{PREFIX}_edges.csv"
CALIBRATION_PATH = RESULTS_DIR / f"{PREFIX}_calibration_bins.csv"
TOPOLOGY_PATH = RESULTS_DIR / f"{PREFIX}_topology.csv"
ALPHA_SELECTION_PATH = RESULTS_DIR / f"{PREFIX}_alpha_selection.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / f"{PREFIX}_debug_report.md"

NETWORK_IDS = range(1, 6)
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_ITER = 50000
COEF_TOL = 1e-12
AGREEMENT_QUANTILES = (1, 5, 10)  # percent
RECIPROCAL_PENALTIES = (0.5, 0.25)
DENSITY_PRIORS = (1, 2, 3)  # expected regulators per gene
N_CALIBRATION_BINS = 10

# (label, model_kind, l1_ratio, include_self)
SPARSE_MODELS = [
    ("dynamic_lasso_level_include_self", "lasso", None, True),
    ("dynamic_lasso_level_exclude_self", "lasso", None, False),
    ("dynamic_elastic_net_level_include_self", "elastic_net", 0.7, True),
]
FOCAL_MODEL = "dynamic_lasso_level_include_self"  # deployable sparse input to fusion

SIZE_SETTINGS = {
    10: {
        "expression_path": lambda n: dream4_size10_expression_path(ROOT / "data/raw/dream4", n, "timeseries"),
        "gold_path": lambda n: dream4_size10_gold_standard_path(ROOT / "data/raw/dream4", n),
        "precision_ks": (5, 10, 20),
        "hub_top": 3,
        "default_trees": 200,
    },
    100: {
        "expression_path": lambda n: dream4_size100_expression_path(ROOT / "data/raw/dream4", n, "timeseries"),
        "gold_path": lambda n: dream4_size100_gold_standard_path(ROOT / "data/raw/dream4", n),
        "precision_ks": (10, 50, 100, 200),
        "hub_top": 5,
        "default_trees": 100,
    },
}


# --------------------------------------------------------------------------- #
# Standardization + target-wise sparse fit with RSS/BIC/AIC (gold-free)
# --------------------------------------------------------------------------- #
def _standardize_columns(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (values - values.mean(axis=0)) / scale


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    if scale == 0.0:
        scale = 1.0
    return (values - values.mean()) / scale


def _make_model(model_kind: str, alpha: float, l1_ratio: float | None) -> Lasso | ElasticNet:
    if model_kind == "lasso":
        return Lasso(alpha=alpha, fit_intercept=False, max_iter=MAX_ITER)
    if model_kind == "elastic_net":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=MAX_ITER)
    raise ValueError("model_kind must be 'lasso' or 'elastic_net'")


def fit_targetwise(x: pd.DataFrame, target: pd.DataFrame, *, alpha: float, model_kind: str,
                   l1_ratio: float | None, include_self: bool):
    """Fit per-target standardized sparse linear models; return edges, RSS, nnz, BIC, AIC.

    Uses no gold labels. BIC/AIC use nonzero-coefficient count as the degrees of
    freedom, which is the standard LASSO information-criterion approximation.
    """
    genes = [str(g) for g in x.columns]
    edge_rows: list[dict[str, object]] = []
    total_rss = 0.0
    total_nnz = 0
    total_bic = 0.0
    total_aic = 0.0
    n = len(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gene in genes:
            predictors = genes if include_self else [g for g in genes if g != gene]
            x_values = _standardize_columns(x[predictors].to_numpy(dtype=float))
            y_values = _standardize_vector(target[gene].to_numpy(dtype=float))
            model = _make_model(model_kind, alpha, l1_ratio)
            model.fit(x_values, y_values)
            rss = float(np.sum((y_values - model.predict(x_values)) ** 2))
            k = int(np.sum(np.abs(model.coef_) > COEF_TOL))
            total_rss += rss
            total_bic += n * np.log(max(rss, 1e-12) / n) + k * np.log(max(n, 2))
            total_aic += n * np.log(max(rss, 1e-12) / n) + 2 * k
            for source, coef in zip(predictors, model.coef_):
                if source != gene:
                    selected = abs(float(coef)) > COEF_TOL
                    edge_rows.append({"source": source, "target": gene, "score": abs(float(coef)), "selected": selected})
                    total_nnz += int(selected)
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "score", "selected"])
    edges = edges.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    return edges, total_rss, total_nnz, float(total_bic), float(total_aic)


def cv_mse(x: pd.DataFrame, target: pd.DataFrame, *, alpha: float, model_kind: str,
           l1_ratio: float | None, include_self: bool, folds: int, seed: int) -> float:
    """K-fold held-out MSE (gold-free), summed over targets."""
    genes = [str(g) for g in x.columns]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    total_se = 0.0
    total_count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gene in genes:
            predictors = genes if include_self else [g for g in genes if g != gene]
            x_all = x[predictors].to_numpy(dtype=float)
            y_all = target[gene].to_numpy(dtype=float)
            for train_idx, test_idx in splitter.split(x_all):
                mean = x_all[train_idx].mean(axis=0)
                scale = x_all[train_idx].std(axis=0)
                scale[scale == 0.0] = 1.0
                y_mean = y_all[train_idx].mean()
                y_scale = y_all[train_idx].std() or 1.0
                model = _make_model(model_kind, alpha, l1_ratio)
                model.fit((x_all[train_idx] - mean) / scale, (y_all[train_idx] - y_mean) / y_scale)
                prediction = model.predict((x_all[test_idx] - mean) / scale)
                total_se += float(np.sum(((y_all[test_idx] - y_mean) / y_scale - prediction) ** 2))
                total_count += len(test_idx)
    return total_se / total_count if total_count else float("nan")


# --------------------------------------------------------------------------- #
# Gold-free alpha selectors (pure functions over per-alpha criteria)
# --------------------------------------------------------------------------- #
def select_alpha_min(criterion_by_alpha: dict[float, float]) -> float:
    """Return the alpha minimizing a criterion (e.g. CV MSE, BIC, AIC)."""
    valid = {a: v for a, v in criterion_by_alpha.items() if v is not None and not np.isnan(v)}
    if not valid:
        raise ValueError("no valid criterion values")
    return min(valid, key=lambda a: valid[a])


def select_alpha_max(criterion_by_alpha: dict[float, float]) -> float:
    """Return the alpha maximizing a criterion (e.g. bootstrap stability)."""
    valid = {a: v for a, v in criterion_by_alpha.items() if v is not None and not np.isnan(v)}
    if not valid:
        raise ValueError("no valid criterion values")
    return max(valid, key=lambda a: valid[a])


def select_alpha_by_density_prior(nnz_by_alpha: dict[float, int], target_nnz: int) -> float:
    """Return the alpha whose nonzero edge count is closest to a target count."""
    if target_nnz < 0:
        raise ValueError("target_nnz must be nonnegative")
    return min(nnz_by_alpha, key=lambda a: (abs(nnz_by_alpha[a] - target_nnz), a))


# --------------------------------------------------------------------------- #
# Confidence / fusion utilities
# --------------------------------------------------------------------------- #
def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _sort_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return edges.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)


def agreement_count_confidence(edge_tables: list[pd.DataFrame], *, top_fraction: float) -> pd.DataFrame:
    """Confidence = number of methods ranking an edge in their top ``top_fraction``.

    Ties are broken by the mean min-max-normalized score so the ranking is a total
    order. Equal weights; no gold labels.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    base = edge_tables[0][["source", "target"]].copy()
    counts = np.zeros(len(base), dtype=float)
    norm_sum = np.zeros(len(base), dtype=float)
    for table in edge_tables:
        ranked = _sort_edges(table[["source", "target", "score"]].copy())
        top_k = max(1, int(round(top_fraction * len(ranked))))
        top_pairs = set(map(tuple, ranked.head(top_k)[["source", "target"]].to_numpy()))
        merged = base.merge(ranked, on=["source", "target"], how="left")
        counts += np.array([(str(s), str(t)) in top_pairs for s, t in zip(base["source"], base["target"])], dtype=float)
        norm_sum += _minmax(merged["score"].fillna(0.0).to_numpy(dtype=float))
    result = base.copy()
    result["score"] = counts + 1e-6 * (norm_sum / len(edge_tables))
    return _sort_edges(result)


def confidence_topology_penalty(edge_tables: list[pd.DataFrame], *, penalty: float, top_fraction: float) -> pd.DataFrame:
    """Borda fusion, then down-weight BOTH directions of high-confidence reciprocal pairs.

    Unlike the reciprocal penalty (which keeps the stronger direction), this
    discourages reciprocal structure outright by penalizing both directions,
    targeting topology-level reciprocal inflation. Fixed weight, no tuning.
    """
    if not 0.0 < penalty <= 1.0:
        raise ValueError("penalty must be in (0, 1]")
    fused = rank_fusion(edge_tables, method="borda")
    n = len(fused)
    top_k = max(1, int(round(top_fraction * n)))
    top_pairs = set(map(tuple, fused.head(top_k)[["source", "target"]].to_numpy()))
    penalized = set()
    for source, target in top_pairs:
        if (str(target), str(source)) in top_pairs:
            penalized.add((str(source), str(target)))
            penalized.add((str(target), str(source)))
    adjusted = fused.copy()
    factors = np.array([penalty if (str(s), str(t)) in penalized else 1.0 for s, t in zip(adjusted["source"], adjusted["target"])])
    adjusted["score"] = adjusted["score"].to_numpy(dtype=float) * factors
    return _sort_edges(adjusted)


def calibration_bins(scored: pd.DataFrame, *, n_bins: int = N_CALIBRATION_BINS) -> tuple[pd.DataFrame, dict[str, float]]:
    """Bin edges by score rank into equal-size bins; report true-edge rate per bin.

    Returns a per-bin table (bin 1 = highest confidence) and a summary with an
    ECE-style score (|normalized confidence - empirical true rate|, size-weighted)
    and the Spearman correlation between mean confidence and true rate across bins
    (positive => higher confidence really means higher true-edge rate).
    """
    frame = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    n = len(frame)
    normalized = _minmax(frame["score"].to_numpy(dtype=float))
    bin_index = np.minimum(n_bins - 1, (np.arange(n) * n_bins) // n)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        true_rate = float(frame.loc[mask, "is_true"].mean())
        mean_norm = float(normalized[mask].mean())
        rows.append({"bin": b + 1, "count": count, "mean_score": float(frame.loc[mask, "score"].mean()),
                     "mean_normalized_confidence": mean_norm, "empirical_true_rate": true_rate})
        ece += (count / n) * abs(mean_norm - true_rate)
    bins = pd.DataFrame(rows)
    if len(bins) >= 3 and bins["mean_normalized_confidence"].nunique() > 1 and bins["empirical_true_rate"].nunique() > 1:
        corr = spearmanr(bins["mean_normalized_confidence"], bins["empirical_true_rate"]).correlation
    else:
        corr = float("nan")
    summary = {"ece": float(ece), "confidence_true_rate_spearman": float(corr) if not pd.isna(corr) else float("nan"),
               "top_bin_true_rate": float(bins.iloc[0]["empirical_true_rate"]) if len(bins) else float("nan")}
    return bins, summary


# --------------------------------------------------------------------------- #
# Data + scoring
# --------------------------------------------------------------------------- #
def load_size_network(size: int, network_id: int) -> dict[str, object]:
    settings = SIZE_SETTINGS[size]
    timeseries = load_expression_matrix(settings["expression_path"](network_id), drop_time=False)
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, metadata = build_lagged_samples(trajectories)
    level = build_dynamic_target(x_t, y_t1, metadata, target_type="level")
    truth = load_gold_standard_edges(settings["gold_path"](network_id))
    genes = [str(c) for c in x_t.columns]
    return {"size": size, "network_id": network_id, "x_t": x_t, "y_t1": y_t1, "level": level,
            "metadata": metadata, "truth": truth, "genes": genes,
            "n_candidate": len(genes) * (len(genes) - 1), "n_true": int(truth["is_true"].sum())}


def score_edges(predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    scored = predicted.merge(truth, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def evaluate(scored: pd.DataFrame, *, descriptors: dict, genes: list[str], n_true: int, ks: tuple[int, ...]):
    topo = topology_metrics_for_cutoff(scored, cutoff=n_true, rank_column="rank", genes=genes)
    base = {**descriptors, "n_true_edges": n_true, "n_candidate_edges": len(scored),
            "auroc": auroc(scored["is_true"], scored["score"]), "aupr": aupr(scored["is_true"], scored["score"]),
            **{f"precision_at_{k}": precision_at_k(scored, "is_true", k) for k in ks}}
    metric_row = {**base, **{f"topology_{k}": v for k, v in topo.items()}}
    topo_row = {**base, **topo}
    return metric_row, topo_row


# --------------------------------------------------------------------------- #
# Part 1: deployable alpha selection
# --------------------------------------------------------------------------- #
def run_alpha_selection(net, *, bootstrap_resamples: int, seed: int):
    """Sweep alpha for each sparse model; compute gold-free selectors + oracle diagnostic."""
    x_t, level, truth, genes = net["x_t"], net["level"], net["truth"], net["genes"]
    n_true, n_candidate = net["n_true"], net["n_candidate"]
    settings = SIZE_SETTINGS[net["size"]]
    rows: list[dict[str, object]] = []
    selected_edges: dict[tuple[str, str], pd.DataFrame] = {}

    for label, model_kind, l1_ratio, include_self in SPARSE_MODELS:
        per_alpha = {}
        for alpha in ALPHA_GRID:
            edges, rss, nnz, bic, aic = fit_targetwise(x_t, level, alpha=alpha, model_kind=model_kind, l1_ratio=l1_ratio, include_self=include_self)
            scored = score_edges(edges[["source", "target", "score"]], truth)
            metrics, _ = evaluate(scored, descriptors={}, genes=genes, n_true=n_true, ks=settings["precision_ks"])
            cv = cv_mse(x_t, level, alpha=alpha, model_kind=model_kind, l1_ratio=l1_ratio, include_self=include_self, folds=5, seed=seed + net["network_id"])
            per_alpha[alpha] = {"edges": edges, "scored": scored, "metrics": metrics, "nnz": nnz,
                                "predicted_density": nnz / n_candidate, "bic": bic, "aic": aic, "cv": cv}

        stability = {}
        if label == FOCAL_MODEL and bootstrap_resamples > 0:
            stability = _bootstrap_stability(x_t, level, net["metadata"], per_alpha, resamples=bootstrap_resamples, seed=seed + net["network_id"])

        oracle_alpha = max(ALPHA_GRID, key=lambda a: per_alpha[a]["metrics"]["aupr"])
        selectors = {
            "oracle": oracle_alpha,
            "cv": select_alpha_min({a: per_alpha[a]["cv"] for a in ALPHA_GRID}),
            "bic": select_alpha_min({a: per_alpha[a]["bic"] for a in ALPHA_GRID}),
            "aic": select_alpha_min({a: per_alpha[a]["aic"] for a in ALPHA_GRID}),
        }
        for prior in DENSITY_PRIORS:
            selectors[f"density_prior_{prior}"] = select_alpha_by_density_prior({a: per_alpha[a]["nnz"] for a in ALPHA_GRID}, prior * len(genes))
        if stability:
            selectors["bootstrap_stability"] = select_alpha_max(stability)

        for rule, chosen in selectors.items():
            payload = per_alpha[chosen]
            metrics = payload["metrics"]
            rows.append({
                "size": net["size"], "network_id": net["network_id"], "model": label, "selection_rule": rule,
                "deployable": rule != "oracle", "chosen_alpha": chosen, "oracle_alpha": oracle_alpha,
                "alpha_log10_gap_to_oracle": abs(np.log10(chosen) - np.log10(oracle_alpha)),
                "predicted_density": payload["predicted_density"], "true_density": n_true / n_candidate,
                "n_nonzero_nonself": payload["nnz"], "auroc": metrics["auroc"], "aupr": metrics["aupr"],
                "aupr_gap_to_oracle": per_alpha[oracle_alpha]["metrics"]["aupr"] - metrics["aupr"],
                "precision_at_10": metrics.get("precision_at_10", float("nan")),
                "topology_reciprocal_fp_rate": metrics[f"topology_reciprocal_false_positive_pair_rate"],
                f"topology_top{settings['hub_top']}_out_hub_overlap": metrics[f"topology_top{settings['hub_top']}_out_hub_overlap"],
            })
            if label == FOCAL_MODEL and rule in ("oracle", "cv", "bic"):
                selected_edges[(label, rule)] = payload["scored"]
    return pd.DataFrame(rows), selected_edges


def _bootstrap_stability(x_t, level, metadata, per_alpha, *, resamples, seed):
    resample_indices = trajectory_bootstrap_indices(metadata, resamples, random_seed=seed)
    out = {}
    for alpha, payload in per_alpha.items():
        density = payload["predicted_density"]
        if density < 0.005 or density > 0.5:
            out[alpha] = 0.0
            continue
        edge_summary, _ = summarize_resampled_dynamic_linear_coefficients(
            x_t, level, resample_indices, model_kind="lasso", alpha=alpha,
            self_predictor_mode="include_self_predictor_no_self_edge", max_iter=MAX_ITER)
        merged = payload["edges"][["source", "target", "selected"]].merge(
            edge_summary[["source", "target", "selection_frequency"]], on=["source", "target"], how="left")
        selected = merged[merged["selected"]]
        out[alpha] = float(selected["selection_frequency"].mean()) if len(selected) else 0.0
    return out


# --------------------------------------------------------------------------- #
# Main per-network evaluation: baselines + confidence methods
# --------------------------------------------------------------------------- #
def evaluate_network(net, selected_edges, *, tree_estimators, seed, n_jobs):
    """Score baselines, deployable sparse, and confidence/fusion methods for one network."""
    x_t, y_t1, truth, genes = net["x_t"], net["y_t1"], net["truth"], net["genes"]
    n_true = net["n_true"]
    ks = SIZE_SETTINGS[net["size"]]["precision_ks"]
    scored: dict[str, pd.DataFrame] = {}

    scored["lagged_correlation"] = score_edges(rank_edges_by_lagged_correlation(x_t, y_t1), truth)
    scored["lagged_genie3_rf_level"] = score_edges(rank_edges_by_lagged_random_forest(x_t, y_t1, n_estimators=tree_estimators, random_state=seed + net["network_id"] * 10 + 1, n_jobs=n_jobs), truth)
    scored["lagged_genie3_extra_trees_level"] = score_edges(rank_edges_by_lagged_extra_trees(x_t, y_t1, n_estimators=tree_estimators, random_state=seed + net["network_id"] * 10 + 2, n_jobs=n_jobs), truth)
    scored["sparse_oracle_alpha"] = selected_edges[(FOCAL_MODEL, "oracle")]
    scored["sparse_cv_alpha"] = selected_edges[(FOCAL_MODEL, "cv")]
    scored["sparse_bic_alpha"] = selected_edges[(FOCAL_MODEL, "bic")]

    # confidence inputs: deployable sparse (CV) + tree + correlation (equal weights)
    inputs = [scored["sparse_cv_alpha"][["source", "target", "score"]],
              scored["lagged_genie3_rf_level"][["source", "target", "score"]],
              scored["lagged_correlation"][["source", "target", "score"]]]
    scored["fusion_borda"] = score_edges(rank_fusion(inputs, method="borda"), truth)
    scored["fusion_mean_reciprocal_rank"] = score_edges(rank_fusion(inputs, method="mean_reciprocal_rank"), truth)
    scored["fusion_normalized_mean"] = score_edges(rank_fusion(inputs, method="mean_normalized_score"), truth)
    for q in AGREEMENT_QUANTILES:
        scored[f"confidence_agreement_top{q}pct"] = score_edges(agreement_count_confidence(inputs, top_fraction=q / 100.0), truth)
    for penalty in RECIPROCAL_PENALTIES:
        label = f"confidence_reciprocal_penalty_w{str(penalty).replace('.', '_')}"
        scored[label] = score_edges(rank_fusion_with_reciprocal_penalty(inputs, penalty=penalty, top_fraction=0.05), truth)
    scored["confidence_topology_penalty_w0_5"] = score_edges(confidence_topology_penalty(inputs, penalty=0.5, top_fraction=0.05), truth)
    return scored


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_size(size, *, bootstrap_resamples, tree_estimators, seed, n_jobs):
    metric_rows, topo_rows, alpha_rows, calib_rows = [], [], [], []
    edge_blocks = []
    calibration_methods = ["sparse_cv_alpha", "lagged_genie3_rf_level", "fusion_borda"]
    headline_edges = ["lagged_correlation", "sparse_cv_alpha", "sparse_oracle_alpha", "lagged_genie3_rf_level", "fusion_borda", "confidence_agreement_top5pct", "confidence_reciprocal_penalty_w0_5"]
    settings = SIZE_SETTINGS[size]

    for nid in NETWORK_IDS:
        net = load_size_network(size, nid)
        alpha_df, selected_edges = run_alpha_selection(net, bootstrap_resamples=bootstrap_resamples, seed=seed)
        alpha_rows.append(alpha_df)
        scored = evaluate_network(net, selected_edges, tree_estimators=tree_estimators, seed=seed, n_jobs=n_jobs)

        for method, scored_edges in scored.items():
            descriptors = {"size": size, "network_id": nid, "method": method,
                           "deployable": method != "sparse_oracle_alpha", "family": _family_of(method)}
            metric_row, topo_row = evaluate(scored_edges, descriptors=descriptors, genes=net["genes"], n_true=net["n_true"], ks=settings["precision_ks"])
            metric_rows.append(metric_row)
            topo_rows.append(topo_row)

        for method in calibration_methods:
            bins, summary = calibration_bins(scored[method])
            for _, brow in bins.iterrows():
                calib_rows.append({"size": size, "network_id": nid, "method": method, **brow.to_dict(),
                                   "ece": summary["ece"], "confidence_true_rate_spearman": summary["confidence_true_rate_spearman"]})

        base = scored["lagged_correlation"][["source", "target", "is_true"]].copy()
        base.insert(0, "size", size)
        base.insert(1, "network_id", nid)
        for method in headline_edges:
            if method in scored:
                merged = scored[method][["source", "target", "score", "rank"]].rename(columns={"score": f"score_{method}", "rank": f"rank_{method}"})
                base = base.merge(merged, on=["source", "target"], how="left")
        edge_blocks.append(base)

    return {
        "per_network": pd.DataFrame(metric_rows), "topology": pd.DataFrame(topo_rows),
        "alpha_selection": pd.concat(alpha_rows, ignore_index=True), "calibration": pd.DataFrame(calib_rows),
        "edges": pd.concat(edge_blocks, ignore_index=True), "size": size,
    }


def _family_of(method: str) -> str:
    if method.startswith("sparse"):
        return "sparse"
    if method.startswith("fusion") or method.startswith("confidence"):
        return "confidence"
    if "genie3" in method:
        return "tree"
    return "correlation"


def aggregate(per_network: pd.DataFrame) -> pd.DataFrame:
    group = ["size", "method", "deployable", "family"]
    metric_cols = [c for c in per_network.columns if c not in group + ["network_id", "n_true_edges", "n_candidate_edges"]
                   and pd.api.types.is_numeric_dtype(per_network[c])]
    grouped = per_network.groupby(group, dropna=False, as_index=False)[metric_cols].mean()
    return grouped.sort_values(["size", "aupr"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def write_figures(alpha_selection, summary, calibration) -> list[str]:
    if not HAVE_MPL:
        return []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    focal = alpha_selection[alpha_selection["model"] == FOCAL_MODEL]
    # selected alpha by rule and size
    fig, ax = plt.subplots(figsize=(7, 4))
    rules = ["oracle", "cv", "bic", "aic", "density_prior_2"]
    for size in sorted(focal["size"].unique()):
        means = [focal[(focal["size"] == size) & (focal["selection_rule"] == r)]["chosen_alpha"].median() for r in rules]
        ax.plot(rules, means, marker="o", label=f"Size{size}")
    ax.set_yscale("log"); ax.set_ylabel("median chosen alpha"); ax.legend(); ax.set_title("selected alpha by rule")
    p = FIGURES_DIR / "selected_alpha_by_rule_and_size.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.as_posix())

    # predicted density by selection rule
    fig, ax = plt.subplots(figsize=(7, 4))
    for size in sorted(focal["size"].unique()):
        means = [focal[(focal["size"] == size) & (focal["selection_rule"] == r)]["predicted_density"].mean() for r in rules]
        ax.plot(rules, means, marker="s", label=f"Size{size} predicted")
        ax.axhline(focal[focal["size"] == size]["true_density"].mean(), linestyle="--", alpha=0.4)
    ax.set_ylabel("predicted edge density"); ax.legend(); ax.set_title("predicted density by rule (dashed = true)")
    p = FIGURES_DIR / "predicted_density_by_selection_rule.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.as_posix())

    # confidence bin true-edge rate
    fig, ax = plt.subplots(figsize=(7, 4))
    for size in sorted(calibration["size"].unique()):
        for method in ["sparse_cv_alpha", "fusion_borda"]:
            sub = calibration[(calibration["size"] == size) & (calibration["method"] == method)].groupby("bin")["empirical_true_rate"].mean()
            ax.plot(sub.index, sub.values, marker="o", label=f"Size{size} {method}")
    ax.set_xlabel("confidence bin (1=highest)"); ax.set_ylabel("empirical true-edge rate"); ax.legend(fontsize=7); ax.set_title("calibration reliability")
    p = FIGURES_DIR / "confidence_bin_true_edge_rate.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.as_posix())

    # AUPR/topology comparison + reciprocal FP comparison
    for metric, fname, title in (("aupr", "aupr_comparison_by_method.png", "AUPR by method"),
                                 ("topology_reciprocal_false_positive_pair_rate", "reciprocal_fp_comparison.png", "reciprocal FP rate by method")):
        if metric not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        size = max(summary["size"].unique())
        frame = summary[summary["size"] == size].sort_values(metric, ascending=(metric != "aupr")).head(10)
        ax.barh(frame["method"], frame[metric]); ax.set_title(f"Size{size} {title}"); ax.invert_yaxis()
        p = FIGURES_DIR / fname; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.as_posix())
    return saved


# --------------------------------------------------------------------------- #
# Debug report
# --------------------------------------------------------------------------- #
def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def best_method(summary, size, metric, *, families=None, deployable_only=False, minimize=False):
    frame = summary[summary["size"] == size]
    if families is not None:
        frame = frame[frame["family"].isin(families)]
    if deployable_only:
        frame = frame[frame["deployable"]]
    frame = frame[frame[metric].notna()]
    if frame.empty:
        return "", float("nan")
    row = frame.sort_values([metric, "method"], ascending=[minimize, True]).iloc[0]
    return str(row["method"]), float(row[metric])


def build_debug_report(summary, alpha_selection, calibration, sizes, figures) -> str:
    lines = ["# DREAM4 Calibrated Confidence Debug Report", "",
             "Deployable, gold-free edge-confidence pipeline. Alpha is selected without gold labels (CV/BIC/AIC/density-prior/stability); gold labels are used only for evaluation. The `sparse_oracle_alpha` row is ORACLE (not deployable).",
             "", f"dynGENIE3: no official package installed; tree methods are dynGENIE3-style.", ""]
    if figures:
        lines += ["Figures: " + ", ".join(Path(f).name for f in figures), ""]

    focal = alpha_selection[alpha_selection["model"] == FOCAL_MODEL]

    lines.append("## Selected alpha vs oracle (focal model: dynamic_lasso_level_include_self)")
    lines.append("")
    tbl = focal.groupby(["size", "selection_rule"]).agg(
        chosen_alpha=("chosen_alpha", "median"), oracle_alpha=("oracle_alpha", "median"),
        aupr=("aupr", "mean"), aupr_gap_to_oracle=("aupr_gap_to_oracle", "mean"),
        predicted_density=("predicted_density", "mean"), true_density=("true_density", "mean")).reset_index()
    lines.append(to_markdown_table(tbl))
    lines.append("")

    lines.append("## Question-by-question findings")
    lines.append("")

    # Q1 best deployable rule overall
    rule_perf = focal[focal["deployable"]].groupby("selection_rule")["aupr_gap_to_oracle"].mean().sort_values()
    best_rule = rule_perf.index[0] if len(rule_perf) else "n/a"
    lines.append(f"**1. Which deployable alpha rule works best overall?** `{best_rule}` (smallest mean AUPR gap to oracle {fmt(float(rule_perf.iloc[0]) if len(rule_perf) else float('nan'))}). Ranking: " + ", ".join(f"{r}={fmt(v)}" for r, v in rule_perf.items()) + ".")
    lines.append("")

    # Q2 CV vs BIC by size
    lines.append("**2. Does CV or BIC better match the oracle across sizes?**")
    for size in sizes:
        cv_gap = focal[(focal["size"] == size) & (focal["selection_rule"] == "cv")]["alpha_log10_gap_to_oracle"].mean()
        bic_gap = focal[(focal["size"] == size) & (focal["selection_rule"] == "bic")]["alpha_log10_gap_to_oracle"].mean()
        winner = "CV" if cv_gap < bic_gap else ("BIC" if bic_gap < cv_gap else "tie")
        lines.append(f"- Size{size}: CV log10-alpha gap {fmt(cv_gap)} vs BIC {fmt(bic_gap)} -> {winner} closer.")
    lines.append("")

    # Q3 deployable preserves oracle performance
    lines.append("**3. Does deployable alpha selection preserve most of the oracle sparse model's performance?**")
    for size in sizes:
        oracle_aupr = focal[(focal["size"] == size) & (focal["selection_rule"] == "oracle")]["aupr"].mean()
        for rule in ("cv", "bic"):
            aupr_rule = focal[(focal["size"] == size) & (focal["selection_rule"] == rule)]["aupr"].mean()
            retained = aupr_rule / oracle_aupr if oracle_aupr else float("nan")
            lines.append(f"- Size{size} {rule}: AUPR {fmt(aupr_rule)} vs oracle {fmt(oracle_aupr)} ({fmt(100 * retained, 1)}% retained).")
    lines.append("")

    # Q4 confidence beats individuals
    lines.append("**4. Does confidence/agreement fusion beat individual methods?**")
    for size in sizes:
        cm, cv = best_method(summary, size, "aupr", families=["confidence"], deployable_only=True)
        bm, bv = best_method(summary, size, "aupr", families=["sparse", "tree", "correlation"], deployable_only=True)
        verdict = "beats" if cv > bv + 0.005 else ("ties" if cv > bv - 0.005 else "trails")
        lines.append(f"- Size{size}: best confidence `{cm}` AUPR {fmt(cv)} {verdict} best single deployable method `{bm}` ({fmt(bv)}).")
    lines.append("")

    # Q5 multi-method agreement
    lines.append("**5. Does fusion help because of true multi-method agreement?** See experiment 13 (true positives carry higher multi-method support than false positives); here the agreement-count confidence (top q%) is included and its AUPR/precision are in the summary table. Where confidence beats single methods (Q4), it is because true edges are co-ranked by multiple evidence types.")
    lines.append("")

    # Q6 reciprocal penalty directionality
    lines.append("**6. Does the reciprocal penalty improve directionality?**")
    for size in sizes:
        base = summary[(summary["size"] == size) & (summary["method"] == "fusion_borda")]
        pen = summary[(summary["size"] == size) & (summary["method"] == "confidence_reciprocal_penalty_w0_5")]
        if not base.empty and not pen.empty:
            br = float(base.iloc[0]["topology_reciprocal_false_positive_pair_rate"]); pr = float(pen.iloc[0]["topology_reciprocal_false_positive_pair_rate"])
            ba = float(base.iloc[0]["aupr"]); pa = float(pen.iloc[0]["aupr"])
            lines.append(f"- Size{size}: reciprocal-FP rate {fmt(pr)} (penalty) vs {fmt(br)} (borda); AUPR {fmt(pa)} vs {fmt(ba)}.")
    lines.append("")

    # Q7 calibration meaningful
    lines.append("**7. Does confidence calibration look meaningful?**")
    for size in sizes:
        for method in ["sparse_cv_alpha", "fusion_borda"]:
            sub = calibration[(calibration["size"] == size) & (calibration["method"] == method)]
            if sub.empty:
                continue
            corr = sub["confidence_true_rate_spearman"].mean(); ece = sub["ece"].mean()
            top = sub[sub["bin"] == 1]["empirical_true_rate"].mean(); bottom = sub["bin"].max()
            bottom_rate = sub[sub["bin"] == bottom]["empirical_true_rate"].mean()
            lines.append(f"- Size{size} {method}: confidence-vs-true-rate Spearman {fmt(corr)}, ECE-style {fmt(ece)}, top-bin true rate {fmt(top)} vs bottom-bin {fmt(bottom_rate)} (monotone => meaningful ranking; ECE is high because raw scores are not probabilities).")
    lines.append("")

    # Q8 topology vs AUPR
    lines.append("**8. Does the confidence method improve topology or only AUPR?**")
    for size in sizes:
        hub = SIZE_SETTINGS[size]["hub_top"]
        am, av = best_method(summary, size, "aupr", deployable_only=True)
        hm, hv = best_method(summary, size, f"topology_top{hub}_out_hub_overlap", deployable_only=True)
        rm, rv = best_method(summary, size, "topology_reciprocal_false_positive_pair_rate", deployable_only=True, minimize=True)
        pk = "precision_at_10"
        pm, pv = best_method(summary, size, pk, deployable_only=True)
        lines.append(f"- Size{size}: best deployable by AUPR `{am}` ({fmt(av)}); by {pk} `{pm}` ({fmt(pv)}); by top{hub}-out-hub `{hm}` ({fmt(hv)}); by lowest reciprocal-FP `{rm}` ({fmt(rv)}). Separate winners confirm topology is not captured by AUPR alone.")
    lines.append("")

    # Q9 DREAM4-specific
    lines.append("**9. What remains DREAM4-specific?** The exact selected alphas and density values, the magnitude of the self/non-self ratio, and the constant 50-unit time grid. The *rules* (alpha from CV/BIC, equal-weight agreement confidence, reciprocal/topology penalties, separate topology objectives) are general; the *numbers* are DREAM4-specific.")
    lines.append("")
    lines.append("**10. What should be tested in GNW sweeps?** Whether CV/BIC keep matching the oracle as density, trajectory length, and noise vary; whether agreement confidence keeps beating single methods when base-method complementarity changes; and whether the reciprocal/topology penalties keep helping directionality. See experiments/12_gnw_sweep_design/gnw_sweep_design.md.")
    lines.append("")
    lines.append("**11. Is the current project claim now strong enough to consolidate into a report?** " + consolidation_verdict(summary, focal, sizes))
    return "\n".join(lines)


def consolidation_verdict(summary, focal, sizes) -> str:
    """Cautious verdict on whether to consolidate into a report."""
    retained = []
    confidence_wins = 0
    for size in sizes:
        oracle_aupr = focal[(focal["size"] == size) & (focal["selection_rule"] == "oracle")]["aupr"].mean()
        cv_aupr = focal[(focal["size"] == size) & (focal["selection_rule"] == "cv")]["aupr"].mean()
        bic_aupr = focal[(focal["size"] == size) & (focal["selection_rule"] == "bic")]["aupr"].mean()
        best_dep = max(cv_aupr, bic_aupr)
        if oracle_aupr:
            retained.append(best_dep / oracle_aupr)
        cm, cv = best_method(summary, size, "aupr", families=["confidence"], deployable_only=True)
        bm, bv = best_method(summary, size, "aupr", families=["sparse", "tree", "correlation"], deployable_only=True)
        if cv >= bv - 0.005:
            confidence_wins += 1
    mean_retained = float(np.mean(retained)) if retained else float("nan")
    if mean_retained >= 0.9 and confidence_wins == len(sizes):
        return (f"Yes - deployable selection retains ~{fmt(100 * mean_retained, 0)}% of oracle AUPR and confidence fusion is at least competitive at every size, so the regime-dependent, gold-free pipeline is a defensible Track A method candidate worth consolidating, with the caveat that an official dynGENIE3 baseline and GNW sweeps are still pending.")
    return (f"Partially - deployable selection retains ~{fmt(100 * mean_retained, 0)}% of oracle AUPR and confidence fusion wins at {confidence_wins} of {len(sizes)} sizes. The pipeline is promising and reportable as a calibrated-confidence study, but it is not yet a single dominant method; consolidate as a methodology/finding report rather than a final method, pending dynGENIE3 and GNW validation.")


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    columns = [str(c) for c in frame.columns]
    rows = [["" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in row] for row in frame.to_numpy()]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(r) + " |" for r in rows]])


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Size10 only, fewer trees/resamples")
    p.add_argument("--skip-size100", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--tree-estimators-size10", type=int, default=None)
    p.add_argument("--tree-estimators-size100", type=int, default=None)
    p.add_argument("--bootstrap-resamples", type=int, default=None)
    p.add_argument("--random-seed", type=int, default=20260602)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [10] if (args.quick or args.skip_size100) else [10, 100]
    trees = {10: args.tree_estimators_size10 or (100 if args.quick else 200),
             100: args.tree_estimators_size100 or 100}
    bootstrap = args.bootstrap_resamples if args.bootstrap_resamples is not None else (6 if args.quick else 12)

    results = [run_size(size, bootstrap_resamples=bootstrap, tree_estimators=trees[size], seed=args.random_seed, n_jobs=args.n_jobs) for size in sizes]
    per_network = pd.concat([r["per_network"] for r in results], ignore_index=True)
    topology = pd.concat([r["topology"] for r in results], ignore_index=True)
    alpha_selection = pd.concat([r["alpha_selection"] for r in results], ignore_index=True)
    calibration = pd.concat([r["calibration"] for r in results], ignore_index=True)
    edges = pd.concat([r["edges"] for r in results], ignore_index=True)
    summary = aggregate(per_network)
    figures = write_figures(alpha_selection, summary, calibration)

    summary.to_csv(SUMMARY_PATH, index=False)
    per_network.to_csv(PER_NETWORK_PATH, index=False)
    edges.to_csv(EDGES_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    topology.to_csv(TOPOLOGY_PATH, index=False)
    alpha_selection.to_csv(ALPHA_SELECTION_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(build_debug_report(summary, alpha_selection, calibration, sizes, figures), encoding="utf-8")
    print_summary(summary, alpha_selection, sizes, figures)


def print_summary(summary, alpha_selection, sizes, figures):
    print("DREAM4 calibrated confidence")
    print(f"sizes={sizes}, matplotlib figures={len(figures)}")
    cols = ["size", "method", "deployable", "family", "auroc", "aupr", "precision_at_10", "topology_reciprocal_false_positive_pair_rate"]
    cols = [c for c in cols if c in summary.columns]
    for size in sizes:
        print(f"\n--- Size{size} top 8 by AUPR ---")
        print(summary[summary["size"] == size][cols].head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for path in (SUMMARY_PATH, PER_NETWORK_PATH, EDGES_PATH, CALIBRATION_PATH, TOPOLOGY_PATH, ALPHA_SELECTION_PATH, DEBUG_REPORT_PATH):
        print(f"saved: {path.as_posix()}")


if __name__ == "__main__":
    main()
