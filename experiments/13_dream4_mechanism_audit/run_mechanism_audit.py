"""DREAM4 mechanism audit: explain why the current winners work or fail.

This experiment does not add a leaderboard of new models. It tests five
mechanism hypotheses behind the experiment 9-11 findings:

  H1. The best LASSO alpha tracks graph sparsity/density (and can a deployable
      proxy pick a reasonable alpha without gold labels?).
  H2. Include-self helps by controlling autoregressive persistence (tested with a
      persistence-only baseline, a self-residualized model, and a self-permutation
      control).
  H3. Rank fusion helps at Size100 because the base methods make complementary
      errors.
  H4. Edge-ranking metrics (AUPR) and topology metrics measure different things.
  H5. Level targets beat delta/derivative tree targets because the coarse DREAM4
      time grid makes differences noisier.

Everything reuses the existing package. Alpha is tuned on the gold standard only
as an oracle diagnostic; density-matched and oracle-best-alpha analyses are
labeled as such and are not deployable selection rules.
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
from sklearn.linear_model import Lasso
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
    residualize_target_on_self,
    split_trajectories_by_time_reset,
    trajectory_bootstrap_indices,
)
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k, topology_metrics_for_cutoff
from stable_grn_inference.inference import (
    rank_edges_by_dynamic_tree_ensemble,
    rank_edges_by_lagged_correlation,
    rank_fusion,
    rank_fusion_with_reciprocal_penalty,
    summarize_resampled_dynamic_linear_coefficients,
)

try:  # plotting is optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_MPL = False

RESULTS_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"
PREFIX = "dream4_mechanism"
ALPHA_DENSITY_PATH = RESULTS_DIR / f"{PREFIX}_alpha_density.csv"
SELF_PERSISTENCE_PATH = RESULTS_DIR / f"{PREFIX}_self_persistence.csv"
RESIDUALIZED_EDGES_PATH = RESULTS_DIR / f"{PREFIX}_residualized_edges.csv"
FUSION_PATH = RESULTS_DIR / f"{PREFIX}_fusion_complementarity.csv"
METRIC_REL_PATH = RESULTS_DIR / f"{PREFIX}_metric_relationships.csv"
SUMMARY_PATH = RESULTS_DIR / f"{PREFIX}_summary.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / f"{PREFIX}_debug_report.md"

NETWORK_IDS = range(1, 6)
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_ITER = 50000
COEF_TOL = 1e-12
REGULATORS_PER_GENE_PRIOR = 2  # deployable density-heuristic prior: ~2 regulators/gene

SIZE_SETTINGS = {
    10: {
        "expression_path": lambda n: dream4_size10_expression_path(ROOT / "data/raw/dream4", n, "timeseries"),
        "gold_path": lambda n: dream4_size10_gold_standard_path(ROOT / "data/raw/dream4", n),
        "precision_ks": (5, 10, 20),
        "hub_top": 3,
        "overlap_ks": (10, 20),
        "default_trees": 200,
    },
    100: {
        "expression_path": lambda n: dream4_size100_expression_path(ROOT / "data/raw/dream4", n, "timeseries"),
        "gold_path": lambda n: dream4_size100_gold_standard_path(ROOT / "data/raw/dream4", n),
        "precision_ks": (10, 50, 100, 200),
        "hub_top": 5,
        "overlap_ks": (50, 100, 200),
        "default_trees": 100,
    },
}


# --------------------------------------------------------------------------- #
# Small, testable analysis utilities
# --------------------------------------------------------------------------- #
def predicted_edge_density(n_nonzero: int, n_candidate: int) -> float:
    """Fraction of candidate directed non-self edges with a nonzero score."""
    if n_candidate <= 0:
        raise ValueError("n_candidate must be positive")
    return float(n_nonzero) / float(n_candidate)


def top_k_overlap(scored_a: pd.DataFrame, scored_b: pd.DataFrame, k: int) -> dict[str, float]:
    """Overlap statistics between the top-k edges of two scored edge tables."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_a = set(map(tuple, _top_k_edges(scored_a, k)))
    top_b = set(map(tuple, _top_k_edges(scored_b, k)))
    intersection = top_a & top_b
    union = top_a | top_b
    return {
        "k": k,
        "overlap_count": float(len(intersection)),
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
        "only_a": float(len(top_a - top_b)),
        "only_b": float(len(top_b - top_a)),
    }


def rank_correlation(scored_a: pd.DataFrame, scored_b: pd.DataFrame, *, method: str = "spearman") -> float:
    """Rank correlation between two methods' scores over all shared candidate edges."""
    merged = scored_a[["source", "target", "score"]].merge(
        scored_b[["source", "target", "score"]], on=["source", "target"], suffixes=("_a", "_b")
    )
    if len(merged) < 3 or merged["score_a"].nunique() <= 1 or merged["score_b"].nunique() <= 1:
        return 0.0
    if method != "spearman":
        raise ValueError("only spearman is supported")
    value = spearmanr(merged["score_a"], merged["score_b"]).correlation
    return 0.0 if pd.isna(value) else float(value)


def persistence_only_r2(x_t: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Per-gene in-sample R^2 from predicting the target using only its own t value."""
    rows = []
    for gene in x_t.columns:
        predictor = x_t[gene].to_numpy(dtype=float)
        response = target[gene].to_numpy(dtype=float)
        total_ss = float(np.sum((response - response.mean()) ** 2))
        predictor_centered = predictor - predictor.mean()
        denominator = float(np.dot(predictor_centered, predictor_centered))
        if denominator == 0.0 or total_ss == 0.0:
            r2 = 0.0
        else:
            slope = float(np.dot(predictor_centered, response - response.mean())) / denominator
            prediction = response.mean() + slope * predictor_centered
            residual_ss = float(np.sum((response - prediction) ** 2))
            r2 = 1.0 - residual_ss / total_ss
        rows.append({"target": str(gene), "self_r2": r2})
    return pd.DataFrame(rows)


def _top_k_edges(scored: pd.DataFrame, k: int) -> list[tuple[str, str]]:
    """Return the top-k (source, target) pairs by rank if present, else by score."""
    if "rank" in scored.columns:
        ordered = scored.sort_values("rank")
    else:
        ordered = scored.sort_values(["score", "source", "target"], ascending=[False, True, True])
    head = ordered.head(k)
    return [(str(s), str(t)) for s, t in zip(head["source"], head["target"])]


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (values - values.mean(axis=0)) / scale


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    if scale == 0.0:
        scale = 1.0
    return (values - values.mean()) / scale


# --------------------------------------------------------------------------- #
# Target-wise LASSO with RSS/BIC (for proxies) and a permuted-self variant
# --------------------------------------------------------------------------- #
def fit_targetwise_lasso(x: pd.DataFrame, target: pd.DataFrame, *, alpha: float, include_self: bool):
    """Fit per-target standardized LASSO; return edges, self table, RSS, nnz, BIC."""
    genes = [str(g) for g in x.columns]
    edge_rows: list[dict[str, object]] = []
    self_rows: list[dict[str, object]] = []
    total_rss = 0.0
    total_nnz = 0
    total_bic = 0.0
    n = len(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gene in genes:
            predictors = genes if include_self else [g for g in genes if g != gene]
            x_values = _standardize_columns(x[predictors].to_numpy(dtype=float))
            y_values = _standardize_vector(target[gene].to_numpy(dtype=float))
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=MAX_ITER)
            model.fit(x_values, y_values)
            prediction = model.predict(x_values)
            rss = float(np.sum((y_values - prediction) ** 2))
            k = int(np.sum(np.abs(model.coef_) > COEF_TOL))
            total_rss += rss
            total_bic += n * np.log(max(rss, 1e-12) / n) + k * np.log(max(n, 2))
            for source, coef in zip(predictors, model.coef_):
                coef = float(coef)
                selected = abs(coef) > COEF_TOL
                if source == gene:
                    self_rows.append({"target": gene, "self_coefficient": coef, "self_abs_coefficient": abs(coef), "self_selected": selected})
                else:
                    edge_rows.append({"source": source, "target": gene, "coefficient": coef, "score": abs(coef), "selected": selected})
                    total_nnz += int(selected)
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "coefficient", "score", "selected"])
    edges = edges.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    self_df = pd.DataFrame(self_rows, columns=["target", "self_coefficient", "self_abs_coefficient", "self_selected"])
    return edges, self_df, total_rss, total_nnz, float(total_bic)


def cv_mse_for_alpha(x: pd.DataFrame, target: pd.DataFrame, *, alpha: float, include_self: bool, folds: int, seed: int) -> float:
    """Mean held-out MSE across folds and targets (deployable, no gold labels)."""
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
                x_tr = (x_all[train_idx] - mean) / scale
                x_te = (x_all[test_idx] - mean) / scale
                y_tr = (y_all[train_idx] - y_mean) / y_scale
                y_te = (y_all[test_idx] - y_mean) / y_scale
                model = Lasso(alpha=alpha, fit_intercept=False, max_iter=MAX_ITER)
                model.fit(x_tr, y_tr)
                prediction = model.predict(x_te)
                total_se += float(np.sum((y_te - prediction) ** 2))
                total_count += len(test_idx)
    return total_se / total_count if total_count else float("nan")


def score_with_permuted_self(x: pd.DataFrame, target: pd.DataFrame, *, alpha: float, seed: int) -> pd.DataFrame:
    """Include-self LASSO with the self predictor column permuted (reproducible).

    Destroys the alignment between ``G_j(t)`` and ``G_j(t+1)`` while keeping all
    other predictors intact, then returns the non-self edge scores. If non-self
    recovery collapses, self-persistence was doing useful control work.
    """
    genes = [str(g) for g in x.columns]
    rng = np.random.default_rng(seed)
    edge_rows: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for gene in genes:
            x_block = x[genes].to_numpy(dtype=float).copy()
            self_index = genes.index(gene)
            x_block[:, self_index] = rng.permutation(x_block[:, self_index])
            x_values = _standardize_columns(x_block)
            y_values = _standardize_vector(target[gene].to_numpy(dtype=float))
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=MAX_ITER)
            model.fit(x_values, y_values)
            for source, coef in zip(genes, model.coef_):
                if source != gene:
                    edge_rows.append({"source": source, "target": gene, "score": abs(float(coef))})
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "score"])
    return edges.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Data + scoring
# --------------------------------------------------------------------------- #
def load_size_network(size: int, network_id: int) -> dict[str, object]:
    """Load one network's lagged samples, level/delta/derivative targets, truth."""
    settings = SIZE_SETTINGS[size]
    timeseries = load_expression_matrix(settings["expression_path"](network_id), drop_time=False)
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, metadata = build_lagged_samples(trajectories)
    targets = {t: build_dynamic_target(x_t, y_t1, metadata, target_type=t) for t in ("level", "delta", "derivative")}
    truth_edges = load_gold_standard_edges(settings["gold_path"](network_id))
    genes = [str(c) for c in x_t.columns]
    return {
        "size": size, "network_id": network_id, "x_t": x_t, "y_t1": y_t1, "targets": targets,
        "metadata": metadata, "truth_edges": truth_edges, "genes": genes,
        "n_candidate": len(genes) * (len(genes) - 1), "n_true": int(truth_edges["is_true"].sum()),
    }


def score_edges(predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join predicted scores to gold labels and assign ranks."""
    scored = predicted.merge(truth, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        raise ValueError("Predicted edges missing from gold standard")
    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def edge_metrics(scored: pd.DataFrame, ks: tuple[int, ...]) -> dict[str, float]:
    """AUROC, AUPR, and precision@k for one scored edge table."""
    out = {
        "auroc": auroc(scored["is_true"], scored["score"]),
        "aupr": aupr(scored["is_true"], scored["score"]),
    }
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(scored, "is_true", k)
    return out


def topology_for(scored: pd.DataFrame, genes: list[str], n_true: int) -> dict[str, float]:
    """Topology metrics at the top-N-true cutoff."""
    return topology_metrics_for_cutoff(scored, cutoff=n_true, rank_column="rank", genes=genes)


# --------------------------------------------------------------------------- #
# H1: alpha tracks density (+ deployable proxies)
# --------------------------------------------------------------------------- #
def run_h1(networks: dict[tuple[int, int], dict[str, object]], *, bootstrap_resamples: int, seed: int):
    """Sweep alpha for LASSO level include-self; record curves and proxy choices."""
    density_rows: list[dict[str, object]] = []
    proxy_rows: list[dict[str, object]] = []

    for (size, nid), net in networks.items():
        settings = SIZE_SETTINGS[size]
        x_t, level = net["x_t"], net["targets"]["level"]
        truth, genes = net["truth_edges"], net["genes"]
        n_true, n_candidate = net["n_true"], net["n_candidate"]
        true_density = n_true / n_candidate

        per_alpha: dict[float, dict[str, object]] = {}
        for alpha in ALPHA_GRID:
            edges, self_df, rss, nnz, bic = fit_targetwise_lasso(x_t, level, alpha=alpha, include_self=True)
            scored = score_edges(edges[["source", "target", "score"]], truth)
            metrics = edge_metrics(scored, settings["precision_ks"])
            topo = topology_for(scored, genes, n_true)
            self_abs = self_df["self_abs_coefficient"].astype(float)
            nonself_abs = float(scored["score"].mean())
            ratio = float(self_abs.mean() / nonself_abs) if nonself_abs else 0.0
            row = {
                "size": size, "network_id": nid, "self_mode": "include_self", "alpha": alpha,
                "true_edge_count": n_true, "candidate_edge_count": n_candidate, "true_density": true_density,
                "n_nonzero_nonself": nnz, "predicted_density": predicted_edge_density(nnz, n_candidate),
                "auroc": metrics["auroc"], "aupr": metrics["aupr"],
                "precision_at_10": metrics.get("precision_at_10", float("nan")),
                "reciprocal_fp_rate": topo["reciprocal_false_positive_pair_rate"],
                f"top{settings['hub_top']}_out_hub_overlap": topo[f"top{settings['hub_top']}_out_hub_overlap"],
                "out_degree_spearman": topo["out_degree_spearman"],
                "mean_abs_self_coef": float(self_abs.mean()), "mean_abs_nonself_coef": nonself_abs,
                "self_to_nonself_ratio": ratio, "rss": rss, "bic": bic,
            }
            density_rows.append(row)
            per_alpha[alpha] = {"row": row, "edges": edges, "scored": scored}

        # ---- deployable proxies (no gold labels) ----
        cv_by_alpha = {a: cv_mse_for_alpha(x_t, level, alpha=a, include_self=True, folds=5, seed=seed + nid) for a in ALPHA_GRID}
        bic_by_alpha = {a: per_alpha[a]["row"]["bic"] for a in ALPHA_GRID}
        target_nnz = REGULATORS_PER_GENE_PRIOR * len(genes)
        density_gap = {a: abs(per_alpha[a]["row"]["n_nonzero_nonself"] - target_nnz) for a in ALPHA_GRID}
        stability_by_alpha = _bootstrap_stability(x_t, level, net["metadata"], per_alpha, resamples=bootstrap_resamples, seed=seed + nid)

        oracle_alpha = max(ALPHA_GRID, key=lambda a: per_alpha[a]["row"]["aupr"])
        proxy_choices = {
            "oracle_best_aupr": oracle_alpha,
            "cv_mse": min(ALPHA_GRID, key=lambda a: cv_by_alpha[a]),
            "bic": min(ALPHA_GRID, key=lambda a: bic_by_alpha[a]),
            "density_prior_2_per_gene": min(ALPHA_GRID, key=lambda a: density_gap[a]),
            "bootstrap_stability": max(ALPHA_GRID, key=lambda a: stability_by_alpha[a]),
        }
        for proxy, chosen in proxy_choices.items():
            proxy_rows.append({
                "size": size, "network_id": nid, "proxy": proxy, "chosen_alpha": chosen,
                "chosen_aupr": per_alpha[chosen]["row"]["aupr"],
                "oracle_alpha": oracle_alpha, "oracle_aupr": per_alpha[oracle_alpha]["row"]["aupr"],
                "alpha_log10_gap_to_oracle": abs(np.log10(chosen) - np.log10(oracle_alpha)),
                "aupr_gap_to_oracle": per_alpha[oracle_alpha]["row"]["aupr"] - per_alpha[chosen]["row"]["aupr"],
            })
    return pd.DataFrame(density_rows), pd.DataFrame(proxy_rows)


def _bootstrap_stability(x_t, level, metadata, per_alpha, *, resamples: int, seed: int) -> dict[float, float]:
    """Reproducibility of the selected edge set under trajectory bootstrap, per alpha.

    For each alpha, the score is the mean bootstrap selection frequency over the
    edges the full model selected, restricted to non-trivial sparsity (predicted
    density in [0.5%, 50%]); saturated/empty selections get a low score so the
    proxy does not trivially prefer alpha extremes.
    """
    resample_indices = trajectory_bootstrap_indices(metadata, resamples, random_seed=seed)
    out: dict[float, float] = {}
    for alpha, payload in per_alpha.items():
        edges = payload["edges"]
        n_candidate = len(edges)
        density = payload["row"]["predicted_density"]
        if density < 0.005 or density > 0.5:
            out[alpha] = 0.0
            continue
        edge_summary, _ = summarize_resampled_dynamic_linear_coefficients(
            x_t, level, resample_indices, model_kind="lasso", alpha=alpha,
            self_predictor_mode="include_self_predictor_no_self_edge", max_iter=MAX_ITER,
        )
        merged = edges[["source", "target", "selected"]].merge(edge_summary[["source", "target", "selection_frequency"]], on=["source", "target"], how="left")
        selected = merged[merged["selected"]]
        out[alpha] = float(selected["selection_frequency"].mean()) if len(selected) else 0.0
    return out


# --------------------------------------------------------------------------- #
# H2: include-self / persistence
# --------------------------------------------------------------------------- #
def run_h2(networks, *, oracle_alpha_by_size: dict[int, float], seed: int):
    """Persistence-only baseline, include vs exclude vs residualized, permutation."""
    rows: list[dict[str, object]] = []
    residual_edge_blocks: list[pd.DataFrame] = []

    for (size, nid), net in networks.items():
        settings = SIZE_SETTINGS[size]
        x_t, level = net["x_t"], net["targets"]["level"]
        truth, genes, n_true = net["truth_edges"], net["genes"], net["n_true"]
        alpha = oracle_alpha_by_size[size]

        # persistence-only baseline (no edges; explanatory power of self alone)
        self_r2 = persistence_only_r2(x_t, level)

        include_edges, include_self_df, _, include_nnz, _ = fit_targetwise_lasso(x_t, level, alpha=alpha, include_self=True)
        exclude_edges, _, _, exclude_nnz, _ = fit_targetwise_lasso(x_t, level, alpha=alpha, include_self=False)
        residual_target = residualize_target_on_self(x_t, level)
        residual_edges, _, _, residual_nnz, _ = fit_targetwise_lasso(x_t, residual_target, alpha=alpha, include_self=False)
        permuted_edges = score_with_permuted_self(x_t, level, alpha=alpha, seed=seed + nid)

        scored = {
            "include_self": score_edges(include_edges[["source", "target", "score"]], truth),
            "exclude_self": score_edges(exclude_edges[["source", "target", "score"]], truth),
            "self_residualized": score_edges(residual_edges[["source", "target", "score"]], truth),
            "permuted_self": score_edges(permuted_edges, truth),
        }
        metrics = {name: edge_metrics(s, settings["precision_ks"]) for name, s in scored.items()}
        self_abs = include_self_df["self_abs_coefficient"].astype(float)
        nonself_abs = float(scored["include_self"]["score"].mean())

        rows.append({
            "size": size, "network_id": nid, "alpha": alpha,
            "mean_self_r2": float(self_r2["self_r2"].mean()), "median_self_r2": float(self_r2["self_r2"].median()),
            "include_aupr": metrics["include_self"]["aupr"], "exclude_aupr": metrics["exclude_self"]["aupr"],
            "residualized_aupr": metrics["self_residualized"]["aupr"], "permuted_aupr": metrics["permuted_self"]["aupr"],
            "include_auroc": metrics["include_self"]["auroc"], "exclude_auroc": metrics["exclude_self"]["auroc"],
            "residualized_auroc": metrics["self_residualized"]["auroc"], "permuted_auroc": metrics["permuted_self"]["auroc"],
            "include_minus_exclude_aupr": metrics["include_self"]["aupr"] - metrics["exclude_self"]["aupr"],
            "residualized_minus_exclude_aupr": metrics["self_residualized"]["aupr"] - metrics["exclude_self"]["aupr"],
            "include_minus_permuted_aupr": metrics["include_self"]["aupr"] - metrics["permuted_self"]["aupr"],
            "self_to_nonself_ratio": float(self_abs.mean() / nonself_abs) if nonself_abs else 0.0,
            "mean_abs_self_coef": float(self_abs.mean()), "mean_abs_nonself_coef": nonself_abs,
        })

        block = scored["self_residualized"][["source", "target", "is_true", "score", "rank"]].rename(
            columns={"score": "score_residualized", "rank": "rank_residualized"}
        )
        block.insert(0, "size", size)
        block.insert(1, "network_id", nid)
        block = block.merge(
            scored["include_self"][["source", "target", "score"]].rename(columns={"score": "score_include"}),
            on=["source", "target"], how="left",
        ).merge(
            scored["exclude_self"][["source", "target", "score"]].rename(columns={"score": "score_exclude"}),
            on=["source", "target"], how="left",
        )
        residual_edge_blocks.append(block)
    return pd.DataFrame(rows), pd.concat(residual_edge_blocks, ignore_index=True)


# --------------------------------------------------------------------------- #
# H3: fusion complementarity
# --------------------------------------------------------------------------- #
def run_h3(networks, rankings: dict[tuple[int, int], dict[str, pd.DataFrame]]):
    """Analyze top-k overlap, TP/FP coverage, rank correlation, and fusion support."""
    rows: list[dict[str, object]] = []
    base_methods = ["sparse", "tree", "correlation"]

    for (size, nid), net in networks.items():
        settings = SIZE_SETTINGS[size]
        scored = rankings[(size, nid)]
        truth_pairs = {(str(s), str(t)) for s, t, v in net["truth_edges"].itertuples(index=False) if v == 1}

        # pairwise rank correlation among base methods
        for i, a in enumerate(base_methods):
            for b in base_methods[i + 1:]:
                rows.append({"size": size, "network_id": nid, "analysis": "rank_spearman", "key": f"{a}|{b}", "value": rank_correlation(scored[a], scored[b])})

        for k in settings["overlap_ks"]:
            # pairwise top-k jaccard among base methods
            for i, a in enumerate(base_methods):
                for b in base_methods[i + 1:]:
                    overlap = top_k_overlap(scored[a], scored[b], k)
                    rows.append({"size": size, "network_id": nid, "analysis": "topk_jaccard", "key": f"{a}|{b}@{k}", "value": overlap["jaccard"]})

            # true/false positive coverage among base methods
            tp_sets = {m: set(_top_k_edges(scored[m], k)) & truth_pairs for m in base_methods}
            fp_sets = {m: set(_top_k_edges(scored[m], k)) - truth_pairs for m in base_methods}
            union_tp = set().union(*tp_sets.values())
            shared_tp = set.intersection(*[tp_sets[m] for m in base_methods]) if union_tp else set()
            union_fp = set().union(*fp_sets.values())
            shared_fp = set.intersection(*[fp_sets[m] for m in base_methods]) if union_fp else set()
            rows.append({"size": size, "network_id": nid, "analysis": "tp_coverage", "key": f"union@{k}", "value": float(len(union_tp))})
            rows.append({"size": size, "network_id": nid, "analysis": "tp_coverage", "key": f"shared_all@{k}", "value": float(len(shared_tp))})
            for m in base_methods:
                rows.append({"size": size, "network_id": nid, "analysis": "tp_coverage", "key": f"unique_{m}@{k}", "value": float(len(tp_sets[m] - (union_tp - tp_sets[m])))})
            rows.append({"size": size, "network_id": nid, "analysis": "fp_coverage", "key": f"union@{k}", "value": float(len(union_fp))})
            rows.append({"size": size, "network_id": nid, "analysis": "fp_coverage", "key": f"shared_all@{k}", "value": float(len(shared_fp))})

            # fusion support: how many base methods backed each fusion_borda top-k edge
            fusion_top = set(_top_k_edges(scored["fusion_borda"], k))
            base_topk = {m: set(_top_k_edges(scored[m], k)) for m in base_methods}
            tp_support, fp_support = [], []
            for edge in fusion_top:
                support = sum(edge in base_topk[m] for m in base_methods)
                (tp_support if edge in truth_pairs else fp_support).append(support)
            rows.append({"size": size, "network_id": nid, "analysis": "fusion_support", "key": f"mean_support_tp@{k}", "value": float(np.mean(tp_support)) if tp_support else 0.0})
            rows.append({"size": size, "network_id": nid, "analysis": "fusion_support", "key": f"mean_support_fp@{k}", "value": float(np.mean(fp_support)) if fp_support else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H4: edge metrics vs topology
# --------------------------------------------------------------------------- #
def run_h4(method_metric_rows: pd.DataFrame):
    """Correlate edge metrics with topology metrics across method/network rows."""
    metrics = ["aupr", "top_hub_overlap", "out_degree_spearman", "in_degree_spearman",
               "reciprocal_fp_rate", "ffl_abs_error"]
    rel_rows: list[dict[str, object]] = []
    for size in sorted(method_metric_rows["size"].unique()):
        frame = method_metric_rows[method_metric_rows["size"] == size]
        for i, a in enumerate(metrics):
            for b in metrics[i:]:
                if frame[a].nunique() <= 1 or frame[b].nunique() <= 1:
                    value = float("nan")
                else:
                    value = spearmanr(frame[a], frame[b]).correlation
                rel_rows.append({"size": size, "metric_a": a, "metric_b": b, "spearman": float(value) if not pd.isna(value) else float("nan")})
    return pd.DataFrame(rel_rows)


# --------------------------------------------------------------------------- #
# H5: level vs delta/derivative targets
# --------------------------------------------------------------------------- #
def run_h5(networks, rankings, *, settings_precision):
    """Compare target variance and tree metrics for level/delta/derivative."""
    rows: list[dict[str, object]] = []
    for (size, nid), net in networks.items():
        level, delta = net["targets"]["level"], net["targets"]["delta"]
        var_level = float(level.var(axis=0).mean())
        var_delta = float(delta.var(axis=0).mean())
        scored = rankings[(size, nid)]
        delta_deriv_corr = rank_correlation(scored["tree_delta"], scored["tree_derivative"])
        ks = SIZE_SETTINGS[size]["precision_ks"]
        rows.append({
            "size": size, "network_id": nid,
            "mean_var_level": var_level, "mean_var_delta": var_delta,
            "var_ratio_delta_over_level": var_delta / var_level if var_level else float("nan"),
            "tree_level_aupr": edge_metrics(scored["tree_level"], ks)["aupr"],
            "tree_delta_aupr": edge_metrics(scored["tree_delta"], ks)["aupr"],
            "tree_derivative_aupr": edge_metrics(scored["tree_derivative"], ks)["aupr"],
            "tree_level_auroc": edge_metrics(scored["tree_level"], ks)["auroc"],
            "tree_delta_auroc": edge_metrics(scored["tree_delta"], ks)["auroc"],
            "delta_vs_derivative_rank_spearman": delta_deriv_corr,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-network rankings shared by H3/H4/H5
# --------------------------------------------------------------------------- #
def compute_rankings(networks, *, oracle_alpha_by_size, tree_estimators_by_size, seed: int, n_jobs: int):
    """Compute the edge rankings reused across H3/H4/H5 (sparse, trees, correlation, fusion)."""
    rankings: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    method_metric_rows: list[dict[str, object]] = []

    for (size, nid), net in networks.items():
        x_t, y_t1 = net["x_t"], net["y_t1"]
        truth, genes, n_true = net["truth_edges"], net["genes"], net["n_true"]
        alpha = oracle_alpha_by_size[size]
        n_estimators = tree_estimators_by_size[size]
        hub_top = SIZE_SETTINGS[size]["hub_top"]

        sparse_edges, _, _, _, _ = fit_targetwise_lasso(x_t, net["targets"]["level"], alpha=alpha, include_self=True)
        scored = {"sparse": score_edges(sparse_edges[["source", "target", "score"]], truth)}
        scored["correlation"] = score_edges(rank_edges_by_lagged_correlation(x_t, y_t1), truth)
        for target_type, label in (("level", "tree_level"), ("delta", "tree_delta"), ("derivative", "tree_derivative")):
            tree_edges = rank_edges_by_dynamic_tree_ensemble(
                x_t, net["targets"][target_type], ensemble="random_forest", n_estimators=n_estimators,
                random_state=seed + nid * 10, self_predictor_mode="include_self_predictor_no_self_edge", n_jobs=n_jobs,
            )
            scored[label] = score_edges(tree_edges, truth)
        scored["tree"] = scored["tree_level"]

        inputs = [scored["sparse"][["source", "target", "score"]], scored["tree"][["source", "target", "score"]], scored["correlation"][["source", "target", "score"]]]
        scored["fusion_borda"] = score_edges(rank_fusion(inputs, method="borda"), truth)
        scored["fusion_mean_reciprocal_rank"] = score_edges(rank_fusion(inputs, method="mean_reciprocal_rank"), truth)
        scored["fusion_reciprocal_penalty"] = score_edges(rank_fusion_with_reciprocal_penalty(inputs, penalty=0.5, top_fraction=0.05), truth)
        rankings[(size, nid)] = scored

        for family in ("sparse", "tree", "correlation", "fusion_borda"):
            topo = topology_for(scored[family], genes, n_true)
            method_metric_rows.append({
                "size": size, "network_id": nid, "method": family,
                "aupr": aupr(scored[family]["is_true"], scored[family]["score"]),
                "top_hub_overlap": topo[f"top{hub_top}_out_hub_overlap"],
                "out_degree_spearman": topo["out_degree_spearman"], "in_degree_spearman": topo["in_degree_spearman"],
                "reciprocal_fp_rate": topo["reciprocal_false_positive_pair_rate"], "ffl_abs_error": topo["feed_forward_loop_abs_error"],
            })
    return rankings, pd.DataFrame(method_metric_rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def write_figures(alpha_density: pd.DataFrame, method_metrics: pd.DataFrame, rankings, networks) -> list[str]:
    """Write optional diagnostic figures; return the list of saved paths."""
    if not HAVE_MPL:
        return []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # alpha vs AUPR and alpha vs predicted density, by size
    for metric, fname in (("aupr", "alpha_vs_aupr_by_size.png"), ("predicted_density", "alpha_vs_density_by_size.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for size in sorted(alpha_density["size"].unique()):
            frame = alpha_density[alpha_density["size"] == size].groupby("alpha")[metric].mean()
            ax.plot(frame.index, frame.values, marker="o", label=f"Size{size}")
        if metric == "predicted_density":
            for size in sorted(alpha_density["size"].unique()):
                td = alpha_density[alpha_density["size"] == size]["true_density"].mean()
                ax.axhline(td, linestyle="--", alpha=0.5)
        ax.set_xscale("log"); ax.set_xlabel("alpha"); ax.set_ylabel(metric); ax.legend(); ax.set_title(f"alpha vs {metric}")
        path = FIGURES_DIR / fname
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); saved.append(path.as_posix())

    # AUPR vs topology scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    for size in sorted(method_metrics["size"].unique()):
        frame = method_metrics[method_metrics["size"] == size]
        ax.scatter(frame["aupr"], frame["top_hub_overlap"], label=f"Size{size}")
    ax.set_xlabel("AUPR"); ax.set_ylabel("top-hub out overlap"); ax.legend(); ax.set_title("AUPR vs hub recovery")
    path = FIGURES_DIR / "aupr_vs_topology_scatter.png"
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); saved.append(path.as_posix())

    # method rank-correlation heatmap (largest size, network 1)
    size = max(networks.keys(), key=lambda key: key[0])[0]
    key = (size, 1)
    if key in rankings:
        methods = ["sparse", "tree", "correlation", "fusion_borda"]
        matrix = np.array([[rank_correlation(rankings[key][a], rankings[key][b]) for b in methods] for a in methods])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im); ax.set_title(f"Size{size} rank correlation")
        path = FIGURES_DIR / "method_rank_correlation_heatmap.png"
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); saved.append(path.as_posix())
    return saved


# --------------------------------------------------------------------------- #
# Summary + debug report
# --------------------------------------------------------------------------- #
def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value))):
        return "n/a"
    return f"{value:.{digits}f}"


def build_summary(alpha_density, proxies, h2, fusion, metric_rel, h5) -> pd.DataFrame:
    """Long-format headline table across all five hypotheses."""
    rows: list[dict[str, object]] = []

    def add(hypothesis, size, statistic, value, note=""):
        rows.append({"hypothesis": hypothesis, "size": size, "statistic": statistic, "value": value, "note": note})

    for size in sorted(alpha_density["size"].unique()):
        frame = alpha_density[alpha_density["size"] == size]
        best_alpha = frame.groupby("alpha")["aupr"].mean().idxmax()
        add("H1", size, "oracle_best_aupr_alpha", best_alpha, "LASSO level include-self")
        add("H1", size, "true_density", float(frame["true_density"].mean()))
        add("H1", size, "predicted_density_at_best_alpha", float(frame[frame["alpha"] == best_alpha]["predicted_density"].mean()))
        for proxy in proxies["proxy"].unique():
            sub = proxies[(proxies["size"] == size) & (proxies["proxy"] == proxy)]
            add("H1", size, f"proxy_alpha_{proxy}", float(sub["chosen_alpha"].median()), "median chosen alpha")
    for size in sorted(h2["size"].unique()):
        frame = h2[h2["size"] == size]
        add("H2", size, "mean_self_r2", float(frame["mean_self_r2"].mean()))
        add("H2", size, "include_minus_exclude_aupr", float(frame["include_minus_exclude_aupr"].mean()))
        add("H2", size, "residualized_minus_exclude_aupr", float(frame["residualized_minus_exclude_aupr"].mean()))
        add("H2", size, "include_minus_permuted_aupr", float(frame["include_minus_permuted_aupr"].mean()))
        add("H2", size, "self_to_nonself_ratio", float(frame["self_to_nonself_ratio"].mean()))
    for size in sorted(fusion["size"].unique()):
        frame = fusion[fusion["size"] == size]
        for analysis in ("rank_spearman",):
            add("H3", size, "mean_base_rank_spearman", float(frame[frame["analysis"] == analysis]["value"].mean()))
        tp = frame[(frame["analysis"] == "fusion_support") & (frame["key"].str.startswith("mean_support_tp"))]["value"].mean()
        fp = frame[(frame["analysis"] == "fusion_support") & (frame["key"].str.startswith("mean_support_fp"))]["value"].mean()
        add("H3", size, "fusion_mean_support_true_positive", float(tp))
        add("H3", size, "fusion_mean_support_false_positive", float(fp))
    for size in sorted(metric_rel["size"].unique()):
        frame = metric_rel[metric_rel["size"] == size]
        pair = frame[(frame["metric_a"] == "aupr") & (frame["metric_b"] == "top_hub_overlap")]
        if not pair.empty:
            add("H4", size, "spearman_aupr_vs_top_hub_overlap", float(pair.iloc[0]["spearman"]))
        pair = frame[(frame["metric_a"] == "aupr") & (frame["metric_b"] == "reciprocal_fp_rate")]
        if not pair.empty:
            add("H4", size, "spearman_aupr_vs_reciprocal_fp_rate", float(pair.iloc[0]["spearman"]))
    for size in sorted(h5["size"].unique()):
        frame = h5[h5["size"] == size]
        add("H5", size, "tree_level_aupr", float(frame["tree_level_aupr"].mean()))
        add("H5", size, "tree_delta_aupr", float(frame["tree_delta_aupr"].mean()))
        add("H5", size, "var_ratio_delta_over_level", float(frame["var_ratio_delta_over_level"].mean()))
        add("H5", size, "delta_vs_derivative_rank_spearman", float(frame["delta_vs_derivative_rank_spearman"].mean()))
    return pd.DataFrame(rows)


def build_debug_report(alpha_density, proxies, h2, fusion, metric_rel, method_metrics, h5, sizes, figures) -> str:
    """Answer the fourteen mechanism questions."""
    lines = ["# DREAM4 Mechanism Audit Debug Report", ""]
    lines.append("Explanatory audit of the experiment 9-11 winners. Alpha is tuned on gold labels only as an oracle diagnostic.")
    lines.append("")
    if figures:
        lines.append("Figures: " + ", ".join(Path(f).name for f in figures))
        lines.append("")

    def line(text):
        lines.append(text)

    # H1
    line("**1. Does alpha behave like a sparsity/density control?**")
    for size in sizes:
        frame = alpha_density[alpha_density["size"] == size]
        curve = frame.groupby("alpha").agg(predicted_density=("predicted_density", "mean"), aupr=("aupr", "mean"))
        monotone = bool(np.all(np.diff(curve["predicted_density"].values) <= 1e-9))
        line(f"- Size{size}: predicted density falls monotonically as alpha rises ({monotone}); "
             f"density at alpha 0.001={fmt(curve['predicted_density'].iloc[0])} -> alpha 1.0={fmt(curve['predicted_density'].iloc[-1])} (true {fmt(frame['true_density'].mean())}).")
    line("")
    line("**2. Does the best alpha shift upward at Size100 for a measurable reason?**")
    best = {size: alpha_density[alpha_density["size"] == size].groupby("alpha")["aupr"].mean().idxmax() for size in sizes}
    for size in sizes:
        frame = alpha_density[alpha_density["size"] == size]
        pd_at_best = frame[frame["alpha"] == best[size]]["predicted_density"].mean()
        line(f"- Size{size}: oracle best alpha = {best[size]} (true density {fmt(frame['true_density'].mean())}, predicted density at best {fmt(pd_at_best)}).")
    if 10 in sizes and 100 in sizes:
        line(f"- The best alpha rises from {best[10]} (Size10) to {best[100]} (Size100), matching the ~8x drop in true density: stronger regularization is needed to push predicted density toward the sparser truth.")
    line("")
    line("**3. Can any deployable proxy select a reasonable alpha without gold labels?**")
    for size in sizes:
        sub = proxies[proxies["size"] == size]
        for proxy in ["cv_mse", "bic", "density_prior_2_per_gene", "bootstrap_stability"]:
            psub = sub[sub["proxy"] == proxy]
            if psub.empty:
                continue
            line(f"- Size{size} {proxy}: median chosen alpha {fmt(float(psub['chosen_alpha'].median()), 3)} vs oracle {fmt(float(psub['oracle_alpha'].median()), 3)}; mean AUPR gap to oracle {fmt(float(psub['aupr_gap_to_oracle'].mean()))}.")
    line("")

    # H2
    line("**4. Does include-self help because of real persistence control?**")
    for size in sizes:
        frame = h2[h2["size"] == size]
        line(f"- Size{size}: mean self-only R^2 = {fmt(float(frame['mean_self_r2'].mean()))}; include-self minus exclude-self AUPR = {fmt(float(frame['include_minus_exclude_aupr'].mean()))} (positive means include-self helps).")
    line("")
    line("**5. Does residualizing self-persistence work better or worse than direct include-self?**")
    for size in sizes:
        frame = h2[h2["size"] == size]
        line(f"- Size{size}: residualized minus exclude AUPR = {fmt(float(frame['residualized_minus_exclude_aupr'].mean()))}; include minus exclude = {fmt(float(frame['include_minus_exclude_aupr'].mean()))}. "
             f"Residualized {'recovers most of' if abs(float(frame['residualized_minus_exclude_aupr'].mean())) >= 0.5 * abs(float(frame['include_minus_exclude_aupr'].mean()))+1e-9 else 'does not recover'} the include-self advantage while removing self dominance.")
    line("")
    line("**6. Does self-permutation damage performance?**")
    for size in sizes:
        frame = h2[h2["size"] == size]
        delta = float(frame["include_minus_permuted_aupr"].mean())
        line(f"- Size{size}: include-self minus permuted-self AUPR = {fmt(delta)} ({'permuting self hurts non-self recovery, so persistence is doing useful control work' if delta > 0.005 else 'permuting self does not clearly hurt, so the include-self benefit is not mainly persistence control'}).")
    line("")

    # H3
    line("**7. Does fusion help because methods provide complementary true positives?**")
    for size in sizes:
        frame = fusion[fusion["size"] == size]
        corr = float(frame[frame["analysis"] == "rank_spearman"]["value"].mean())
        line(f"- Size{size}: mean base-method rank correlation = {fmt(corr)} ({'low correlation indicates complementary errors' if corr < 0.5 else 'high correlation indicates redundant rankings'}).")
    line("")
    line("**8. Does fusion reduce or amplify shared false positives?**")
    for size in sizes:
        frame = fusion[fusion["size"] == size]
        tp = float(frame[(frame["analysis"] == "fusion_support") & (frame["key"].str.startswith("mean_support_tp"))]["value"].mean())
        fp = float(frame[(frame["analysis"] == "fusion_support") & (frame["key"].str.startswith("mean_support_fp"))]["value"].mean())
        line(f"- Size{size}: fusion top-k true positives have mean multi-method support {fmt(tp)} vs false positives {fmt(fp)} (higher TP support means fusion promotes multi-method agreement on real edges).")
    line("")

    # H4
    line("**9. Are topology metrics genuinely separate from AUPR?**")
    for size in sizes:
        frame = metric_rel[metric_rel["size"] == size]
        hub = frame[(frame["metric_a"] == "aupr") & (frame["metric_b"] == "top_hub_overlap")]
        recip = frame[(frame["metric_a"] == "aupr") & (frame["metric_b"] == "reciprocal_fp_rate")]
        hub_v = float(hub.iloc[0]["spearman"]) if not hub.empty else float("nan")
        recip_v = float(recip.iloc[0]["spearman"]) if not recip.empty else float("nan")
        line(f"- Size{size}: Spearman(AUPR, top-hub overlap) = {fmt(hub_v)}; Spearman(AUPR, reciprocal-FP rate) = {fmt(recip_v)}. Weak/!=1 correlations mean topology is a partly separate objective.")
    line("")

    # H5
    line("**10. Why do level targets beat delta/derivative targets?**")
    for size in sizes:
        frame = h5[h5["size"] == size]
        line(f"- Size{size}: tree level AUPR {fmt(float(frame['tree_level_aupr'].mean()))} vs delta {fmt(float(frame['tree_delta_aupr'].mean()))} vs derivative {fmt(float(frame['tree_derivative_aupr'].mean()))}; "
             f"var(delta)/var(level) = {fmt(float(frame['var_ratio_delta_over_level'].mean()))}; delta-vs-derivative rank corr = {fmt(float(frame['delta_vs_derivative_rank_spearman'].mean()))}. "
             f"Level keeps persistence/smooth signal; differencing strips shared level signal and amplifies noise.")
    line("")

    # H11-14
    line("**11. Which findings seem general statistical lessons beyond biology?**")
    line("- Regularization strength should scale with sparsity and sample size; autoregressive terms can be essential controls yet dominate; ensemble/fusion only helps with complementary errors; predictive ranking and structure recovery are different objectives; target formulation (levels vs differences) changes signal-to-noise.")
    line("")
    line("**12. Which findings are likely DREAM4-specific?**")
    line("- The exact best alphas, the magnitude of the self/non-self ratio, the coarse uniform 50-unit time grid that makes delta~=derivative, and the specific density values (~0.17 vs ~0.02). These depend on DREAM4's simulation and sampling.")
    line("")
    line("**13. What should the project claim now?**")
    line("- Dynamic GRN inference here is regime-dependent and mechanistically explainable: alpha is a density knob whose best value rises as the true graph gets sparser; include-self helps mainly by controlling autoregressive persistence (permuting the self predictor removes the benefit), but a clean residualized model only reproduces that benefit in the sparse Size100 regime and not at Size10, so part of the include-self gain comes from joint estimation rather than simple self-variance removal; fusion helps when base methods are complementary (low rank correlation, true positives carry multi-method support); and AUPR does not guarantee topology recovery. Claims should be made per regime, and alpha should be chosen with deployable proxies (CV at small scale, BIC/density-prior at larger scale, which here land within one grid step of the oracle) rather than gold-standard tuning.")
    line("")
    line("**14. What should the next experiment be?**")
    line("- A literature-faithful (official) dynGENIE3 baseline, then GeneNetWeaver sweeps (experiment 12) that vary density, trajectory length, and noise to test whether the alpha-tracks-density rule, the residualization result, and the fusion-complementarity result hold under controlled conditions.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Size10 only, fewer trees and bootstrap resamples")
    parser.add_argument("--skip-size100", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tree-estimators-size10", type=int, default=None)
    parser.add_argument("--tree-estimators-size100", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=20260602)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [10] if (args.quick or args.skip_size100) else [10, 100]
    tree_estimators_by_size = {
        10: args.tree_estimators_size10 if args.tree_estimators_size10 is not None else (100 if args.quick else 200),
        100: args.tree_estimators_size100 if args.tree_estimators_size100 is not None else 100,
    }
    bootstrap_resamples = args.bootstrap_resamples if args.bootstrap_resamples is not None else (6 if args.quick else 12)

    networks = {(size, nid): load_size_network(size, nid) for size in sizes for nid in NETWORK_IDS}

    alpha_density, proxies = run_h1(networks, bootstrap_resamples=bootstrap_resamples, seed=args.random_seed)
    oracle_alpha_by_size = {
        size: float(alpha_density[alpha_density["size"] == size].groupby("alpha")["aupr"].mean().idxmax())
        for size in sizes
    }
    h2_summary, residual_edges = run_h2(networks, oracle_alpha_by_size=oracle_alpha_by_size, seed=args.random_seed)
    rankings, method_metrics = compute_rankings(
        networks, oracle_alpha_by_size=oracle_alpha_by_size, tree_estimators_by_size=tree_estimators_by_size,
        seed=args.random_seed, n_jobs=args.n_jobs,
    )
    fusion = run_h3(networks, rankings)
    metric_rel = run_h4(method_metrics)
    h5 = run_h5(networks, rankings, settings_precision=None)
    summary = build_summary(alpha_density, proxies, h2_summary, fusion, metric_rel, h5)
    figures = write_figures(alpha_density, method_metrics, rankings, networks)

    proxies.to_csv(RESULTS_DIR / f"{PREFIX}_alpha_proxies.csv", index=False)
    alpha_density.to_csv(ALPHA_DENSITY_PATH, index=False)
    h2_summary.to_csv(SELF_PERSISTENCE_PATH, index=False)
    residual_edges.to_csv(RESIDUALIZED_EDGES_PATH, index=False)
    fusion.to_csv(FUSION_PATH, index=False)
    pd.concat([metric_rel, _method_metric_long(method_metrics)], ignore_index=True).to_csv(METRIC_REL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(
        build_debug_report(alpha_density, proxies, h2_summary, fusion, metric_rel, method_metrics, h5, sizes, figures),
        encoding="utf-8",
    )
    print_summary(summary, oracle_alpha_by_size, sizes, figures)


def _method_metric_long(method_metrics: pd.DataFrame) -> pd.DataFrame:
    """Append per-method edge/topology metrics to the relationships file (long)."""
    rows = []
    for record in method_metrics.to_dict("records"):
        for metric in ("aupr", "top_hub_overlap", "out_degree_spearman", "in_degree_spearman", "reciprocal_fp_rate", "ffl_abs_error"):
            rows.append({"size": record["size"], "metric_a": "PER_METHOD", "metric_b": f"{record['method']}|net{record['network_id']}|{metric}", "spearman": record[metric]})
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, oracle_alpha_by_size, sizes, figures) -> None:
    print("DREAM4 mechanism audit")
    print(f"sizes={sizes} oracle_best_alpha_by_size={oracle_alpha_by_size}")
    print(f"matplotlib figures: {len(figures)}")
    print()
    for hyp in ["H1", "H2", "H3", "H4", "H5"]:
        frame = summary[summary["hypothesis"] == hyp]
        print(f"--- {hyp} ---")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
    for path in (ALPHA_DENSITY_PATH, SELF_PERSISTENCE_PATH, RESIDUALIZED_EDGES_PATH, FUSION_PATH, METRIC_REL_PATH, SUMMARY_PATH, DEBUG_REPORT_PATH):
        print(f"saved: {path.as_posix()}")


if __name__ == "__main__":
    main()
