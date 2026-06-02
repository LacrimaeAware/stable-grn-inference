"""Dynamic target-wise edge rankers for lagged time-series audits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import itertools
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor


SelfPredictorMode = Literal["exclude_self_predictor", "include_self_predictor_no_self_edge"]
TreeKind = Literal["random_forest", "extra_trees"]
LinearModelKind = Literal["lasso", "elastic_net"]
Ranker = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def rank_edges_by_dynamic_correlation(x_t: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Rank non-self source(t) -> target edges by absolute correlation."""
    x, y = _prepare_matrices(x_t, target)
    rows = []
    for source, target_gene in itertools.permutations(x.columns, 2):
        rows.append(
            {
                "source": source,
                "target": target_gene,
                "score": _absolute_correlation(x[source], y[target_gene]),
            }
        )
    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_dynamic_lasso(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_iter: int = 50000,
) -> pd.DataFrame:
    """Rank dynamic edges by target-wise LASSO coefficient magnitude."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return _rank_edges_by_linear_model(
        x_t,
        target,
        model_factory=lambda: Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter),
        self_predictor_mode=self_predictor_mode,
    )


def rank_edges_by_dynamic_elastic_net(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float,
    l1_ratio: float,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_iter: int = 50000,
) -> pd.DataFrame:
    """Rank dynamic edges by target-wise Elastic Net coefficient magnitude."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must be in (0, 1]")
    return _rank_edges_by_linear_model(
        x_t,
        target,
        model_factory=lambda: ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=max_iter,
        ),
        self_predictor_mode=self_predictor_mode,
    )


def _rank_edges_by_linear_model(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    model_factory: Callable[[], Lasso | ElasticNet],
    self_predictor_mode: SelfPredictorMode,
) -> pd.DataFrame:
    """Rank edges using target-wise linear model coefficients."""
    x, y = _prepare_matrices(x_t, target)
    rows = []
    for target_gene in x.columns:
        predictors = _predictor_columns(x.columns, target_gene, self_predictor_mode)
        x_values = _standardize_columns(x[predictors].to_numpy(dtype=float))
        y_values = _standardize_vector(y[target_gene].to_numpy(dtype=float))
        model = model_factory()
        model.fit(x_values, y_values)
        for source, coefficient in zip(predictors, model.coef_):
            if source != target_gene:
                rows.append({"source": source, "target": target_gene, "score": float(abs(coefficient))})
    return _sort_ranked_edges(pd.DataFrame(rows))


def fit_dynamic_linear_coefficients(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    model_kind: LinearModelKind,
    alpha: float,
    l1_ratio: float | None = None,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_iter: int = 50000,
    coefficient_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit target-wise sparse linear models and return signed coefficients.

    The edge table contains one row for every directed non-self edge and uses
    absolute coefficient magnitude as ``score``. When self predictors are
    included during fitting, self coefficients are returned in a separate
    target-level table; self-edges are never emitted as candidate edges.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance must be nonnegative")
    if model_kind == "elastic_net" and l1_ratio is None:
        raise ValueError("l1_ratio is required for elastic_net")

    x, y = _prepare_matrices(x_t, target)
    edge_rows: list[dict[str, float | str | bool]] = []
    self_rows: list[dict[str, float | str | bool]] = []

    for target_gene in x.columns:
        predictors = _predictor_columns(x.columns, target_gene, self_predictor_mode)
        x_values = _standardize_columns(x[predictors].to_numpy(dtype=float))
        y_values = _standardize_vector(y[target_gene].to_numpy(dtype=float))
        model = _make_linear_model(
            model_kind,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
        )
        model.fit(x_values, y_values)
        for source, coefficient in zip(predictors, model.coef_):
            coefficient = float(coefficient)
            selected = abs(coefficient) > coefficient_tolerance
            row = {
                "source": str(source),
                "target": str(target_gene),
                "coefficient": coefficient,
                "score": abs(coefficient),
                "selected": selected,
            }
            if source == target_gene:
                self_rows.append(
                    {
                        "target": str(target_gene),
                        "self_coefficient": coefficient,
                        "self_abs_coefficient": abs(coefficient),
                        "self_selected": selected,
                    }
                )
            else:
                edge_rows.append(row)

    edge_columns = ["source", "target", "coefficient", "score", "selected"]
    self_columns = ["target", "self_coefficient", "self_abs_coefficient", "self_selected"]
    edges = pd.DataFrame(edge_rows, columns=edge_columns)
    self_coefficients = pd.DataFrame(self_rows, columns=self_columns)
    return _sort_ranked_edges(edges), self_coefficients


def summarize_resampled_dynamic_linear_coefficients(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    resample_indices: list[np.ndarray],
    *,
    model_kind: LinearModelKind,
    alpha: float,
    l1_ratio: float | None = None,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_iter: int = 50000,
    coefficient_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sparse linear coefficients over resampled lagged rows.

    Returns edge-level nonzero selection frequency plus mean signed and mean
    absolute coefficients. If self predictors are included, a target-level
    self-coefficient summary is returned as the second table.
    """
    if not resample_indices:
        raise ValueError("resample_indices must not be empty")

    x, y = _prepare_matrices(x_t, target)
    edges = directed_nonself_edges(list(x.columns))
    n_edges = len(edges)
    coefficient_sum = np.zeros(n_edges, dtype=float)
    abs_coefficient_sum = np.zeros(n_edges, dtype=float)
    selected_sum = np.zeros(n_edges, dtype=float)

    target_names = pd.DataFrame({"target": [str(gene) for gene in x.columns]})
    self_coefficient_sum = np.zeros(len(target_names), dtype=float)
    self_abs_sum = np.zeros(len(target_names), dtype=float)
    self_selected_sum = np.zeros(len(target_names), dtype=float)
    saw_self_coefficients = False

    for indices in resample_indices:
        edge_coefficients, self_coefficients = fit_dynamic_linear_coefficients(
            x.iloc[indices].reset_index(drop=True),
            y.iloc[indices].reset_index(drop=True),
            model_kind=model_kind,
            alpha=alpha,
            l1_ratio=l1_ratio,
            self_predictor_mode=self_predictor_mode,
            max_iter=max_iter,
            coefficient_tolerance=coefficient_tolerance,
        )
        merged_edges = edges.merge(
            edge_coefficients[["source", "target", "coefficient", "score", "selected"]],
            on=["source", "target"],
            how="left",
        )
        coefficient_sum += merged_edges["coefficient"].fillna(0.0).to_numpy(dtype=float)
        abs_coefficient_sum += merged_edges["score"].fillna(0.0).to_numpy(dtype=float)
        selected_sum += merged_edges["selected"].fillna(False).to_numpy(dtype=bool)

        if not self_coefficients.empty:
            saw_self_coefficients = True
            merged_self = target_names.merge(self_coefficients, on="target", how="left")
            self_coefficient_sum += merged_self["self_coefficient"].fillna(0.0).to_numpy(dtype=float)
            self_abs_sum += merged_self["self_abs_coefficient"].fillna(0.0).to_numpy(dtype=float)
            self_selected_sum += merged_self["self_selected"].fillna(False).to_numpy(dtype=bool)

    n_resamples = len(resample_indices)
    edge_summary = edges.copy()
    edge_summary["selection_frequency"] = selected_sum / n_resamples
    edge_summary["mean_coefficient"] = coefficient_sum / n_resamples
    edge_summary["mean_abs_coefficient"] = abs_coefficient_sum / n_resamples

    if not saw_self_coefficients:
        return edge_summary, pd.DataFrame(
            columns=[
                "target",
                "self_selection_frequency",
                "mean_self_coefficient",
                "mean_abs_self_coefficient",
            ]
        )

    self_summary = target_names.copy()
    self_summary["self_selection_frequency"] = self_selected_sum / n_resamples
    self_summary["mean_self_coefficient"] = self_coefficient_sum / n_resamples
    self_summary["mean_abs_self_coefficient"] = self_abs_sum / n_resamples
    return edge_summary, self_summary


def build_dynamic_sparse_linear_grid(
    *,
    lasso_alphas: Sequence[float],
    elastic_net_alphas: Sequence[float],
    elastic_net_l1_ratios: Sequence[float],
    target_types: Sequence[str] = ("level", "delta"),
) -> pd.DataFrame:
    """Return the focused sparse-linear validation grid."""
    rows: list[dict[str, float | str | None]] = []
    for target_type in target_types:
        for self_mode in ["include_self_predictor_no_self_edge", "exclude_self_predictor"]:
            for alpha in lasso_alphas:
                rows.append(
                    {
                        "model_kind": "lasso",
                        "target_type": target_type,
                        "self_predictor_mode": self_mode,
                        "alpha": float(alpha),
                        "l1_ratio": None,
                        "method": (
                            f"dynamic_lasso_{target_type}_{_short_self_mode(self_mode)}"
                            f"_a{_format_alpha(alpha)}"
                        ),
                    }
                )
        for alpha in elastic_net_alphas:
            for l1_ratio in elastic_net_l1_ratios:
                rows.append(
                    {
                        "model_kind": "elastic_net",
                        "target_type": target_type,
                        "self_predictor_mode": "include_self_predictor_no_self_edge",
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                        "method": (
                            f"dynamic_elastic_net_{target_type}_include_self"
                            f"_a{_format_alpha(alpha)}_l1_{_format_alpha(l1_ratio)}"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def rank_edges_by_dynamic_tree_ensemble(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    ensemble: TreeKind,
    n_estimators: int = 200,
    random_state: int = 0,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank dynamic edges by target-wise tree feature importance."""
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if ensemble not in {"random_forest", "extra_trees"}:
        raise ValueError("ensemble must be 'random_forest' or 'extra_trees'")

    x, y = _prepare_matrices(x_t, target)
    rows = []
    for target_index, target_gene in enumerate(x.columns):
        predictors = _predictor_columns(x.columns, target_gene, self_predictor_mode)
        model = _make_tree_ensemble(
            ensemble,
            n_estimators=n_estimators,
            random_state=random_state + target_index,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        model.fit(x[predictors].to_numpy(dtype=float), y[target_gene].to_numpy(dtype=float))
        for source, importance in zip(predictors, model.feature_importances_):
            if source != target_gene:
                rows.append({"source": source, "target": target_gene, "score": float(importance)})
    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_dynamic_mlp_permutation(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    *,
    hidden_layer_sizes: tuple[int, ...] = (16,),
    alpha: float = 0.01,
    random_state: int = 0,
    self_predictor_mode: SelfPredictorMode = "exclude_self_predictor",
    max_iter: int = 500,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Rank dynamic edges with target-wise MLP permutation importance.

    This is a small neural-network sanity baseline. It fits one MLP per target
    gene and scores predictors by the increase in mean squared error after
    permuting that predictor.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")

    x, y = _prepare_matrices(x_t, target)
    rows = []
    rng = np.random.default_rng(random_state)
    for target_index, target_gene in enumerate(x.columns):
        predictors = _predictor_columns(x.columns, target_gene, self_predictor_mode)
        x_values = _standardize_columns(x[predictors].to_numpy(dtype=float))
        y_values = _standardize_vector(y[target_gene].to_numpy(dtype=float))
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=len(x_values) >= 50,
            random_state=random_state + target_index,
        )
        model.fit(x_values, y_values)
        importances = _permutation_importance(model, x_values, y_values, n_repeats=n_repeats, rng=rng)
        for source, importance in zip(predictors, importances):
            if source != target_gene:
                rows.append({"source": source, "target": target_gene, "score": float(max(0.0, importance))})
    return _sort_ranked_edges(pd.DataFrame(rows))


def summarize_resampled_dynamic_scores(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
    ranker: Ranker,
    resample_indices: list[np.ndarray],
    *,
    top_k: int = 20,
    selection_threshold: float = 0.0,
) -> pd.DataFrame:
    """Summarize dynamic edge scores over resampled lagged rows."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not resample_indices:
        raise ValueError("resample_indices must not be empty")

    x, y = _prepare_matrices(x_t, target)
    edges = directed_nonself_edges(list(x.columns))
    n_edges = len(edges)
    score_sum = np.zeros(n_edges, dtype=float)
    reciprocal_rank_sum = np.zeros(n_edges, dtype=float)
    top_k_count = np.zeros(n_edges, dtype=float)
    selected_count = np.zeros(n_edges, dtype=float)

    for indices in resample_indices:
        ranked = ranker(x.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True))
        ranked = ranked[["source", "target", "score"]].copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        merged = edges.merge(ranked, on=["source", "target"], how="left")
        merged["score"] = merged["score"].fillna(0.0)
        merged["rank"] = merged["rank"].fillna(n_edges + 1)
        scores = merged["score"].to_numpy(dtype=float)
        ranks = merged["rank"].to_numpy(dtype=float)
        score_sum += scores
        reciprocal_rank_sum += 1.0 / ranks
        top_k_count += ranks <= top_k
        selected_count += scores > selection_threshold

    n_resamples = len(resample_indices)
    result = edges.copy()
    result["mean_score"] = score_sum / n_resamples
    result["mean_reciprocal_rank"] = reciprocal_rank_sum / n_resamples
    result["top_k_frequency"] = top_k_count / n_resamples
    result["selection_frequency"] = selected_count / n_resamples
    return result


def directed_nonself_edges(genes: Sequence[str]) -> pd.DataFrame:
    """Return all directed non-self edges for a gene list."""
    return pd.DataFrame(
        [{"source": str(source), "target": str(target)} for source, target in itertools.permutations(genes, 2)]
    )


def rank_fusion(
    edge_tables: Sequence[pd.DataFrame],
    *,
    method: Literal["mean_normalized_score", "mean_reciprocal_rank", "borda"] = "mean_reciprocal_rank",
) -> pd.DataFrame:
    """Fuse multiple edge rankings using equal-weight score/rank aggregation."""
    if not edge_tables:
        raise ValueError("edge_tables must not be empty")
    base_edges = edge_tables[0][["source", "target"]].copy()
    fused = base_edges.copy()
    values = []
    n_edges = len(base_edges)
    for table in edge_tables:
        ranked = table[["source", "target", "score"]].copy()
        ranked = _sort_ranked_edges(ranked)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        merged = base_edges.merge(ranked, on=["source", "target"], how="left")
        if method == "mean_normalized_score":
            scores = merged["score"].fillna(0.0).to_numpy(dtype=float)
            values.append(_minmax(scores))
        elif method == "mean_reciprocal_rank":
            ranks = merged["rank"].fillna(n_edges + 1).to_numpy(dtype=float)
            values.append(1.0 / ranks)
        elif method == "borda":
            ranks = merged["rank"].fillna(n_edges + 1).to_numpy(dtype=float)
            values.append((n_edges - ranks + 1.0) / n_edges)
        else:
            raise ValueError("unknown rank fusion method")
    fused["score"] = np.vstack(values).mean(axis=0)
    return _sort_ranked_edges(fused)


def rank_fusion_with_reciprocal_penalty(
    edge_tables: Sequence[pd.DataFrame],
    *,
    penalty: float = 0.5,
    top_fraction: float = 0.05,
    base_method: Literal["mean_normalized_score", "mean_reciprocal_rank", "borda"] = "mean_reciprocal_rank",
) -> pd.DataFrame:
    """Fuse rankings, then down-weight the weaker side of high-confidence reciprocal pairs.

    The audits repeatedly saw reciprocal-direction false positives, where both
    ``G_i -> G_j`` and ``G_j -> G_i`` rank highly even though real regulation is
    usually directional. This variant first fuses the inputs with ``base_method``,
    then, for every unordered gene pair whose *both* directed edges fall in the
    top ``top_fraction`` of the fused ranking, multiplies the weaker-scoring
    direction by ``penalty`` (0 < penalty <= 1). It is a simple fixed-weight
    heuristic, not a tuned model.
    """
    if not 0.0 < penalty <= 1.0:
        raise ValueError("penalty must be in (0, 1]")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    fused = rank_fusion(edge_tables, method=base_method)
    n_edges = len(fused)
    if n_edges == 0:
        return fused

    top_count = max(1, int(round(top_fraction * n_edges)))
    top_pairs = {
        (str(row.source), str(row.target))
        for row in fused.head(top_count).itertuples(index=False)
    }
    scores = {
        (str(row.source), str(row.target)): float(row.score)
        for row in fused.itertuples(index=False)
    }

    penalized: set[tuple[str, str]] = set()
    for source, target in top_pairs:
        if source >= target:  # process each unordered pair once
            continue
        reverse = (target, source)
        if reverse not in top_pairs:
            continue
        forward_score = scores[(source, target)]
        reverse_score = scores[reverse]
        weaker = (source, target) if forward_score <= reverse_score else reverse
        penalized.add(weaker)

    adjusted = fused.copy()
    adjusted["score"] = adjusted["score"].astype(float)
    if penalized:
        keys = list(zip(adjusted["source"].astype(str), adjusted["target"].astype(str)))
        factors = np.array([penalty if key in penalized else 1.0 for key in keys], dtype=float)
        adjusted["score"] = adjusted["score"].to_numpy(dtype=float) * factors
    return _sort_ranked_edges(adjusted)


def _prepare_matrices(x_t: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize predictor and target matrices."""
    if len(x_t) != len(target):
        raise ValueError("x_t and target must have the same number of rows")
    if list(x_t.columns) != list(target.columns):
        raise ValueError("x_t and target must have matching gene columns")
    x = x_t.apply(pd.to_numeric).copy()
    y = target.apply(pd.to_numeric).copy()
    x.columns = [str(column) for column in x.columns]
    y.columns = [str(column) for column in y.columns]
    return x, y


def _predictor_columns(
    genes: Sequence[str],
    target_gene: str,
    self_predictor_mode: SelfPredictorMode,
) -> list[str]:
    """Return predictor columns for one target gene."""
    if self_predictor_mode == "exclude_self_predictor":
        return [gene for gene in genes if gene != target_gene]
    if self_predictor_mode == "include_self_predictor_no_self_edge":
        return list(genes)
    raise ValueError("unknown self_predictor_mode")


def _make_tree_ensemble(
    ensemble: TreeKind,
    *,
    n_estimators: int,
    random_state: int,
    max_features: str | int | float | None,
    n_jobs: int | None,
) -> RandomForestRegressor | ExtraTreesRegressor:
    """Create a tree ensemble."""
    kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "max_features": max_features,
        "n_jobs": n_jobs,
    }
    if ensemble == "random_forest":
        return RandomForestRegressor(**kwargs)
    return ExtraTreesRegressor(**kwargs)


def _make_linear_model(
    model_kind: LinearModelKind,
    *,
    alpha: float,
    l1_ratio: float | None,
    max_iter: int,
) -> Lasso | ElasticNet:
    """Create the requested sparse linear model."""
    if model_kind == "lasso":
        return Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    if model_kind == "elastic_net":
        if l1_ratio is None:
            raise ValueError("l1_ratio is required for elastic_net")
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter)
    raise ValueError("model_kind must be 'lasso' or 'elastic_net'")


def _permutation_importance(
    model: MLPRegressor,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    n_repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute nonnegative permutation importances from MSE increase."""
    baseline = _mean_squared_error(y_values, model.predict(x_values))
    importances = np.zeros(x_values.shape[1], dtype=float)
    for column_index in range(x_values.shape[1]):
        losses = []
        for _ in range(n_repeats):
            permuted = x_values.copy()
            permuted[:, column_index] = rng.permutation(permuted[:, column_index])
            losses.append(_mean_squared_error(y_values, model.predict(permuted)))
        importances[column_index] = max(0.0, float(np.mean(losses) - baseline))
    return importances


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def _absolute_correlation(source: pd.Series, target: pd.Series) -> float:
    """Return absolute Pearson correlation while tolerating constant vectors."""
    if source.std() == 0.0 or target.std() == 0.0:
        return 0.0
    value = source.corr(target)
    if pd.isna(value):
        return 0.0
    return float(abs(value))


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    """Standardize matrix columns while tolerating constant columns."""
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (values - values.mean(axis=0)) / scale


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    """Standardize a vector while tolerating constant targets."""
    scale = values.std()
    if scale == 0.0:
        scale = 1.0
    return (values - values.mean()) / scale


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize values while tolerating constants."""
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _sort_ranked_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Sort edge scores deterministically from strongest to weakest."""
    return edges.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _short_self_mode(self_mode: str) -> str:
    """Return compact self-predictor mode text for method names."""
    return "exclude_self" if self_mode == "exclude_self_predictor" else "include_self"


def _format_alpha(alpha: float) -> str:
    """Format a regularization value for method names."""
    return str(alpha).replace(".", "_")
