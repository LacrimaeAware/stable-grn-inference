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
