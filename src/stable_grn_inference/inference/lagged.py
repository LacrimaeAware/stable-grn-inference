"""Lagged time-series edge-ranking baselines."""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


LaggedTreeEnsembleKind = Literal["random_forest", "extra_trees"]


def rank_edges_by_lagged_correlation(x_t: pd.DataFrame, y_t1: pd.DataFrame) -> pd.DataFrame:
    """Rank source(t) -> target(t+1) edges by absolute lagged correlation."""
    x, y = _prepare_lagged_matrices(x_t, y_t1)
    rows: list[dict[str, float | str]] = []
    for source, target in itertools.permutations(x.columns, 2):
        score = _absolute_correlation(x[source], y[target])
        rows.append({"source": str(source), "target": str(target), "score": score})
    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_lagged_lasso(
    x_t: pd.DataFrame,
    y_t1: pd.DataFrame,
    *,
    alpha: float = 0.1,
    max_iter: int = 50000,
) -> pd.DataFrame:
    """Rank lagged directed edges by target-wise LASSO coefficients.

    For each target gene at ``t+1``, this fits a LASSO model using non-self
    genes at ``t`` as predictors. Self-lag edges are excluded to match the
    DREAM4 directed non-self edge candidate set used elsewhere in the repo.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    x, y = _prepare_lagged_matrices(x_t, y_t1)
    genes = list(x.columns)
    rows: list[dict[str, float | str]] = []

    for target in genes:
        sources = [gene for gene in genes if gene != target]
        x_values = _standardize_columns(x[sources].to_numpy(dtype=float))
        y_values = _standardize_vector(y[target].to_numpy(dtype=float))
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        model.fit(x_values, y_values)
        for source, coefficient in zip(sources, model.coef_):
            rows.append({"source": str(source), "target": str(target), "score": float(abs(coefficient))})

    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_lagged_random_forest(
    x_t: pd.DataFrame,
    y_t1: pd.DataFrame,
    *,
    n_estimators: int = 500,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank lagged edges with target-wise random-forest feature importance."""
    return rank_edges_by_lagged_tree_ensemble(
        x_t,
        y_t1,
        ensemble="random_forest",
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features,
        n_jobs=n_jobs,
    )


def rank_edges_by_lagged_extra_trees(
    x_t: pd.DataFrame,
    y_t1: pd.DataFrame,
    *,
    n_estimators: int = 500,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank lagged edges with target-wise Extra Trees feature importance."""
    return rank_edges_by_lagged_tree_ensemble(
        x_t,
        y_t1,
        ensemble="extra_trees",
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features,
        n_jobs=n_jobs,
    )


def rank_edges_by_lagged_tree_ensemble(
    x_t: pd.DataFrame,
    y_t1: pd.DataFrame,
    *,
    ensemble: LaggedTreeEnsembleKind = "random_forest",
    n_estimators: int = 500,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank lagged edges by target-wise tree-ensemble feature importance."""
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if ensemble not in {"random_forest", "extra_trees"}:
        raise ValueError("ensemble must be 'random_forest' or 'extra_trees'")

    x, y = _prepare_lagged_matrices(x_t, y_t1)
    genes = list(x.columns)
    rows: list[dict[str, float | str]] = []

    for target_index, target in enumerate(genes):
        sources = [gene for gene in genes if gene != target]
        model = _make_tree_ensemble(
            ensemble,
            n_estimators=n_estimators,
            random_state=random_state + target_index,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        model.fit(x[sources].to_numpy(dtype=float), y[target].to_numpy(dtype=float))
        for source, importance in zip(sources, model.feature_importances_):
            rows.append({"source": str(source), "target": str(target), "score": float(importance)})

    return _sort_ranked_edges(pd.DataFrame(rows))


def _prepare_lagged_matrices(x_t: pd.DataFrame, y_t1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize lagged predictor and target matrices."""
    if len(x_t) != len(y_t1):
        raise ValueError("x_t and y_t1 must have the same number of rows")
    if list(x_t.columns) != list(y_t1.columns):
        raise ValueError("x_t and y_t1 must have matching gene columns")
    x = x_t.apply(pd.to_numeric).copy()
    y = y_t1.apply(pd.to_numeric).copy()
    x.columns = [str(column) for column in x.columns]
    y.columns = [str(column) for column in y.columns]
    return x, y


def _absolute_correlation(source: pd.Series, target: pd.Series) -> float:
    """Return absolute Pearson correlation while tolerating constant vectors."""
    if source.std() == 0.0 or target.std() == 0.0:
        return 0.0
    value = source.corr(target)
    if pd.isna(value):
        return 0.0
    return float(abs(value))


def _make_tree_ensemble(
    ensemble: LaggedTreeEnsembleKind,
    *,
    n_estimators: int,
    random_state: int,
    max_features: str | int | float | None,
    n_jobs: int | None,
) -> RandomForestRegressor | ExtraTreesRegressor:
    """Create the requested lagged tree ensemble."""
    common_kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "max_features": max_features,
        "n_jobs": n_jobs,
    }
    if ensemble == "random_forest":
        return RandomForestRegressor(**common_kwargs)
    return ExtraTreesRegressor(**common_kwargs)


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


def _sort_ranked_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Sort edge scores deterministically from strongest to weakest."""
    return edges.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
