"""GENIE3-style target-wise tree ensemble edge ranking."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


TreeEnsembleKind = Literal["random_forest", "extra_trees"]


def rank_edges_by_genie3_random_forest(
    expression: pd.DataFrame,
    *,
    n_estimators: int = 1000,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank directed edges with a GENIE3-style random forest ensemble.

    Parameters
    ----------
    expression:
        Numeric expression matrix with samples in rows and genes in columns.
    n_estimators:
        Number of trees fit for each target gene.
    random_state:
        Base random seed. Each target gene receives a deterministic offset.
    max_features:
        Number of candidate predictors considered at each split. The default
        ``"sqrt"`` follows the usual GENIE3-style feature subspacing.
    n_jobs:
        Number of jobs passed to scikit-learn tree ensembles.

    Returns
    -------
    pandas.DataFrame
        One row for every directed non-self edge with ``source``, ``target``,
        and nonnegative ``score`` columns sorted from strongest to weakest.
    """
    return rank_edges_by_genie3(
        expression,
        ensemble="random_forest",
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features,
        n_jobs=n_jobs,
    )


def rank_edges_by_genie3_extra_trees(
    expression: pd.DataFrame,
    *,
    n_estimators: int = 1000,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank directed edges with a GENIE3-style Extra Trees ensemble.

    Parameters and return values match
    :func:`rank_edges_by_genie3_random_forest`.
    """
    return rank_edges_by_genie3(
        expression,
        ensemble="extra_trees",
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features,
        n_jobs=n_jobs,
    )


def rank_edges_by_genie3(
    expression: pd.DataFrame,
    *,
    ensemble: TreeEnsembleKind = "random_forest",
    n_estimators: int = 1000,
    random_state: int = 0,
    max_features: str | int | float | None = "sqrt",
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Rank edges by target-wise tree ensemble feature importance.

    For each target gene, all other genes are used as predictors. The fitted
    model's feature importances become directed source-to-target edge scores.
    This mirrors the core GENIE3 baseline idea without adding DREAM-specific
    post-processing.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if ensemble not in {"random_forest", "extra_trees"}:
        raise ValueError("ensemble must be 'random_forest' or 'extra_trees'")

    expression = expression.apply(pd.to_numeric).copy()
    expression.columns = [str(gene) for gene in expression.columns]
    genes = list(expression.columns)
    rows: list[dict[str, float | str]] = []

    for target_index, target in enumerate(genes):
        sources = [gene for gene in genes if gene != target]
        x = expression[sources].to_numpy(dtype=float)
        y = expression[target].to_numpy(dtype=float)

        model = _make_tree_ensemble(
            ensemble,
            n_estimators=n_estimators,
            random_state=random_state + target_index,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        model.fit(x, y)

        for source, importance in zip(sources, model.feature_importances_):
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "score": float(importance),
                }
            )

    return _sort_ranked_edges(pd.DataFrame(rows))


def _make_tree_ensemble(
    ensemble: TreeEnsembleKind,
    *,
    n_estimators: int,
    random_state: int,
    max_features: str | int | float | None,
    n_jobs: int | None,
) -> RandomForestRegressor | ExtraTreesRegressor:
    """Create the requested target-wise tree ensemble."""
    common_kwargs = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "max_features": max_features,
        "n_jobs": n_jobs,
    }
    if ensemble == "random_forest":
        return RandomForestRegressor(**common_kwargs)
    return ExtraTreesRegressor(**common_kwargs)


def _sort_ranked_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Sort edge scores deterministically from strongest to weakest."""
    return edges.sort_values(
        ["score", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
