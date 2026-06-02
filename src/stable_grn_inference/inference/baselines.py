import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso


def rank_edges_by_correlation(expression: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate directed edges by absolute pairwise correlation.

    Parameters
    ----------
    expression:
        Data frame whose columns are genes and whose rows are samples or time
        points. The orientation should be confirmed when real data is added.

    Returns
    -------
    pandas.DataFrame
        Candidate directed edges with columns ``source``, ``target``, and
        ``score``, sorted from highest to lowest score.

    Notes
    -----
    This is intentionally naive. It gives the first pipeline a simple edge
    ranking to evaluate before sparse lagged models are implemented.
    """
    correlations = expression.corr().fillna(0.0)
    rows: list[dict[str, float | str]] = []

    for source, target in itertools.permutations(correlations.columns, 2):
        rows.append(
            {
                "source": str(source),
                "target": str(target),
                "score": float(abs(correlations.loc[source, target])),
            }
        )

    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_lasso(
    expression: pd.DataFrame,
    *,
    alpha: float = 0.01,
    max_iter: int = 10000,
) -> pd.DataFrame:
    """Rank directed edges by sparse target-wise LASSO coefficient magnitude.

    Parameters
    ----------
    expression:
        Data frame whose columns are genes and whose rows are samples.
    alpha:
        LASSO regularization strength. Larger values produce sparser rankings.
    max_iter:
        Maximum number of coordinate-descent iterations per target model.

    Returns
    -------
    pandas.DataFrame
        Candidate directed edges with columns ``source``, ``target``, and
        ``score``, sorted from highest to lowest score.

    Notes
    -----
    For each target gene, this fits a simple LASSO model using all other genes
    as predictors. Predictor genes become candidate sources, and the target
    gene is the predicted response. This is a baseline edge-ranking method, not
    a causal claim.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    expression = expression.apply(pd.to_numeric)
    genes = list(expression.columns)
    rows: list[dict[str, float | str]] = []

    for target in genes:
        sources = [gene for gene in genes if gene != target]
        x = expression[sources].to_numpy(dtype=float)
        y = expression[target].to_numpy(dtype=float)

        x_scaled = _standardize_columns(x)
        y_scaled = _standardize_vector(y)

        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
        model.fit(x_scaled, y_scaled)

        for source, coefficient in zip(sources, model.coef_):
            rows.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "score": float(abs(coefficient)),
                }
            )

    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_elastic_net(
    expression: pd.DataFrame,
    *,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 10000,
) -> pd.DataFrame:
    """Rank directed edges by target-wise Elastic Net coefficient magnitude.

    Parameters
    ----------
    expression:
        Data frame whose columns are genes and whose rows are samples.
    alpha:
        Regularization strength. Larger values produce smaller coefficients.
    l1_ratio:
        Mixing parameter between ridge-like and lasso-like regularization.
        ``1.0`` is equivalent to LASSO.
    max_iter:
        Maximum number of coordinate-descent iterations per target model.

    Returns
    -------
    pandas.DataFrame
        Candidate directed edges with columns ``source``, ``target``, and
        ``score``, sorted from highest to lowest score.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must be in (0, 1]")

    expression = expression.apply(pd.to_numeric)
    genes = list(expression.columns)
    rows: list[dict[str, float | str]] = []

    for target in genes:
        sources = [gene for gene in genes if gene != target]
        x = expression[sources].to_numpy(dtype=float)
        y = expression[target].to_numpy(dtype=float)

        x_scaled = _standardize_columns(x)
        y_scaled = _standardize_vector(y)

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=max_iter,
        )
        model.fit(x_scaled, y_scaled)

        for source, coefficient in zip(sources, model.coef_):
            rows.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "score": float(abs(coefficient)),
                }
            )

    return _sort_ranked_edges(pd.DataFrame(rows))


def rank_edges_by_random_forest(
    expression: pd.DataFrame,
    *,
    n_estimators: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Rank directed edges by target-wise random-forest feature importance.

    Parameters
    ----------
    expression:
        Data frame whose columns are genes and whose rows are samples.
    n_estimators:
        Number of trees per target-gene forest.
    random_state:
        Random seed used for reproducible baseline scores.

    Returns
    -------
    pandas.DataFrame
        Candidate directed edges with columns ``source``, ``target``, and
        ``score``, sorted from highest to lowest score.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")

    expression = expression.apply(pd.to_numeric)
    genes = list(expression.columns)
    rows: list[dict[str, float | str]] = []

    for target_index, target in enumerate(genes):
        sources = [gene for gene in genes if gene != target]
        x = expression[sources].to_numpy(dtype=float)
        y = expression[target].to_numpy(dtype=float)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state + target_index,
        )
        model.fit(x, y)

        for source, importance in zip(sources, model.feature_importances_):
            rows.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "score": float(importance),
                }
            )

    return _sort_ranked_edges(pd.DataFrame(rows))


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
