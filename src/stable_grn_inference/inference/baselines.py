import itertools

import pandas as pd


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

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
