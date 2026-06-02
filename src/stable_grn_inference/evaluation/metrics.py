import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true: list[int] | pd.Series, y_score: list[float] | pd.Series) -> float:
    """Compute AUROC for a binary edge-recovery ranking."""
    return float(roc_auc_score(y_true, y_score))


def aupr(y_true: list[int] | pd.Series, y_score: list[float] | pd.Series) -> float:
    """Compute area under the precision-recall curve for edge recovery."""
    return float(average_precision_score(y_true, y_score))


def precision_at_k(scored_edges: pd.DataFrame, truth_column: str, k: int) -> float:
    """Compute precision among the top-k scored edges.

    Parameters
    ----------
    scored_edges:
        Edge table sorted from most likely to least likely.
    truth_column:
        Name of a binary column indicating whether each edge is present in the
        gold-standard network.
    k:
        Number of highest-ranked edges to evaluate.

    Returns
    -------
    float
        Fraction of the top-k edges marked as true.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = scored_edges.head(k)
    if top_k.empty:
        return 0.0
    return float(top_k[truth_column].mean())
