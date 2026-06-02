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


def aggregate_per_network_metrics(
    network_metrics: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-network metric rows with mean, std, and network counts."""
    if network_metrics.empty:
        return pd.DataFrame()
    missing = [column for column in group_columns if column not in network_metrics.columns]
    if missing:
        raise ValueError(f"missing group columns: {missing}")

    if metric_columns is None:
        excluded = set(group_columns) | {"row_type", "network_id", "network"}
        metric_columns = [
            column
            for column in network_metrics.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(network_metrics[column])
        ]

    grouped = network_metrics.groupby(group_columns, dropna=False, as_index=False)
    mean_rows = grouped[metric_columns].mean()
    std_rows = grouped[metric_columns].std().rename(
        columns={column: f"std_{column}" for column in metric_columns}
    )
    # Name the count column directly so it never collides with a group column
    # that happens to be named "size" (the default name groupby.size() would use).
    counts = (
        network_metrics.groupby(group_columns, dropna=False).size().reset_index(name="n_networks")
    )
    return mean_rows.merge(std_rows, on=group_columns, how="left").merge(counts, on=group_columns, how="left")
