import pandas as pd


def edge_selection_frequencies(edge_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Summarize how often each edge appears across repeated inference runs.

    Parameters
    ----------
    edge_tables:
        List of edge tables. Each table must contain ``source`` and ``target``
        columns for selected edges from one bootstrap or subsampling run.

    Returns
    -------
    pandas.DataFrame
        Edge table with ``source``, ``target``, and ``frequency`` columns.

    Notes
    -----
    TODO: Add resampling and estimator wrappers after the first baseline works.
    """
    if not edge_tables:
        return pd.DataFrame(columns=["source", "target", "frequency"])

    counts: dict[tuple[str, str], int] = {}
    for table in edge_tables:
        for row in table[["source", "target"]].drop_duplicates().itertuples(index=False):
            edge = (str(row.source), str(row.target))
            counts[edge] = counts.get(edge, 0) + 1

    total_runs = len(edge_tables)
    rows = [
        {"source": source, "target": target, "frequency": count / total_runs}
        for (source, target), count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)
