from pathlib import Path

import pandas as pd


def _normalize_gene_id(value: object) -> str:
    """Return a DREAM-style gene identifier such as ``G1``."""
    text = str(value).strip().strip('"')
    if text.upper().startswith("G"):
        return f"G{text[1:]}"
    if text.isdigit():
        return f"G{text}"
    return text


def load_expression_matrix(path: str | Path, *, drop_time: bool = True) -> pd.DataFrame:
    """Load a DREAM4-style tab-delimited expression matrix.

    Parameters
    ----------
    path:
        Path to a manually downloaded expression matrix.
    drop_time:
        If true, drop a ``Time`` column from time-series files so returned
        columns are gene IDs only.

    Returns
    -------
    pandas.DataFrame
        Numeric expression values. Columns are preserved as clear gene IDs
        such as ``G1`` through ``G10``.

    Notes
    -----
    Inspected DREAM4 files in ``data/raw/dream4`` are tab-delimited with quoted
    headers. Time-series files include a ``Time`` column and blank-separated
    trajectories; pandas skips the blank separator rows.
    """
    expression = pd.read_csv(path, sep="\t")
    expression.columns = [str(column).strip().strip('"') for column in expression.columns]

    if drop_time and "Time" in expression.columns:
        expression = expression.drop(columns=["Time"])

    expression.columns = [_normalize_gene_id(column) for column in expression.columns]
    return expression.apply(pd.to_numeric)


def load_gold_standard_edges(path: str | Path) -> pd.DataFrame:
    """Load and normalize a DREAM-style gold-standard edge table.

    Parameters
    ----------
    path:
        Path to a manually downloaded gold-standard network file.

    Returns
    -------
    pandas.DataFrame
        Edge table with ``source``, ``target``, and ``is_true`` columns.

    Notes
    -----
    DREAM4 gold-standard files are headerless, tab-delimited three-column
    tables: regulator, target, and binary edge label.
    """
    edges = pd.read_csv(path, sep="\t", header=None)
    if edges.shape[1] < 2:
        raise ValueError("Gold-standard edge table must have at least two columns")

    edges = edges.iloc[:, :3].copy() if edges.shape[1] >= 3 else edges.iloc[:, :2].copy()
    edges.columns = ["source", "target", "is_true"] if edges.shape[1] == 3 else ["source", "target"]

    if "is_true" not in edges.columns:
        edges["is_true"] = 1

    normalized = edges[["source", "target", "is_true"]].copy()
    normalized["source"] = normalized["source"].map(_normalize_gene_id)
    normalized["target"] = normalized["target"].map(_normalize_gene_id)
    normalized["is_true"] = normalized["is_true"].astype(int)
    return normalized
