from pathlib import Path

import pandas as pd


def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Load an expression matrix from a known local file.

    Parameters
    ----------
    path:
        Path to a manually downloaded expression matrix.

    Returns
    -------
    pandas.DataFrame
        Expression values with the file's rows and columns preserved.

    Notes
    -----
    TODO: Confirm the DREAM4 file delimiter, header convention, and orientation
    before making this loader format-specific.
    """
    return pd.read_csv(path, sep=None, engine="python")


def load_gold_standard_edges(path: str | Path) -> pd.DataFrame:
    """Load a gold-standard edge table from a known local file.

    Parameters
    ----------
    path:
        Path to a manually downloaded gold-standard network file.

    Returns
    -------
    pandas.DataFrame
        Raw edge table. Expected columns should be normalized after the real
        format is inspected.

    Notes
    -----
    TODO: Confirm the official edge columns and labels before assuming a schema.
    """
    return pd.read_csv(path, sep=None, engine="python")
