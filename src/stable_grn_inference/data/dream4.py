from pathlib import Path

import pandas as pd

SIZE10_DATA_REGIMES = ("multifactorial", "knockouts", "knockdowns", "timeseries")
SIZE100_DATA_REGIMES = ("knockouts", "knockdowns", "timeseries", "wildtype")


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


def dream4_size10_expression_path(
    root: str | Path,
    network_id: int,
    data_regime: str,
) -> Path:
    """Return the local DREAM4 Size10 expression path for one data regime.

    Parameters
    ----------
    root:
        DREAM4 raw-data root, usually ``data/raw/dream4``.
    network_id:
        DREAM4 Size10 network number, from 1 through 5.
    data_regime:
        One of ``multifactorial``, ``knockouts``, ``knockdowns``, or
        ``timeseries``.
    """
    _validate_size10_network_id(network_id)
    if data_regime not in SIZE10_DATA_REGIMES:
        choices = ", ".join(SIZE10_DATA_REGIMES)
        raise ValueError(f"data_regime must be one of: {choices}")

    return (
        Path(root)
        / "DREAM4_InSilico_Size10"
        / f"insilico_size10_{network_id}"
        / f"insilico_size10_{network_id}_{data_regime}.tsv"
    )


def dream4_size10_gold_standard_path(root: str | Path, network_id: int) -> Path:
    """Return the local DREAM4 Size10 gold-standard edge path."""
    _validate_size10_network_id(network_id)
    return (
        Path(root)
        / "DREAM4_InSilicoNetworks_GoldStandard"
        / "DREAM4_Challenge2_GoldStandards"
        / "Size 10"
        / f"DREAM4_GoldStandard_InSilico_Size10_{network_id}.tsv"
    )


def dream4_size100_expression_path(
    root: str | Path,
    network_id: int,
    data_regime: str,
) -> Path:
    """Return the local DREAM4 Size100 expression path for one data regime.

    Parameters
    ----------
    root:
        DREAM4 raw-data root, usually ``data/raw/dream4``.
    network_id:
        DREAM4 Size100 network number, from 1 through 5.
    data_regime:
        One of ``knockouts``, ``knockdowns``, ``timeseries``, or ``wildtype``.
        Size100 multifactorial files live in a separate top-level directory and
        are not addressed by this helper.
    """
    _validate_size100_network_id(network_id)
    if data_regime not in SIZE100_DATA_REGIMES:
        choices = ", ".join(SIZE100_DATA_REGIMES)
        raise ValueError(f"data_regime must be one of: {choices}")

    return (
        Path(root)
        / "DREAM4_InSilico_Size100"
        / f"insilico_size100_{network_id}"
        / f"insilico_size100_{network_id}_{data_regime}.tsv"
    )


def dream4_size100_gold_standard_path(root: str | Path, network_id: int) -> Path:
    """Return the local DREAM4 Size100 gold-standard edge path.

    The Size100 gold-standard files are headerless three-column tables with
    9900 rows, one per directed non-self gene pair for the 100 genes.
    """
    _validate_size100_network_id(network_id)
    return (
        Path(root)
        / "DREAM4_InSilicoNetworks_GoldStandard"
        / "DREAM4_Challenge2_GoldStandards"
        / "Size 100"
        / f"DREAM4_GoldStandard_InSilico_Size100_{network_id}.tsv"
    )


def _validate_size10_network_id(network_id: int) -> None:
    """Validate a DREAM4 Size10 network id."""
    if network_id not in range(1, 6):
        raise ValueError("network_id must be between 1 and 5")


def _validate_size100_network_id(network_id: int) -> None:
    """Validate a DREAM4 Size100 network id."""
    if network_id not in range(1, 6):
        raise ValueError("network_id must be between 1 and 5")
