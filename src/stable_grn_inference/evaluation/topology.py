"""Topology-aware evaluation helpers for directed GRN edge rankings."""

from __future__ import annotations

from collections.abc import Sequence
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def directed_adjacency(
    edges: pd.DataFrame,
    *,
    genes: Sequence[str] | None = None,
    cutoff: int | None = None,
    rank_column: str = "rank",
    truth_column: str | None = None,
) -> pd.DataFrame:
    """Build a binary directed adjacency matrix from an edge table.

    Parameters
    ----------
    edges:
        Data frame with ``source`` and ``target`` columns. If ``cutoff`` is
        supplied, the table is sorted by ``rank_column`` and only the top
        edges are used. If ``truth_column`` is supplied, only rows with truth
        value 1 are used.
    genes:
        Optional ordered gene list. If omitted, genes are inferred from the
        edge table.
    cutoff:
        Number of top-ranked edges to include for predicted graphs.
    rank_column:
        Rank column used when applying ``cutoff``.
    truth_column:
        Optional binary column used to build a gold-standard graph.

    Returns
    -------
    pandas.DataFrame
        Square 0/1 adjacency matrix with sources as rows and targets as
        columns. Self-edges are always excluded.
    """
    if cutoff is not None and cutoff <= 0:
        raise ValueError("cutoff must be positive")

    table = edges[["source", "target", *optional_columns(edges, [rank_column, truth_column])]].copy()
    table["source"] = table["source"].astype(str)
    table["target"] = table["target"].astype(str)
    table = table[table["source"] != table["target"]]

    if genes is None:
        genes = sorted(set(table["source"]) | set(table["target"]))
    genes = [str(gene) for gene in genes]

    if truth_column is not None:
        if truth_column not in table.columns:
            raise ValueError(f"missing truth column: {truth_column}")
        table = table[table[truth_column].astype(int) == 1]
    elif cutoff is not None:
        if rank_column not in table.columns:
            raise ValueError(f"missing rank column: {rank_column}")
        table = table.sort_values([rank_column, "source", "target"]).head(cutoff)

    adjacency = pd.DataFrame(0, index=genes, columns=genes, dtype=int)
    for row in table.itertuples(index=False):
        if row.source in adjacency.index and row.target in adjacency.columns:
            adjacency.loc[row.source, row.target] = 1

    for gene in genes:
        adjacency.loc[gene, gene] = 0
    return adjacency


def optional_columns(edges: pd.DataFrame, columns: Sequence[str | None]) -> list[str]:
    """Return existing non-core columns requested by a caller."""
    return [
        column
        for column in columns
        if column is not None
        and column in edges.columns
        and column not in {"source", "target"}
    ]


def degree_by_node(adjacency: pd.DataFrame, *, direction: str) -> pd.Series:
    """Return in-degree or out-degree for each node in a directed graph."""
    if direction == "out":
        return adjacency.sum(axis=1).astype(int)
    if direction == "in":
        return adjacency.sum(axis=0).astype(int)
    raise ValueError("direction must be 'in' or 'out'")


def spearman_degree_correlation(
    true_adjacency: pd.DataFrame,
    predicted_adjacency: pd.DataFrame,
    *,
    direction: str,
) -> float:
    """Compute Spearman correlation between true and predicted degrees."""
    true_degree = degree_by_node(true_adjacency, direction=direction)
    predicted_degree = degree_by_node(predicted_adjacency, direction=direction)
    predicted_degree = predicted_degree.reindex(true_degree.index).fillna(0)

    true_values = true_degree.to_numpy(dtype=float)
    predicted_values = predicted_degree.to_numpy(dtype=float)
    if np.all(true_values == true_values[0]) or np.all(predicted_values == predicted_values[0]):
        return 1.0 if np.array_equal(true_values, predicted_values) else 0.0

    correlation = spearmanr(true_values, predicted_values).correlation
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def top_hub_overlap(
    true_degree: pd.Series,
    predicted_degree: pd.Series,
    *,
    top_n: int,
) -> float:
    """Return fractional overlap between true and predicted top-degree hubs."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if true_degree.empty:
        return 0.0

    denominator = min(top_n, len(true_degree))
    true_hubs = set(top_hubs(true_degree, denominator))
    predicted_hubs = set(top_hubs(predicted_degree.reindex(true_degree.index).fillna(0), denominator))
    return len(true_hubs & predicted_hubs) / denominator


def top_hubs(degree: pd.Series, top_n: int) -> list[str]:
    """Return deterministic top-degree hub names."""
    ranked = (
        degree.rename("degree")
        .reset_index()
        .rename(columns={"index": "gene"})
        .assign(gene=lambda frame: frame["gene"].astype(str))
        .sort_values(["degree", "gene"], ascending=[False, True])
    )
    return ranked.head(top_n)["gene"].tolist()


def reciprocal_pair_count(adjacency: pd.DataFrame) -> int:
    """Count unordered node pairs where both directed edges are present."""
    count = 0
    genes = list(adjacency.index)
    for source, target in itertools.combinations(genes, 2):
        if adjacency.loc[source, target] == 1 and adjacency.loc[target, source] == 1:
            count += 1
    return count


def reciprocal_false_positive_pair_count(
    predicted_adjacency: pd.DataFrame,
    true_adjacency: pd.DataFrame,
) -> int:
    """Count predicted reciprocal pairs containing at least one false edge."""
    count = 0
    genes = list(predicted_adjacency.index)
    for source, target in itertools.combinations(genes, 2):
        predicted_pair = (
            predicted_adjacency.loc[source, target] == 1
            and predicted_adjacency.loc[target, source] == 1
        )
        if not predicted_pair:
            continue
        true_pair = (
            true_adjacency.loc[source, target] == 1
            and true_adjacency.loc[target, source] == 1
        )
        if not true_pair:
            count += 1
    return count


def reciprocal_false_positive_edge_count(
    predicted_adjacency: pd.DataFrame,
    true_adjacency: pd.DataFrame,
) -> int:
    """Count false directed edges that belong to predicted reciprocal pairs."""
    count = 0
    genes = list(predicted_adjacency.index)
    for source, target in itertools.combinations(genes, 2):
        predicted_pair = (
            predicted_adjacency.loc[source, target] == 1
            and predicted_adjacency.loc[target, source] == 1
        )
        if not predicted_pair:
            continue
        count += int(true_adjacency.loc[source, target] == 0)
        count += int(true_adjacency.loc[target, source] == 0)
    return count


def feed_forward_loop_count(adjacency: pd.DataFrame) -> int:
    """Count directed feed-forward loops ``A -> B``, ``B -> C``, ``A -> C``.

    For a 0/1 adjacency matrix ``A`` with a zero diagonal, the number of
    distinct ordered triples ``(source, middle, target)`` forming a
    feed-forward loop equals ``sum(A * (A @ A))``: ``(A @ A)[s, t]`` counts the
    intermediate nodes ``b`` with ``A[s, b] = A[b, t] = 1`` (the zero diagonal
    drops the ``b = s`` and ``b = t`` terms), and multiplying by ``A[s, t]``
    keeps only pairs that also have the closing edge. This vectorized form is
    equivalent to enumerating triples but scales to 100-gene networks.
    """
    values = adjacency.to_numpy(dtype=np.int64, copy=True)
    np.fill_diagonal(values, 0)
    return int((values * (values @ values)).sum())


def hub_edge_precision_recall(
    true_adjacency: pd.DataFrame,
    predicted_adjacency: pd.DataFrame,
    *,
    direction: str,
    top_n: int = 1,
) -> tuple[float, float]:
    """Return precision/recall for edges incident to top true hubs."""
    if direction not in {"in", "out"}:
        raise ValueError("direction must be 'in' or 'out'")

    true_degree = degree_by_node(true_adjacency, direction=direction)
    hubs = top_hubs(true_degree, min(top_n, len(true_degree)))

    if direction == "out":
        predicted_edges = predicted_adjacency.loc[hubs, :]
        true_edges = true_adjacency.loc[hubs, :]
    else:
        predicted_edges = predicted_adjacency.loc[:, hubs]
        true_edges = true_adjacency.loc[:, hubs]

    predicted_mask = predicted_edges.to_numpy(dtype=bool)
    true_mask = true_edges.to_numpy(dtype=bool)
    true_positive_count = int((predicted_mask & true_mask).sum())
    predicted_count = int(predicted_mask.sum())
    true_count = int(true_mask.sum())
    precision = true_positive_count / predicted_count if predicted_count else 0.0
    recall = true_positive_count / true_count if true_count else 0.0
    return float(precision), float(recall)


def topology_metrics_for_cutoff(
    edge_table: pd.DataFrame,
    *,
    cutoff: int,
    rank_column: str,
    genes: Sequence[str] | None = None,
) -> dict[str, float | int]:
    """Compute topology-aware metrics for one ranked edge table and cutoff."""
    if genes is None:
        genes = sorted(set(edge_table["source"].astype(str)) | set(edge_table["target"].astype(str)))

    true_adjacency = directed_adjacency(edge_table, genes=genes, truth_column="is_true")
    predicted_adjacency = directed_adjacency(
        edge_table,
        genes=genes,
        cutoff=cutoff,
        rank_column=rank_column,
    )
    top_edges = (
        edge_table[edge_table["source"].astype(str) != edge_table["target"].astype(str)]
        .sort_values([rank_column, "source", "target"])
        .head(cutoff)
    )

    true_out_degree = degree_by_node(true_adjacency, direction="out")
    predicted_out_degree = degree_by_node(predicted_adjacency, direction="out")
    true_in_degree = degree_by_node(true_adjacency, direction="in")
    predicted_in_degree = degree_by_node(predicted_adjacency, direction="in")

    reciprocal_pairs = reciprocal_pair_count(predicted_adjacency)
    reciprocal_false_pairs = reciprocal_false_positive_pair_count(predicted_adjacency, true_adjacency)
    out_precision, out_recall = hub_edge_precision_recall(
        true_adjacency,
        predicted_adjacency,
        direction="out",
        top_n=1,
    )
    in_precision, in_recall = hub_edge_precision_recall(
        true_adjacency,
        predicted_adjacency,
        direction="in",
        top_n=1,
    )

    predicted_ffl = feed_forward_loop_count(predicted_adjacency)
    true_ffl = feed_forward_loop_count(true_adjacency)
    predicted_reciprocal_edges = reciprocal_pairs * 2
    true_reciprocal_edges = reciprocal_pair_count(true_adjacency) * 2

    return {
        "edge_precision_at_k": float(top_edges["is_true"].astype(int).mean()) if len(top_edges) else 0.0,
        "out_degree_spearman": spearman_degree_correlation(
            true_adjacency,
            predicted_adjacency,
            direction="out",
        ),
        "in_degree_spearman": spearman_degree_correlation(
            true_adjacency,
            predicted_adjacency,
            direction="in",
        ),
        "top1_out_hub_overlap": top_hub_overlap(true_out_degree, predicted_out_degree, top_n=1),
        "top3_out_hub_overlap": top_hub_overlap(true_out_degree, predicted_out_degree, top_n=3),
        "top5_out_hub_overlap": top_hub_overlap(true_out_degree, predicted_out_degree, top_n=5),
        "top1_in_hub_overlap": top_hub_overlap(true_in_degree, predicted_in_degree, top_n=1),
        "top3_in_hub_overlap": top_hub_overlap(true_in_degree, predicted_in_degree, top_n=3),
        "top5_in_hub_overlap": top_hub_overlap(true_in_degree, predicted_in_degree, top_n=5),
        "reciprocal_pair_count": reciprocal_pairs,
        "reciprocal_false_positive_pair_count": reciprocal_false_pairs,
        "reciprocal_false_positive_edge_count": reciprocal_false_positive_edge_count(
            predicted_adjacency,
            true_adjacency,
        ),
        "reciprocal_false_positive_pair_rate": (
            reciprocal_false_pairs / reciprocal_pairs if reciprocal_pairs else 0.0
        ),
        "predicted_reciprocal_edge_count": predicted_reciprocal_edges,
        "true_reciprocal_edge_count": true_reciprocal_edges,
        "reciprocal_edge_count_abs_error": abs(predicted_reciprocal_edges - true_reciprocal_edges),
        "predicted_feed_forward_loop_count": predicted_ffl,
        "true_feed_forward_loop_count": true_ffl,
        "feed_forward_loop_abs_error": abs(predicted_ffl - true_ffl),
        "true_out_hub_edge_precision": out_precision,
        "true_out_hub_edge_recall": out_recall,
        "true_in_hub_edge_precision": in_precision,
        "true_in_hub_edge_recall": in_recall,
    }
