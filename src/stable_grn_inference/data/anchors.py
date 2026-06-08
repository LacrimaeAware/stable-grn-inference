"""External ground-truth / proxy networks for grading inferred structure.

These are the anchors the project flagged it needed (next_direction.md section 8): downloadable,
literature-validated reference networks that let reproducibility be checked against correctness.
STRING functional associations are an undirected proxy (validated in Replogle 2022 and the
co-essentiality literature). Loaders never enter the inference path; truth is used only for
evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_string_network(path: str | Path) -> pd.DataFrame:
    """Load a STRING ``network`` TSV (from the STRING API) as ``source, target, score`` edges.

    The STRING API ``/network`` endpoint returns ``preferredName_A`` / ``preferredName_B`` gene
    symbols and a combined ``score`` in [0, 1]. Edges are undirected functional associations.
    """
    df = pd.read_csv(path, sep="\t")
    a = "preferredName_A" if "preferredName_A" in df.columns else df.columns[2]
    b = "preferredName_B" if "preferredName_B" in df.columns else df.columns[3]
    score = "score" if "score" in df.columns else df.columns[-1]
    return pd.DataFrame({
        "source": df[a].astype(str),
        "target": df[b].astype(str),
        "score": df[score].astype(float),
    })


def skeleton_truth_matrix(edges: pd.DataFrame, genes, *, min_score: float = 0.4) -> np.ndarray:
    """Build a symmetric (undirected) boolean truth matrix over ``genes`` from scored edges.

    An off-diagonal entry is 1 when the gene pair has an edge with ``score >= min_score`` in
    either direction. Suitable for grading an edge score's SKELETON (interaction present),
    since STRING associations are undirected.
    """
    genes = [str(g) for g in genes]
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    T = np.zeros((n, n))
    for s, t, sc in zip(edges["source"], edges["target"], edges["score"]):
        if float(sc) >= min_score and s in idx and t in idx and s != t:
            i, j = idx[s], idx[t]
            T[i, j] = 1.0
            T[j, i] = 1.0
    np.fill_diagonal(T, 0.0)
    return T
