"""Adapter for BEELINE-format single-cell GRN benchmark datasets.

BEELINE (Pratapa et al. 2020) ships each dataset as a small set of CSV files:

* ``ExpressionData.csv`` - genes x cells (genes in rows), raw or normalized.
* ``refNetwork.csv``     - directed reference edges (columns ``Gene1, Gene2``).
* ``PseudoTime.csv``     - optional per-cell pseudotime (one or more trajectories).
* an optional TF/regulator list.

This module turns those files into a :class:`GrnBenchmarkDataset` that the
existing DREAM4-style scorers and evaluation helpers can consume: an expression
matrix oriented as samples (cells) x genes, a directed candidate edge set
(TF -> gene when a regulator list is available, otherwise all directed non-self
gene pairs), and a densified ``edge_labels`` table (every candidate edge gets
``is_true`` in {0, 1}).

Important caveats vs DREAM4:

* References are biological proxies (cell-type / non-specific ChIP-seq, or curated
  functional priors), not exact simulator truth. AUPR here is *agreement with a
  prior*, so pair it with early-precision / EPR and the ``reference_kind`` label.
* DREAM4 dynamic lagged methods do not transfer without time/pseudotime; static
  single-cell snapshots support correlation, GENIE3/trees, static sparse, and
  rank fusion.

Loaders never read or use the gold labels; ``is_true`` lives only in
``edge_labels`` and is for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPRESSION_FILE = "ExpressionData.csv"
REFERENCE_FILE = "refNetwork.csv"
PSEUDOTIME_FILE = "PseudoTime.csv"
TF_FILE_CANDIDATES = ("TFs.csv", "TF.csv", "tf_list.csv", "tfs.txt", "TFs.txt")

# Map a BEELINE reference label to how trustworthy the edges are.
_REFERENCE_KIND = {
    "cell_type_specific": "chip_seq_proxy",
    "non_specific": "chip_seq_proxy",
    "string": "functional_prior",
    "functional": "functional_prior",
    "synthetic": "exact",
    "boolode": "exact",
    "exact": "exact",
}


@dataclass
class GrnBenchmarkDataset:
    """A GRN benchmark in a model-agnostic shape.

    Attributes
    ----------
    name:
        Dataset name.
    expression:
        Expression matrix as samples/cells (rows) x genes (columns).
    genes:
        Gene names (``expression`` columns).
    samples:
        Sample/cell ids (``expression`` index).
    candidate_edges:
        Directed candidate edges (columns ``source``, ``target``); the universe
        a scorer must rank. Self-edges are excluded.
    reference_edges:
        Directed reference edges present within the gene space (columns
        ``source``, ``target``); the "true" edges.
    edge_labels:
        ``candidate_edges`` densified with an ``is_true`` column in {0, 1}. This
        is the evaluation target; the only place gold labels live.
    metadata:
        Provenance and shape info (source, reference kind, densities, ...).
    tf_list:
        Candidate regulators (sources), or ``None`` if all genes are regulators.
    pseudotime:
        Optional per-sample pseudotime (rows align to ``expression``), else ``None``.
    perturbation_labels:
        Optional per-sample perturbation label, else ``None`` (BEELINE is
        observational).
    """

    name: str
    expression: pd.DataFrame
    genes: list[str]
    samples: list[str]
    candidate_edges: pd.DataFrame
    reference_edges: pd.DataFrame
    edge_labels: pd.DataFrame
    metadata: dict
    tf_list: list[str] | None = None
    pseudotime: pd.DataFrame | None = None
    perturbation_labels: pd.Series | None = None


# --------------------------------------------------------------------------- #
# Expression reading + orientation
# --------------------------------------------------------------------------- #
def infer_expression_orientation(frame: pd.DataFrame, *, gene_hint: set[str] | None = None) -> str:
    """Decide whether genes are in rows or columns of a raw expression frame.

    Returns ``"genes_in_rows"`` (BEELINE's convention) or ``"genes_in_columns"``.
    If ``gene_hint`` (known gene names, e.g. from the reference) is given, the
    axis whose labels overlap it more is taken as the gene axis. Without a hint,
    defaults to BEELINE's ``genes_in_rows``.
    """
    if gene_hint:
        hint = {str(g) for g in gene_hint}
        row_overlap = len(hint & {str(r) for r in frame.index})
        col_overlap = len(hint & {str(c) for c in frame.columns})
        if row_overlap != col_overlap:
            return "genes_in_rows" if row_overlap > col_overlap else "genes_in_columns"
    return "genes_in_rows"


def _looks_raw_counts(expression: pd.DataFrame) -> bool:
    """Heuristic: does the matrix look like raw counts (so log1p is appropriate)?"""
    values = expression.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() < 0.0:
        return False  # negatives => already transformed/standardized
    integer_like = bool(np.allclose(finite, np.round(finite)))
    return integer_like or float(finite.max()) > 30.0


def _orient_and_normalize(raw: pd.DataFrame, *, log1p: bool, gene_hint: set[str] | None):
    """Return (expression samples x genes, orientation, log1p_applied)."""
    orientation = infer_expression_orientation(raw, gene_hint=gene_hint)
    expression = raw.T.copy() if orientation == "genes_in_rows" else raw.copy()
    expression = expression.apply(pd.to_numeric)
    expression.columns = [str(c) for c in expression.columns]
    expression.index = [str(i) for i in expression.index]
    applied = bool(log1p and _looks_raw_counts(expression))
    if applied:
        expression = np.log1p(expression)
    return expression, orientation, applied


def read_beeline_expression(
    path: str | Path, *, log1p: bool = True, gene_hint: set[str] | None = None
) -> pd.DataFrame:
    """Read a BEELINE ``ExpressionData.csv`` as a samples/cells x genes matrix.

    Orientation is detected (BEELINE stores genes in rows) and corrected, and
    ``log1p`` is applied only when the values look raw/count-like.
    """
    raw = pd.read_csv(path, index_col=0)
    expression, _, _ = _orient_and_normalize(raw, log1p=log1p, gene_hint=gene_hint)
    return expression


# --------------------------------------------------------------------------- #
# Reference edges, candidate edges, labels
# --------------------------------------------------------------------------- #
def read_beeline_reference_edges(path: str | Path) -> pd.DataFrame:
    """Read a BEELINE ``refNetwork.csv`` as directed ``source -> target`` edges.

    Tolerates common column names (``Gene1/Gene2``, ``source/target``,
    ``regulator/target``); falls back to the first two columns.
    """
    reference = pd.read_csv(path)
    if reference.shape[1] < 2:
        raise ValueError("refNetwork must have at least two columns (source, target)")
    lower = {str(c).lower(): c for c in reference.columns}
    source_col = lower.get("gene1") or lower.get("source") or lower.get("regulator") or reference.columns[0]
    target_col = lower.get("gene2") or lower.get("target") or reference.columns[1]
    edges = pd.DataFrame(
        {
            "source": reference[source_col].astype(str),
            "target": reference[target_col].astype(str),
        }
    )
    edges = edges[edges["source"] != edges["target"]]
    return edges.drop_duplicates().reset_index(drop=True)


def build_tf_to_gene_candidate_edges(
    genes: list[str], tf_list: list[str] | None = None
) -> pd.DataFrame:
    """Build the directed candidate edge set.

    With ``tf_list``, sources are restricted to regulators present among
    ``genes`` (TF -> gene). Without one, all directed non-self gene pairs are
    used (less biologically realistic; documented in metadata). Self-edges are
    always excluded.
    """
    genes = [str(g) for g in genes]
    gene_set = set(genes)
    if tf_list:
        sources = [str(t) for t in tf_list if str(t) in gene_set]
    else:
        sources = list(genes)
    rows = [
        {"source": source, "target": target}
        for source in sources
        for target in genes
        if source != target
    ]
    return pd.DataFrame(rows, columns=["source", "target"])


def label_candidate_edges(
    candidate_edges: pd.DataFrame, reference_edges: pd.DataFrame
) -> pd.DataFrame:
    """Densify labels: each candidate edge gets ``is_true`` = 1 if in reference."""
    reference_set = set(
        zip(reference_edges["source"].astype(str), reference_edges["target"].astype(str))
    )
    labels = candidate_edges.copy()
    labels["is_true"] = [
        int((str(source), str(target)) in reference_set)
        for source, target in zip(labels["source"], labels["target"])
    ]
    return labels.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Optional files
# --------------------------------------------------------------------------- #
def _read_pseudotime(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    pseudotime = pd.read_csv(path, index_col=0)
    pseudotime.index = [str(i) for i in pseudotime.index]
    return pseudotime


def _resolve_tf_list(base: Path, tf_list: object, genes: list[str]) -> list[str] | None:
    """Resolve a TF list from a passed list/path, a dataset TF file, or None."""
    gene_set = set(genes)
    if isinstance(tf_list, (list, tuple, set)):
        resolved = [str(t) for t in tf_list]
    elif isinstance(tf_list, (str, Path)) and Path(tf_list).exists():
        resolved = _read_tf_file(Path(tf_list))
    else:
        resolved = None
        for candidate in TF_FILE_CANDIDATES:
            path = base / candidate
            if path.exists():
                resolved = _read_tf_file(path)
                break
    if resolved is None:
        return None
    present = [t for t in resolved if t in gene_set]
    return present or None


def _read_tf_file(path: Path) -> list[str]:
    """Read a TF list from a one-column csv/txt (header optional)."""
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = frame.columns[0]
        # if the header itself looks like a gene (no 'tf'/'gene' word), include it
        values = frame[column].astype(str).tolist()
        if not any(token in str(column).lower() for token in ("tf", "gene", "regulator", "name")):
            values = [str(column)] + values
        return values
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_beeline_dataset(
    root: str | Path,
    name: str,
    *,
    reference: str = "cell_type_specific",
    tf_list: object = None,
    log1p: bool = True,
) -> GrnBenchmarkDataset:
    """Load one BEELINE-format dataset directory into a :class:`GrnBenchmarkDataset`.

    Parameters
    ----------
    root:
        Directory containing dataset subfolders (e.g. ``data/raw/beeline``).
    name:
        Dataset subfolder name (contains ``ExpressionData.csv`` etc.).
    reference:
        Label describing the reference network's nature; controls
        ``metadata["reference_kind"]`` (does not by itself pick a file).
    tf_list:
        A list of regulators, a path to a TF file, or ``None`` (then a TF file in
        the dataset folder is used if present, else all genes are regulators).
    log1p:
        Apply ``log1p`` when the expression looks raw/count-like.
    """
    base = Path(root) / name
    expression_path = base / EXPRESSION_FILE
    if not expression_path.exists():
        raise FileNotFoundError(
            f"BEELINE expression file not found: {expression_path}. "
            f"Place a BEELINE dataset under {base}."
        )

    reference_path = base / REFERENCE_FILE
    raw_reference = (
        read_beeline_reference_edges(reference_path)
        if reference_path.exists()
        else pd.DataFrame(columns=["source", "target"])
    )

    gene_hint: set[str] | None = None
    if not raw_reference.empty:
        gene_hint = set(raw_reference["source"]) | set(raw_reference["target"])
    elif isinstance(tf_list, (list, tuple, set)):
        gene_hint = {str(t) for t in tf_list}

    raw_expression = pd.read_csv(expression_path, index_col=0)
    expression, orientation, log1p_applied = _orient_and_normalize(
        raw_expression, log1p=log1p, gene_hint=gene_hint
    )
    genes = list(expression.columns)
    samples = list(expression.index)
    gene_set = set(genes)

    # keep only reference edges whose genes are present in the expression matrix
    n_reference_raw = len(raw_reference)
    reference_edges = raw_reference[
        raw_reference["source"].isin(gene_set) & raw_reference["target"].isin(gene_set)
    ].reset_index(drop=True)
    n_reference_missing_genes = n_reference_raw - len(reference_edges)

    resolved_tf = _resolve_tf_list(base, tf_list, genes)
    candidate_edges = build_tf_to_gene_candidate_edges(genes, resolved_tf)
    edge_labels = label_candidate_edges(candidate_edges, reference_edges)

    # reference edges whose source is not in the candidate (TF) space cannot be recovered
    candidate_pairs = set(zip(candidate_edges["source"], candidate_edges["target"]))
    n_reference_outside_candidates = sum(
        (str(s), str(t)) not in candidate_pairs
        for s, t in zip(reference_edges["source"], reference_edges["target"])
    )

    pseudotime = _read_pseudotime(base / PSEUDOTIME_FILE)
    n_true = int(edge_labels["is_true"].sum())
    n_candidate = len(candidate_edges)
    metadata = {
        "source": "beeline",
        "name": name,
        "reference": reference,
        "reference_kind": _REFERENCE_KIND.get(str(reference).lower(), "chip_seq_proxy"),
        "n_genes": len(genes),
        "n_samples": len(samples),
        "n_candidate_edges": n_candidate,
        "n_reference_edges": len(reference_edges),
        "n_true_edges": n_true,
        "true_edge_density": (n_true / n_candidate) if n_candidate else 0.0,
        "tf_restricted": resolved_tf is not None,
        "n_regulators": len(resolved_tf) if resolved_tf is not None else len(genes),
        "expression_orientation": orientation,
        "log1p_applied": log1p_applied,
        "has_pseudotime": pseudotime is not None,
        "has_reference": not raw_reference.empty,
        "n_reference_dropped_missing_genes": int(n_reference_missing_genes),
        "n_reference_outside_candidate_space": int(n_reference_outside_candidates),
    }

    return GrnBenchmarkDataset(
        name=name,
        expression=expression,
        genes=genes,
        samples=samples,
        candidate_edges=candidate_edges,
        reference_edges=reference_edges,
        edge_labels=edge_labels,
        metadata=metadata,
        tf_list=resolved_tf,
        pseudotime=pseudotime,
        perturbation_labels=None,
    )
