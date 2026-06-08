"""Loader for the RENGE time-resolved CRISPR Perturb-seq dataset (GEO GSE213069).

Real time-resolved single-cell CRISPR knockout in human iPSCs: four daily timepoints (days 2
to 5 after transduction), 23 knocked-out transcription factors plus non-targeting controls,
standard 10x CRISPR guide-capture output (``barcodes.tsv.gz``, ``features.tsv.gz``,
``matrix.mtx.gz``) per day. This is the real time-resolved counterpart to the static RPE1
data, with a time axis the static analysis lacked.

The loader assigns each cell its dominant guide (CRISPR Guide Capture features), maps the guide
to its target gene (the prefix before the guide-number suffix, e.g. ``SOX2_1`` -> ``SOX2``),
treats the safe-harbor / non-targeting guides (AAVS1, CTRL) as control, and returns a normalized
expression matrix restricted to the perturbed transcription factors (the square block on which
an interventional response operator is defined). Per-cell total UMI is computed over the full
gene block without densifying it; only the selected gene columns are materialized.

Requires ``scipy`` (sparse Matrix Market reader). No network access; point it at the extracted
GSE213069 day folders.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .interventional import CONTROL_LABEL, load_interventional_frames

# Safe-harbor (AAVS1) and explicit control (CTRL) guides are the non-targeting controls.
RENGE_CONTROL_TARGETS = frozenset({"AAVS1", "CTRL"})
RENGE_DAYS = ("day2", "day3", "day4", "day5")
GUIDE_TYPE = "CRISPR Guide Capture"


def parse_guide_target(guide_name: str) -> str:
    """Map a guide feature name to its target gene (the prefix before the guide-number suffix).

    ``SOX2_1`` -> ``SOX2``; ``ZNF398_2`` -> ``ZNF398``; ``AAVS1_1`` -> ``AAVS1``.
    """
    return str(guide_name).split("_")[0]


def _read_10x(day_dir: Path):
    """Read a 10x triplet (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz) as (csr, features, barcodes).

    The matrix is features x cells (Cell Ranger convention).
    """
    from scipy.io import mmread

    day_dir = Path(day_dir)
    with gzip.open(day_dir / "matrix.mtx.gz", "rb") as fh:
        matrix = mmread(fh).tocsr()
    features = pd.read_csv(day_dir / "features.tsv.gz", sep="\t", header=None, compression="gzip")
    features = features.iloc[:, : 3]
    features.columns = ["id", "name", "type"][: features.shape[1]]
    barcodes = pd.read_csv(day_dir / "barcodes.tsv.gz", sep="\t", header=None, compression="gzip")
    return matrix, features, list(barcodes.iloc[:, 0].astype(str))


def assign_guides(
    guide_counts: np.ndarray,
    guide_targets: np.ndarray,
    *,
    min_umi: int = 1,
    dominance: float = 2.0,
) -> np.ndarray:
    """Assign each cell its target gene from per-cell guide counts (cells x guides).

    A cell is assigned to the target of its top guide when that guide has at least ``min_umi``
    counts and is at least ``dominance`` times the second guide (a confident single assignment);
    otherwise the cell is ``unassigned``. Control-target guides map to ``CONTROL_LABEL``.
    """
    G = np.asarray(guide_counts, dtype=float)
    if G.shape[1] == 0:
        return np.array(["unassigned"] * G.shape[0])
    top = G.argmax(axis=1)
    top_val = G.max(axis=1)
    if G.shape[1] >= 2:
        second_val = np.partition(G, -2, axis=1)[:, -2]
    else:
        second_val = np.zeros(G.shape[0])
    confident = (top_val >= min_umi) & (top_val >= dominance * np.maximum(second_val, 1.0))
    target = np.array([guide_targets[i] for i in top])
    labels = np.where(confident, target, "unassigned")
    is_control = np.isin(labels, list(RENGE_CONTROL_TARGETS))
    return np.where(is_control, CONTROL_LABEL, labels)


def load_renge_day(
    day_dir: str | Path,
    *,
    target_genes: list[str] | None = None,
    min_umi: int = 1,
    dominance: float = 2.0,
):
    """Load one RENGE day folder into ``(expression, perturbation_labels)``.

    ``expression`` is cells x genes (the perturbed transcription factors that are also measured,
    or ``target_genes`` if given), per-cell normalized to the median total UMI then log1p.
    ``perturbation_labels`` is a per-cell Series of target gene or ``CONTROL_LABEL`` (cells with
    no confident guide are dropped).
    """
    matrix, features, barcodes = _read_10x(Path(day_dir))
    is_guide = (features["type"] == GUIDE_TYPE).to_numpy()
    gene_rows = np.where(~is_guide)[0]
    guide_rows = np.where(is_guide)[0]

    guide_targets = np.array([parse_guide_target(n) for n in features["name"].to_numpy()[guide_rows]])
    guide_counts = np.asarray(matrix[guide_rows].todense()).T  # cells x guides (small)
    labels = assign_guides(guide_counts, guide_targets, min_umi=min_umi, dominance=dominance)

    # per-cell total UMI over the full gene block (sparse column sums; no densify)
    total_umi = np.asarray(matrix[gene_rows].sum(axis=0)).ravel().astype(float)
    median_umi = float(np.median(total_umi[total_umi > 0])) if np.any(total_umi > 0) else 1.0

    gene_syms = features["name"].to_numpy()[gene_rows]
    perturbed_targets = sorted({t for t in labels if t not in (CONTROL_LABEL, "unassigned")})
    wanted = target_genes if target_genes is not None else perturbed_targets
    sym_to_row: dict[str, int] = {}
    for local_idx, sym in enumerate(gene_syms):
        sym_to_row.setdefault(str(sym), gene_rows[local_idx])
    cols = [g for g in wanted if g in sym_to_row]
    sub_rows = [sym_to_row[g] for g in cols]
    sub = np.asarray(matrix[sub_rows].todense()).T.astype(float)  # cells x len(cols)

    scale = np.divide(median_umi, total_umi, out=np.ones_like(total_umi), where=total_umi > 0)
    expr = np.log1p(sub * scale[:, None])

    keep = labels != "unassigned"
    expr_df = pd.DataFrame(expr[keep], columns=cols, index=[b for b, k in zip(barcodes, keep) if k])
    label_series = pd.Series(labels[keep], index=expr_df.index, name="perturbation")
    return expr_df, label_series


def load_renge_day_hvg(
    day_dir: str | Path,
    *,
    n_hvg: int = 1000,
    min_umi: int = 1,
    dominance: float = 2.0,
):
    """Load a RENGE day as ``(expression, perturbation_labels)`` over the top high-variance genes.

    Unlike :func:`load_renge_day` (which keeps only the perturbed transcription factors), this keeps
    the ``n_hvg`` most variable gene-expression features for program discovery and heterogeneity
    analysis. Per-gene variance is computed from the sparse matrix without densifying; only the
    selected columns are materialized, then per-cell normalized to the median UMI and log1p.
    """
    matrix, features, barcodes = _read_10x(Path(day_dir))
    is_guide = (features["type"] == GUIDE_TYPE).to_numpy()
    gene_rows = np.where(~is_guide)[0]
    guide_rows = np.where(is_guide)[0]

    guide_targets = np.array([parse_guide_target(n) for n in features["name"].to_numpy()[guide_rows]])
    guide_counts = np.asarray(matrix[guide_rows].todense()).T
    labels = assign_guides(guide_counts, guide_targets, min_umi=min_umi, dominance=dominance)

    G = matrix[gene_rows]
    n_cells = G.shape[1]
    s = np.asarray(G.sum(axis=1)).ravel()
    s2 = np.asarray(G.multiply(G).sum(axis=1)).ravel()
    var = s2 / n_cells - (s / n_cells) ** 2
    top = np.argsort(var)[::-1][:n_hvg]
    sel_rows = gene_rows[top]
    sel_syms = features["name"].to_numpy()[sel_rows]

    total_umi = np.asarray(matrix[gene_rows].sum(axis=0)).ravel().astype(float)
    median_umi = float(np.median(total_umi[total_umi > 0])) if np.any(total_umi > 0) else 1.0
    sub = np.asarray(matrix[sel_rows].todense()).T.astype(float)
    scale = np.divide(median_umi, total_umi, out=np.ones_like(total_umi), where=total_umi > 0)
    expr = np.log1p(sub * scale[:, None])

    keep = labels != "unassigned"
    cols = [str(g) for g in sel_syms]
    expr_df = pd.DataFrame(expr[keep], columns=cols, index=[b for b, k in zip(barcodes, keep) if k])
    label_series = pd.Series(labels[keep], index=expr_df.index, name="perturbation")
    return expr_df, label_series


def load_renge_timecourse(
    root: str | Path,
    *,
    days: tuple[str, ...] = RENGE_DAYS,
    min_umi: int = 1,
    dominance: float = 2.0,
):
    """Load the RENGE day folders into a dict ``day -> InterventionalDataset`` over a shared gene set.

    The gene set is the perturbed transcription factors present and measured in every loaded day,
    so each day's response matrix is the same square block and the days are comparable across
    time. ``root`` contains the extracted ``dayN/dayN/`` 10x folders.
    """
    root = Path(root)
    raw = {}
    for day in days:
        day_dir = root / day / day
        if not day_dir.exists():
            day_dir = root / day
        if not (day_dir / "matrix.mtx.gz").exists():
            continue
        raw[day] = load_renge_day(day_dir, min_umi=min_umi, dominance=dominance)
    if not raw:
        raise FileNotFoundError(f"No RENGE day folders with matrix.mtx.gz under {root}.")

    # shared perturbed gene set: targets that appear (non-control) and are measured in every day
    shared = None
    for expr, labels in raw.values():
        perturbed = set(labels[labels != CONTROL_LABEL].unique()) & set(expr.columns)
        shared = perturbed if shared is None else (shared & perturbed)
    shared = sorted(shared or [])

    datasets = {}
    for day, (expr, labels) in raw.items():
        cols = [g for g in shared if g in expr.columns]
        datasets[day] = load_interventional_frames(
            f"renge_{day}", expr[cols], labels, reference_kind="perturbation_derived"
        )
    return datasets
