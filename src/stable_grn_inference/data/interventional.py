"""Adapter shape + diagnostics for interventional (perturbation) GRN benchmarks.

Motivation (experiments 17-18). DREAM4 gave orientation almost for free because it
is lagged time-series (temporal precedence). BEELINE Curated showed that on *static
observational* single-cell data, orientation-given-skeleton is weak and highly
network-dependent (GSD collapses to chance, VSC orients perfectly). The next regime
that can actually settle direction is **interventional / perturbation data**, where
direction is read from an *intervention asymmetry*: knocking down gene A should move
gene B more than knocking down B moves A, when A -> B.

This module is the adapter + diagnostic SHAPE for that regime. It deliberately does
*not* download any large real dataset. It provides:

* :class:`InterventionalDataset` - a model-agnostic container (cells x genes plus a
  per-cell perturbation label and a control mask), mirroring the CausalBench /
  Perturb-seq schema (Replogle et al. CRISPRi screens: RPE1 ~163k cells / 383
  interventions, K562 ~163k cells / 622 interventions; ~10-11k observational cells).
* :func:`make_synthetic_interventional` - a tiny linear-SEM fixture with a known DAG,
  control cells, and single-gene knockdowns, matching the expected on-disk schema so
  the pipeline can be exercised with zero download.
* :func:`load_interventional_frames` - build a dataset from an expression matrix plus a
  per-cell perturbation label column (the real-data ingest shape).
* :func:`interventional_effect_matrix` - the CausalBench statistical signal: for edge
  A -> B, the Wasserstein distance between B's expression under perturb(A) vs control.
* :func:`interventional_orientation_asymmetry` - the REBUILT orientation diagnostic.
  The experiment-17/18 observational ``orientation_accuracy_given_skeleton`` (compare
  static edge scores for i->j vs j->i) is the wrong instrument here; direction comes
  from comparing interventional effects effect(A->B) vs effect(B->A). Requires *both*
  endpoints to have been perturbed.

CAUTION (exploratory). Intervention asymmetry is directional *evidence*, not proof:
indirect/downstream effects, genetic compensation, off-target knockdown, and
cell-state shifts can all distort it. Treat its orientation accuracy as a diagnostic,
not a causal guarantee.

Loaders never read reference/true labels into the inference path; truth lives only in
``edge_labels`` / ``reference_edges`` for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_LABEL = "control"

# CausalBench / Replogle Perturb-seq schema, for reference when ingesting real data.
# Expression: cells (rows) x genes (cols). Per-cell perturbation label names the
# knocked-down gene, or CONTROL_LABEL for non-targeting / observational cells.
EXPRESSION_FILE = "expression.csv"
PERTURBATION_FILE = "perturbations.csv"  # one column: per-cell target gene or "control"


@dataclass
class InterventionalDataset:
    """An interventional GRN benchmark in a model-agnostic shape.

    Attributes
    ----------
    name:
        Dataset name.
    expression:
        Expression matrix, cells (rows) x genes (columns).
    genes, cells:
        Column / row labels of ``expression``.
    perturbation_labels:
        Per-cell Series (index = ``cells``) naming the perturbed gene, or
        ``CONTROL_LABEL`` for observational/non-targeting cells.
    is_control:
        Boolean Series, ``perturbation_labels == CONTROL_LABEL``.
    perturbed_genes:
        Sorted unique perturbed genes (intervention is available only on these, so
        only these can be directed-edge *sources* via interventional evidence).
    candidate_edges:
        Directed candidate edges (columns ``source``, ``target``); sources restricted
        to ``perturbed_genes``.
    reference_edges:
        Optional directed reference edges (columns ``source``, ``target``). For real
        Perturb-seq these are PROXIES (CORUM/STRING/ChIP-seq), not exact truth.
    edge_labels:
        Densified candidate table with ``is_true`` in {0, 1} (empty when no reference).
    metadata:
        Free-form dict: ``reference_kind`` (exact / chip_seq_proxy / functional_prior /
        perturbation_derived), ``n_perturbations``, ``n_control_cells``, etc.
    """

    name: str
    expression: pd.DataFrame
    genes: list[str]
    cells: list[str]
    perturbation_labels: pd.Series
    is_control: pd.Series
    perturbed_genes: list[str]
    candidate_edges: pd.DataFrame
    reference_edges: pd.DataFrame
    edge_labels: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    tf_list: list[str] | None = None


def build_candidate_edges_from_perturbations(
    genes: list[str], perturbed_genes: list[str], *, tf_list: list[str] | None = None
) -> pd.DataFrame:
    """Directed candidates source -> target with source in perturbed_genes (and tf_list
    if given), target any other gene. Interventional evidence exists only for perturbed
    sources, so the candidate universe is restricted accordingly."""
    sources = [g for g in perturbed_genes if (tf_list is None or g in set(tf_list))]
    rows = [(s, t) for s in sources for t in genes if s != t]
    return pd.DataFrame(rows, columns=["source", "target"])


def label_candidate_edges(
    candidate_edges: pd.DataFrame, reference_edges: pd.DataFrame
) -> pd.DataFrame:
    """Densify: every candidate edge gets is_true in {0,1} from the reference set."""
    labels = candidate_edges.copy()
    if reference_edges is None or len(reference_edges) == 0:
        labels["is_true"] = 0
        return labels
    truth = {(s, t) for s, t in zip(reference_edges["source"], reference_edges["target"])}
    labels["is_true"] = [
        1 if (s, t) in truth else 0
        for s, t in zip(labels["source"], labels["target"])
    ]
    return labels


def load_interventional_frames(
    name: str,
    expression: pd.DataFrame,
    perturbation_labels: pd.Series,
    *,
    reference_edges: pd.DataFrame | None = None,
    reference_kind: str = "perturbation_derived",
    tf_list: list[str] | None = None,
    control_label: str = CONTROL_LABEL,
) -> InterventionalDataset:
    """Build an :class:`InterventionalDataset` from in-memory frames (the real-ingest
    shape: a cells x genes matrix and a per-cell perturbation label)."""
    expression = expression.copy()
    genes = list(map(str, expression.columns))
    expression.columns = genes
    cells = list(map(str, expression.index))
    expression.index = cells

    perturbation_labels = perturbation_labels.copy()
    perturbation_labels.index = cells
    perturbation_labels = perturbation_labels.astype(str)
    is_control = perturbation_labels == control_label

    gene_set = set(genes)
    perturbed = sorted(
        {p for p in perturbation_labels[~is_control].unique() if p in gene_set}
    )

    candidate_edges = build_candidate_edges_from_perturbations(
        genes, perturbed, tf_list=tf_list
    )
    if reference_edges is None:
        reference_edges = pd.DataFrame(columns=["source", "target"])
    edge_labels = label_candidate_edges(candidate_edges, reference_edges)

    metadata = {
        "reference_kind": reference_kind,
        "has_reference": len(reference_edges) > 0,
        "n_genes": len(genes),
        "n_cells": len(cells),
        "n_perturbations": len(perturbed),
        "n_control_cells": int(is_control.sum()),
        "n_candidate_edges": len(candidate_edges),
        "n_true_edges": int(edge_labels["is_true"].sum()),
    }
    return InterventionalDataset(
        name=name,
        expression=expression,
        genes=genes,
        cells=cells,
        perturbation_labels=perturbation_labels,
        is_control=is_control,
        perturbed_genes=perturbed,
        candidate_edges=candidate_edges,
        reference_edges=reference_edges,
        edge_labels=edge_labels,
        metadata=metadata,
        tf_list=tf_list,
    )


def make_synthetic_interventional(
    *,
    n_genes: int = 6,
    n_cells_per_condition: int = 80,
    edge_density: float = 0.35,
    effect: float = 1.6,
    knockdown_shift: float = -3.0,
    noise: float = 0.4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Tiny linear-SEM interventional fixture with a known DAG.

    Returns ``(expression, perturbation_labels, true_edges)`` matching the expected
    on-disk schema: ``expression`` is cells x genes, ``perturbation_labels`` is a
    per-cell Series (gene name or ``control``), ``true_edges`` has columns
    ``source, target``.

    The DAG is acyclic with weights along a topological order, so knocking down an
    ancestor shifts its descendants (large effect down-edges) while knocking down a
    descendant leaves ancestors unchanged - i.e. the intervention asymmetry recovers
    direction by construction. This is the positive-control fixture for
    :func:`interventional_orientation_asymmetry`.
    """
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]

    # upper-triangular weight matrix W[parent, child] over the topological order
    weight = np.zeros((n_genes, n_genes))
    true_pairs: list[tuple[str, str]] = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if rng.random() < edge_density:
                weight[i, j] = effect * rng.uniform(0.7, 1.3)
                true_pairs.append((genes[i], genes[j]))

    def simulate(n: int, knockout: int | None) -> np.ndarray:
        x = np.zeros((n, n_genes))
        for j in range(n_genes):
            x[:, j] = x @ weight[:, j] + rng.normal(0.0, noise, size=n)
            if knockout is not None and j == knockout:
                x[:, j] = knockdown_shift + rng.normal(0.0, noise, size=n)
        return x

    blocks, labels = [], []
    control = simulate(n_cells_per_condition, None)
    blocks.append(control)
    labels.extend([CONTROL_LABEL] * n_cells_per_condition)
    for k in range(n_genes):  # every gene is perturbed (so asymmetry is defined on all pairs)
        blocks.append(simulate(n_cells_per_condition, k))
        labels.extend([genes[k]] * n_cells_per_condition)

    data = np.vstack(blocks)
    cells = [f"c{i}" for i in range(data.shape[0])]
    expression = pd.DataFrame(data, index=cells, columns=genes)
    perturbation_labels = pd.Series(labels, index=cells, name="perturbation")
    true_edges = pd.DataFrame(true_pairs, columns=["source", "target"])
    return expression, perturbation_labels, true_edges


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance, scipy when available else a quantile-grid fallback."""
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except Exception:  # pragma: no cover - scipy expected in this project
        qs = np.linspace(0.0, 1.0, 101)
        return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def interventional_effect_matrix(dataset: InterventionalDataset) -> pd.DataFrame:
    """CausalBench statistical signal. For each perturbed source A and each target B,
    the Wasserstein distance between B under perturb(A) and B under control. Returns a
    long DataFrame (``source``, ``target``, ``effect``) over ``candidate_edges``."""
    expr = dataset.expression
    control_mask = dataset.is_control.to_numpy()
    rows = []
    for source in dataset.perturbed_genes:
        pert_mask = (dataset.perturbation_labels == source).to_numpy()
        for target in dataset.genes:
            if target == source:
                continue
            b_pert = expr[target].to_numpy()[pert_mask]
            b_ctrl = expr[target].to_numpy()[control_mask]
            if b_pert.size == 0 or b_ctrl.size == 0:
                continue
            rows.append((source, target, _wasserstein(b_pert, b_ctrl)))
    return pd.DataFrame(rows, columns=["source", "target", "effect"])


def interventional_orientation_asymmetry(
    dataset: InterventionalDataset,
    effect_matrix: pd.DataFrame | None = None,
    *,
    true_only: bool = True,
) -> dict:
    """Rebuilt orientation diagnostic for the interventional regime.

    For each unordered pair {A, B} where *both* A and B were perturbed, direction is
    inferred from the larger interventional effect: predict A -> B if
    effect(A->B) > effect(B->A). Orientation accuracy is the fraction of (true,
    non-reciprocal) pairs whose predicted direction matches the reference, using only
    pairs where both endpoints are perturbed.

    Returns a dict with ``accuracy``, ``n_pairs``, ``n_pairs_both_perturbed``,
    ``n_reciprocal_excluded``, and a per-pair DataFrame ``pairs``.
    """
    if effect_matrix is None:
        effect_matrix = interventional_effect_matrix(dataset)
    eff = {
        (s, t): float(e)
        for s, t, e in zip(effect_matrix["source"], effect_matrix["target"], effect_matrix["effect"])
    }
    perturbed = set(dataset.perturbed_genes)

    truth = {
        (s, t) for s, t in zip(dataset.reference_edges["source"], dataset.reference_edges["target"])
    }
    # which unordered pairs to evaluate
    if true_only:
        directed = list(truth)
    else:
        directed = list(
            zip(dataset.candidate_edges["source"], dataset.candidate_edges["target"])
        )

    seen: set[frozenset] = set()
    records = []
    n_recip = 0
    for s, t in directed:
        key = frozenset((s, t))
        if key in seen or s == t:
            continue
        seen.add(key)
        reciprocal_true = (s, t) in truth and (t, s) in truth
        if reciprocal_true:
            n_recip += 1
            continue
        both = s in perturbed and t in perturbed
        # canonical true direction for this pair (when known)
        if (s, t) in truth:
            true_src, true_dst = s, t
        elif (t, s) in truth:
            true_src, true_dst = t, s
        else:
            true_src, true_dst = s, t  # no truth (true_only=False path)
        fwd = eff.get((true_src, true_dst), np.nan)
        rev = eff.get((true_dst, true_src), np.nan)
        correct = bool(fwd > rev) if both and np.isfinite(fwd) and np.isfinite(rev) else None
        records.append(
            {
                "source": true_src,
                "target": true_dst,
                "effect_forward": fwd,
                "effect_reverse": rev,
                "both_perturbed": both,
                "correct": correct,
            }
        )
    pairs = pd.DataFrame.from_records(records)
    scored = pairs[pairs["correct"].notna()] if len(pairs) else pairs
    accuracy = float(scored["correct"].mean()) if len(scored) else float("nan")
    return {
        "accuracy": accuracy,
        "n_pairs": int(len(pairs)),
        "n_pairs_both_perturbed": int(len(scored)),
        "n_reciprocal_excluded": int(n_recip),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Perturbation-response geometry (experiment 21).
#
# The shared object with Track B: an intervention produces a displacement vector
# (here, a gene perturbation -> an expression-response delta), and we study the
# GEOMETRY of those displacements -- their rank, sparsity, stability, and how much
# is a broad/global mode versus a sharp/direct effect. None of this assumes a gene
# ordering or graph, so no wavelets/scattering are forced onto unordered genes.
# ---------------------------------------------------------------------------


def perturbation_response_matrix(dataset: "InterventionalDataset", *, split_half: bool = False,
                                 seed: int = 0):
    """Mean-shift response matrix D[g, j] = mean(X_j | perturb g) - mean(X_j | control).

    Rows are perturbed genes, columns are all measured genes. This is the Track-A analog
    of Track B's transformation delta. If ``split_half`` is True, also returns two
    independent half-sample response matrices (D_a, D_b) for stability analysis: control
    and each perturbation's cells are split in half and the response recomputed on each.
    """
    expr = dataset.expression
    genes = list(expr.columns)
    X = expr.to_numpy(dtype=float)
    labels = dataset.perturbation_labels.to_numpy()
    ctrl = dataset.is_control.to_numpy()
    perturbed = list(dataset.perturbed_genes)

    def _matrix(row_mask_fn, ctrl_mask):
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        rows = [X[row_mask_fn(g)].mean(axis=0) - ctrl_mean for g in perturbed]
        return pd.DataFrame(np.vstack(rows), index=perturbed, columns=genes)

    full = _matrix(lambda g: labels == g, ctrl)
    if not split_half:
        return full

    rng = np.random.default_rng(seed)
    half = np.zeros(X.shape[0], dtype=int)
    for grp in [None] + perturbed:
        idx = np.where(ctrl if grp is None else (labels == grp))[0]
        sel = rng.permutation(idx)
        half[sel[: len(sel) // 2]] = 0
        half[sel[len(sel) // 2:]] = 1
    ca, cb = ctrl & (half == 0), ctrl & (half == 1)
    da = _matrix(lambda g: (labels == g) & (half == 0), ca)
    db = _matrix(lambda g: (labels == g) & (half == 1), cb)
    return full, da, db


def response_low_rank(response: pd.DataFrame, *, var_cutoff: float = 0.9) -> dict:
    """SVD spectrum of a response matrix: how low-rank / global-mode-dominated is it?

    Returns rank at the variance cutoff, normalized spectral entropy (1 = diffuse,
    ~0 = single dominant mode), singular values, and cumulative variance explained.
    """
    M = response.to_numpy(dtype=float)
    s = np.linalg.svd(M, compute_uv=False)
    var = s ** 2
    total = var.sum()
    frac = var / total if total > 0 else np.zeros_like(var)
    cum = np.cumsum(frac)
    rank = int(np.searchsorted(cum, var_cutoff) + 1)
    nz = frac[frac > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(frac))) if len(frac) > 1 else 0.0
    return {
        "rank_at_cutoff": rank,
        "spectral_entropy": entropy,
        "singular_values": s,
        "var_explained": frac,
        "cum_var_explained": cum,
        "top1_var": float(frac[0]) if len(frac) else float("nan"),
    }


def direct_effect_filter(response: pd.DataFrame, *, n_modes: int):
    """Remove the top ``n_modes`` global SVD modes from a response matrix, returning the
    'direct' residual responses (broad/global transcriptional component subtracted).
    Also returns the removed component. Use to sharpen a dense interventional response."""
    M = response.to_numpy(dtype=float)
    if n_modes <= 0:
        return response.copy(), response.iloc[:, :0].copy()
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(n_modes, len(s))
    broad = (U[:, :k] * s[:k]) @ Vt[:k]
    direct = M - broad
    idx, cols = response.index, response.columns
    return pd.DataFrame(direct, index=idx, columns=cols), pd.DataFrame(broad, index=idx, columns=cols)


def shared_response_program(response: pd.DataFrame) -> dict:
    """Decompose a response matrix into a shared global PROGRAM plus a gene-specific
    residual: D_g = a_g * p + R_g, where p is the (unit-norm) average response profile
    across perturbations and a_g = <D_g, p>. Unlike SVD-mode removal this is an
    interpretable covariate (the generic program every perturbation partly engages); the
    program is RETURNED as its own object, not discarded."""
    M = response.to_numpy(dtype=float)
    p = M.mean(axis=0)
    pn = p / (np.linalg.norm(p) or 1.0)
    a = M @ pn
    R = M - np.outer(a, pn)
    total = (M ** 2).sum()
    frac = float(1.0 - (R ** 2).sum() / total) if total > 0 else 0.0
    return {
        "program": pd.Series(pn, index=response.columns),
        "loadings": pd.Series(a, index=response.index),
        "residual": pd.DataFrame(R, index=response.index, columns=response.columns),
        "program_var_explained": frac,
    }


def residualize_against_covariates(response: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Remove the part of each gene's response explained by per-perturbation covariates
    (e.g. self-knockdown strength, log #cells) via per-column OLS. Returns residuals
    aligned to ``response``. Covariates are indexed by perturbation (``response.index``)."""
    cov = covariates.reindex(response.index)
    C = np.column_stack([np.ones(len(cov)), cov.to_numpy(dtype=float)])
    M = response.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(C, M, rcond=None)
    R = M - C @ beta
    return pd.DataFrame(R, index=response.index, columns=response.columns)


def response_sparsity(response: pd.DataFrame) -> pd.Series:
    """Per-perturbation diffuseness as an effective number of responding genes:
    L1^2 / L2^2 in [1, n_genes] (1 = one gene moves, n = uniform/diffuse)."""
    M = response.to_numpy(dtype=float)
    l1 = np.abs(M).sum(axis=1)
    l2sq = (M ** 2).sum(axis=1)
    eff = np.divide(l1 ** 2, l2sq, out=np.full(M.shape[0], np.nan), where=l2sq > 0)
    return pd.Series(eff, index=response.index, name="effective_responders")


def split_half_stability(response_a: pd.DataFrame, response_b: pd.DataFrame) -> pd.Series:
    """Per-perturbation cosine similarity between two independent half-sample responses.
    High = the response vector is reproducible (real); low = noise-dominated."""
    A = response_a.to_numpy(dtype=float)
    B = response_b.to_numpy(dtype=float)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    cos = np.divide(num, den, out=np.full(A.shape[0], np.nan), where=den > 0)
    return pd.Series(cos, index=response_a.index, name="split_half_cosine")


def detect_causalbench(search_dirs: list[Path] | None = None) -> dict:
    """Best-effort check for a local CausalBench install / data, without downloading.

    Returns a dict describing what is present so an experiment can report a blocker
    cleanly instead of attempting a large download."""
    info: dict = {"package_installed": False, "data_dirs_found": []}
    try:  # the pip package is 'causalscbench'
        import importlib.util

        info["package_installed"] = (
            importlib.util.find_spec("causalscbench") is not None
        )
    except Exception:
        info["package_installed"] = False
    for d in search_dirs or []:
        if Path(d).exists():
            info["data_dirs_found"].append(str(d))
    return info


# CausalBench / Weissmann (Replogle) figshare downloads, for reference. We do not
# auto-download in the library; an experiment or the user fetches these explicitly.
CAUSALBENCH_FILES = {
    "rpe1": "https://plus.figshare.com/ndownloader/files/35775606",
    "k562": "https://plus.figshare.com/ndownloader/files/35773219",
}


def _normalize_per_cell_log1p(counts: np.ndarray) -> np.ndarray:
    """Replicate CausalBench preprocessing: scanpy normalize_per_cell (scale each cell to
    the median total count) then log1p. ``counts`` is cells x genes raw counts."""
    totals = counts.sum(axis=1)
    median = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
    scale = np.divide(median, totals, out=np.ones_like(totals, dtype=float), where=totals > 0)
    normed = counts * scale[:, None]
    return np.log1p(normed)


def load_causalbench(
    h5ad_path: str | Path,
    *,
    name: str | None = None,
    min_cells_per_perturbation: int = 100,
    perturbed_genes_only: bool = True,
    control_obs_value: str = "non-targeting",
    gene_col: str = "gene",
    reference_edges: pd.DataFrame | None = None,
    reference_kind: str = "perturbation_derived",
) -> InterventionalDataset:
    """Load a CausalBench Weissmann/Replogle Perturb-seq ``.h5ad`` into an
    :class:`InterventionalDataset`.

    Mirrors CausalBench's own preprocessing: per-cell normalization + log1p, keep
    perturbations with > ``min_cells_per_perturbation`` cells, and (when
    ``perturbed_genes_only``) restrict the gene columns to the intersection of
    *measured* and *perturbed* genes - the square block on which the interventional
    effect / orientation diagnostics are defined. ``obs[gene_col] == control_obs_value``
    marks observational/control cells (mapped to ``CONTROL_LABEL``).

    Requires ``anndata`` (read-only); no network access.
    """
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    adata = ad.read_h5ad(h5ad_path)

    # gene (column) names: prefer var['gene_name'] if present, else var index
    if "gene_name" in adata.var.columns:
        var_genes = [str(g) for g in adata.var["gene_name"].to_numpy()]
    else:
        var_genes = [str(g) for g in adata.var_names]

    counts = adata.X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    expr = _normalize_per_cell_log1p(counts.astype(float))

    per_cell_gene = [str(g) for g in adata.obs[gene_col].to_numpy()]
    labels = pd.Series(
        [CONTROL_LABEL if g == control_obs_value else g for g in per_cell_gene]
    )

    # perturbations passing the cell-count floor (excluding control)
    counts_by_pert = labels[labels != CONTROL_LABEL].value_counts()
    kept_perts = {
        g for g, c in counts_by_pert.items()
        if c > min_cells_per_perturbation and g in set(var_genes)
    }

    expr_df = pd.DataFrame(expr, columns=var_genes)
    # collapse duplicate gene-name columns (Perturb-seq var can repeat names) by mean
    if expr_df.columns.duplicated().any():
        expr_df = expr_df.T.groupby(level=0).mean().T
        var_genes = list(expr_df.columns)

    if perturbed_genes_only:
        keep_cols = [g for g in expr_df.columns if g in kept_perts]
        expr_df = expr_df[keep_cols]

    cells = [f"cell{i}" for i in range(expr_df.shape[0])]
    expr_df.index = cells
    labels.index = cells
    # relabel interventions on dropped perturbations back to their gene name is fine;
    # load_interventional_frames intersects perturbed genes with measured columns.

    return load_interventional_frames(
        name or h5ad_path.stem,
        expr_df,
        labels,
        reference_edges=reference_edges,
        reference_kind=reference_kind,
        control_label=CONTROL_LABEL,
    )


def _decode_h5_categorical(node, categories_group, key):
    """Decode an AnnData categorical column (modern group of codes+categories, or legacy
    codes dataset + a shared ``__categories`` group), else a plain string/bytes column."""
    import h5py

    def _dec(arr):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])

    if isinstance(node, h5py.Group) and "codes" in node:
        return _dec(node["categories"][:])[node["codes"][:]]
    if categories_group is not None and key in categories_group:
        return _dec(categories_group[key][:])[node[:]]
    return _dec(node[:])


def load_replogle_raw_h5ad(
    path: str | Path,
    *,
    name: str = "replogle",
    min_cells: int = 100,
    max_perturbations: int | None = None,
    chunk: int = 20000,
    control_label: str = "non-targeting",
    gene_col: str = "gene",
    gene_name_col: str = "gene_name",
    umi_col: str = "UMI_count",
) -> InterventionalDataset:
    """Memory-efficient loader for a LARGE raw Replogle/Weissman Perturb-seq ``.h5ad``
    whose ``X`` is stored dense (e.g. RPE1: 247914 cells x 8749 genes, ~8.7 GB).

    Reads only the perturbed&measured gene-column block in row chunks (never densifying
    all genes), normalizes each cell by ``obs[umi_col]`` (= scanpy ``normalize_per_cell``
    to the median total count) then ``log1p`` - matching CausalBench preprocessing - and
    assembles an :class:`InterventionalDataset`. Keeps perturbations with more than
    ``min_cells`` cells; ``max_perturbations`` optionally caps to the most-sampled ones.

    Requires ``h5py``. No network access. Reference is left empty (real Perturb-seq has no
    exact directed truth); build an interventional reference downstream.
    """
    import collections

    import h5py

    path = Path(path)
    f = h5py.File(path, "r")
    try:
        obs_cat = f["obs"].get("__categories")
        var_cat = f["var"].get("__categories")
        labels = _decode_h5_categorical(f["obs"][gene_col], obs_cat, gene_col)
        umi = f["obs"][umi_col][:].astype(float)
        gene_names = _decode_h5_categorical(f["var"][gene_name_col], var_cat, gene_name_col)

        measured_first_idx: dict[str, int] = {}
        for i, g in enumerate(gene_names):
            measured_first_idx.setdefault(str(g), i)

        counts = collections.Counter(labels.tolist())
        kept = [g for g, c in counts.items()
                if g != control_label and g in measured_first_idx and c > min_cells]
        kept.sort(key=lambda g: -counts[g])
        if max_perturbations is not None:
            kept = kept[:max_perturbations]
        col_idx = np.array([measured_first_idx[g] for g in kept], dtype=int)
        kept_set = set(kept)

        keep_row = (labels == control_label) | np.isin(labels, list(kept_set))
        median_umi = float(np.median(umi[umi > 0])) if np.any(umi > 0) else 1.0

        n = labels.shape[0]
        out_blocks = []
        X = f["X"]
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            rmask = keep_row[sl]
            if not rmask.any():
                continue
            block = np.asarray(X[sl])[rmask][:, col_idx].astype(float)
            u = umi[sl][rmask].copy()
            u[u <= 0] = 1.0
            out_blocks.append(np.log1p(block / u[:, None] * median_umi).astype(np.float32))
        expr = np.vstack(out_blocks) if out_blocks else np.zeros((0, len(kept)), dtype=np.float32)
        kept_labels = labels[keep_row]
    finally:
        f.close()

    expr_df = pd.DataFrame(expr, columns=kept)
    labels_series = pd.Series(
        [CONTROL_LABEL if l == control_label else l for l in kept_labels]
    )
    ds = load_interventional_frames(
        name, expr_df, labels_series, reference_kind="perturbation_derived"
    )
    ds.metadata["source_file"] = path.name
    ds.metadata["min_cells_per_perturbation"] = min_cells
    return ds
