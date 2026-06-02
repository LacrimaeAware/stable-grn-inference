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
