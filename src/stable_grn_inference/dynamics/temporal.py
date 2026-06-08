"""Directed-structure recovery when the data has a time axis (experiment 30, Direction B).

Experiment 28 mapped where specific structure is recoverable from a STATIC snapshot and
placed RPE1 in the unrecoverable corner (high dominant-mode fraction, low specific-SNR). The
missing ingredient is a time axis: a static, symmetric statistic (correlation, covariance)
cannot orient an edge at all, and a dominant shared mode swamps what little it sees. A
dynamical operator estimated from consecutive states can, in principle, recover the directed
operator regardless of how dominant the shared input mode is, because the least-squares /
DMD estimate of A in ``x_{t+1} = A x_t + noise`` is unbiased by input covariance.

This module provides:

* :func:`make_dynamical_system` - a linear stochastic system (VAR(1)) driven by a sparse
  directed operator W (the truth), with a tunable dominant shared input mode and noise. The
  operator's off-diagonal IS the truth, so recovery is gradeable.
* :func:`dmd_operator` - the dynamic-mode / least-squares operator estimate A_hat from
  snapshot pairs (X1 = states at t, X2 = states at t+1).
* :func:`dmd_edges` / :func:`static_correlation_edges` - directed score from A_hat vs the
  symmetric static score that ignores time order (the comparator that cannot orient).
* :func:`edges_to_operator` - build a ground-truth operator matrix from a directed edge list
  (for grading on DREAM4 time-series, which has a known network).
* :func:`dynamical_recovery_grid` - sweep dominant-mode strength and noise; report directed
  and skeleton recovery for the dynamic operator vs the static comparator.

Grading reuses ``specific_recovery_aupr`` and ``normalized_recovery`` from
``separability`` so the two experiments share one recovery metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.interventional import make_sparse_dag
from .separability import normalized_recovery, specific_recovery_aupr

DYNAMICAL_METHODS = ("dmd", "static")


@dataclass
class DynamicalSystem:
    """A linear stochastic system with a known directed operator and a dominant input mode."""

    A: np.ndarray  # the full operator, x_{t+1} = A x_t + noise
    true_W: np.ndarray  # the sparse directed truth (== A off-diagonal here)
    X1: np.ndarray  # samples x genes, states at t
    X2: np.ndarray  # samples x genes, states at t+1
    n_genes: int
    density: float
    decay: float
    mode_strength: float
    noise: float
    realized_mode_fraction: float  # top-1 variance fraction of the stationary covariance
    seed: int

    @property
    def true_edge_mask(self) -> np.ndarray:
        mask = self.true_W != 0.0
        np.fill_diagonal(mask, False)
        return mask

    @property
    def true_edge_density(self) -> float:
        n = self.n_genes
        off = n * (n - 1)
        return float(self.true_edge_mask.sum()) / off if off else float("nan")


def make_dynamical_system(
    *,
    n_genes: int = 30,
    density: float = 0.06,
    coupling: float = 0.18,
    decay: float = 0.6,
    mode_strength: float = 2.0,
    noise: float = 0.3,
    n_steps: int = 2000,
    burn_in: int = 200,
    seed: int = 0,
) -> DynamicalSystem:
    """Simulate ``x_{t+1} = A x_t + eps_t`` with a sparse directed operator and a dominant mode.

    ``A = decay * I + W`` where ``W`` is a sparse upper-triangular (acyclic) operator: because
    ``W`` is nilpotent, every eigenvalue of ``A`` equals ``decay`` (< 1), so the system is
    stable for any coupling. The off-diagonal of ``A`` is exactly ``W``, so ``W`` is the
    directed truth to recover. The driving noise is
    ``eps_t = noise * N(0, I) + mode_strength * z_t * p`` with ``z_t`` a scalar shared shock and
    ``p`` a fixed shared loading: ``mode_strength`` makes one direction dominate the stationary
    covariance (the cascade analog) WITHOUT changing the operator, so it tests whether a
    dynamical estimate sees through a dominant mode that a static covariance cannot.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    rng = np.random.default_rng(seed)
    W = make_sparse_dag(n_genes, density, weight_scale=coupling, seed=seed, dag=True)
    A = decay * np.eye(n_genes) + W
    p = rng.standard_normal(n_genes)
    p = p / (np.linalg.norm(p) or 1.0)

    total = n_steps + burn_in
    X = np.zeros((total + 1, n_genes))
    for t in range(total):
        z = rng.standard_normal()
        eps = noise * rng.standard_normal(n_genes) + mode_strength * z * p
        X[t + 1] = A @ X[t] + eps
    states = X[burn_in:]
    X1, X2 = states[:-1], states[1:]

    cov = np.cov(X1, rowvar=False)
    sv = np.linalg.svd(np.atleast_2d(cov), compute_uv=False)
    realized = float(sv[0] / sv.sum()) if sv.sum() > 0 else float("nan")

    return DynamicalSystem(
        A=A, true_W=W, X1=X1, X2=X2, n_genes=n_genes, density=density, decay=decay,
        mode_strength=float(mode_strength), noise=float(noise),
        realized_mode_fraction=realized, seed=seed,
    )


def dmd_operator(X1, X2, *, ridge: float = 0.0) -> np.ndarray:
    """Least-squares / dynamic-mode operator A_hat with ``X2 ~ X1 @ A_hat.T`` (rows = samples).

    Returns the genes x genes operator ``A_hat[j, k]`` = estimated effect of gene k at t on
    gene j at t+1. ``ridge`` adds a small Tikhonov term to stabilize the normal equations when
    a dominant mode makes ``X1`` ill-conditioned.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    g = X1.shape[1]
    gram = X1.T @ X1 + ridge * np.eye(g)
    b = np.linalg.solve(gram, X1.T @ X2)  # g x g, approximates A.T
    return b.T


def dmd_edges(A_hat) -> np.ndarray:
    """Directed off-diagonal edge score |A_hat| (diagonal zeroed)."""
    score = np.abs(np.asarray(A_hat, dtype=float)).copy()
    np.fill_diagonal(score, 0.0)
    return score


def static_correlation_edges(X) -> np.ndarray:
    """Symmetric static edge score |corr(X)| over genes, ignoring time order (cannot orient)."""
    X = np.asarray(X, dtype=float)
    C = np.corrcoef(X, rowvar=False)
    C = np.abs(np.atleast_2d(C))
    np.fill_diagonal(C, 0.0)
    return np.nan_to_num(C)


def pseudotime_ordered_pairs(expression, pseudotime):
    """Snapshot pairs (X1, X2) by ordering cells along pseudotime, within each trajectory.

    ``expression`` is cells x genes, ``pseudotime`` is cells x trajectories (one or more
    pseudotime columns, e.g. branches). Within each pseudotime column, cells are sorted and
    consecutive cells become a (state at t, state at t+1) pair; pairs are pooled across columns
    so no pair crosses a branch. This turns a pseudotemporally ordered single-cell snapshot into
    the snapshot pairs a dynamic operator estimate needs. Returns ``(X1, X2)`` arrays.
    """
    expr = expression
    pt = pseudotime
    n_genes = expr.shape[1]
    x1_blocks, x2_blocks = [], []
    for col in pt.columns:
        order = pt[col].dropna().sort_values()
        cells = [c for c in order.index if c in expr.index]
        if len(cells) < 2:
            continue
        M = expr.loc[cells].to_numpy(dtype=float)
        x1_blocks.append(M[:-1])
        x2_blocks.append(M[1:])
    if not x1_blocks:
        return np.zeros((0, n_genes)), np.zeros((0, n_genes))
    return np.vstack(x1_blocks), np.vstack(x2_blocks)


def edges_to_operator(edges: pd.DataFrame, genes) -> np.ndarray:
    """Build a ground-truth operator matrix from a directed edge list.

    For an edge source -> target (source regulates target), the operator entry is
    ``op[target, source]`` = 1, matching :func:`dmd_operator`'s convention that ``A_hat[j, k]``
    is the effect of gene k on gene j. Only rows with ``is_true`` != 0 are used when present.
    """
    genes = list(genes)
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    op = np.zeros((n, n))
    has_label = "is_true" in edges.columns
    for _, row in edges.iterrows():
        if has_label and int(row["is_true"]) == 0:
            continue
        s, t = str(row["source"]), str(row["target"])
        if s in idx and t in idx:
            op[idx[t], idx[s]] = 1.0
    np.fill_diagonal(op, 0.0)
    return op


def skeleton_recovery_aupr(score, true_W) -> float:
    """AUPR for the UNDIRECTED skeleton: symmetrize both truth and score, then score off-diagonal.

    A symmetric static method can match the dynamic operator here while failing the directed
    AUPR, isolating the part of recovery that needs the time axis (direction).
    """
    from sklearn.metrics import average_precision_score

    true_W = np.asarray(true_W, dtype=float)
    score = np.asarray(score, dtype=float)
    n = true_W.shape[0]
    off = ~np.eye(n, dtype=bool)
    sym_true = ((np.abs(true_W) + np.abs(true_W.T)) > 0).astype(int)
    sym_score = np.abs(score) + np.abs(score).T
    y_true, y_score = sym_true[off], sym_score[off]
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def dynamical_recovery_grid(
    mode_strength_values,
    noise_values,
    *,
    n_genes: int = 30,
    density: float = 0.08,
    coupling: float = 0.2,
    decay: float = 0.6,
    n_steps: int = 2000,
    n_seeds: int = 3,
    base_seed: int = 0,
    ridge: float = 1e-3,
) -> pd.DataFrame:
    """Sweep dominant-mode strength and noise; directed and skeleton recovery for dmd vs static.

    For each cell, average over seeds the chance-normalized directed recovery (and the raw
    directed / skeleton AUPR) of the dynamic operator (uses time order) and the static
    correlation (ignores it). The contrast is the experiment: the time axis should keep
    directed recovery above the static floor as the dominant mode grows.
    """
    rows: list[dict[str, object]] = []
    for ms in mode_strength_values:
        for noise in noise_values:
            systems = [
                make_dynamical_system(
                    n_genes=n_genes, density=density, coupling=coupling, decay=decay,
                    mode_strength=float(ms), noise=float(noise), n_steps=n_steps,
                    seed=base_seed + s,
                )
                for s in range(n_seeds)
            ]
            chance = float(np.mean([s.true_edge_density for s in systems]))
            realized_mode = float(np.mean([s.realized_mode_fraction for s in systems]))
            scores = {
                "dmd": [dmd_edges(dmd_operator(s.X1, s.X2, ridge=ridge)) for s in systems],
                "static": [static_correlation_edges(s.X1) for s in systems],
            }
            for method in DYNAMICAL_METHODS:
                directed = [specific_recovery_aupr(sc, s.true_W)
                            for sc, s in zip(scores[method], systems)]
                skeleton = [skeleton_recovery_aupr(sc, s.true_W)
                            for sc, s in zip(scores[method], systems)]
                directed = [a for a in directed if np.isfinite(a)]
                skeleton = [a for a in skeleton if np.isfinite(a)]
                mean_dir = float(np.mean(directed)) if directed else float("nan")
                mean_skel = float(np.mean(skeleton)) if skeleton else float("nan")
                rows.append({
                    "mode_strength": float(ms),
                    "noise": float(noise),
                    "method": method,
                    "realized_mode_fraction": realized_mode,
                    "true_edge_density": chance,
                    "directed_aupr": mean_dir,
                    "directed_normalized": normalized_recovery(mean_dir, chance),
                    "skeleton_aupr": mean_skel,
                    "n_seeds": len(directed),
                })
    return pd.DataFrame(rows)
