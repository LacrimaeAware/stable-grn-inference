"""Interpretable gene-program discovery and single-cell response heterogeneity (exp 37-38).

The reframe: stop recovering edges; characterize structure. Two deliverables the field values and
that the project's tooling supports:

* Program discovery (exp 37): decompose expression into a few interpretable gene programs and judge
  them by REPRODUCIBILITY (do the same programs recur on independent cells) rather than by beating a
  baseline on edges. cNMF-style consensus uses non-negative factorization; PCA is the linear baseline.
* Response heterogeneity (exp 38): the field models the population-MEAN perturbation response; the open
  problem is the per-cell deviation from that mean (the "ripples under the dominant mode"). This asks
  whether that heterogeneity is structured (low-rank), reproducible (the deviation direction recurs on
  independent cells), and interpretable (aligned with a known program such as the cell-cycle cascade).

* :func:`discover_programs` - NMF or PCA programs (W samples x k, H k x genes).
* :func:`match_programs` / :func:`program_reproducibility` - recurrence of programs across a split.
* :func:`residual_heterogeneity` - per-cell deviation from a group mean.
* :func:`heterogeneity_structure` - low-rankness, split-half reproducibility, and alignment of the
  dominant deviation direction with a reference program.
"""

from __future__ import annotations

import numpy as np


def discover_programs(samples, k: int, *, method: str = "nmf", seed: int = 0):
    """Decompose samples x features into k programs. Returns (W samples x k, H k x features).

    ``nmf`` is non-negative (interpretable additive programs; features are clipped at 0); ``pca`` is
    the linear baseline.
    """
    X = np.asarray(samples, dtype=float)
    if method == "nmf":
        from sklearn.decomposition import NMF

        model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=600)
        W = model.fit_transform(np.maximum(X, 0.0))
        H = model.components_
    elif method == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=k, random_state=seed)
        W = model.fit_transform(X - X.mean(0))
        H = model.components_
    else:
        raise ValueError(f"unknown method {method!r}")
    return W, H


def _unit_rows(M):
    M = np.asarray(M, dtype=float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def match_programs(H1, H2):
    """Match program sets by absolute cosine (Hungarian); return (mean matched cosine, per-program)."""
    from scipy.optimize import linear_sum_assignment

    A, B = _unit_rows(H1), _unit_rows(H2)
    S = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-S)
    matched = S[r, c]
    return float(matched.mean()), matched


def program_reproducibility(samples, k: int, *, method: str = "nmf", seed: int = 0):
    """Split samples in half, discover programs on each, and report how well the programs recur."""
    X = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    h = len(idx) // 2
    _, H1 = discover_programs(X[idx[:h]], k, method=method, seed=seed)
    _, H2 = discover_programs(X[idx[h:2 * h]], k, method=method, seed=seed)
    return match_programs(H1, H2)


def residual_heterogeneity(group_samples):
    """Per-sample deviation from the group mean (the structured part of single-cell variability)."""
    X = np.asarray(group_samples, dtype=float)
    return X - X.mean(axis=0, keepdims=True)


def heterogeneity_structure(group_samples, *, reference_program=None, seed: int = 0) -> dict:
    """Characterize the per-cell deviation from a group's mean response.

    Returns: ``top_var_fraction`` (how low-rank the deviation is; >> isotropic if structured),
    ``reproducibility`` (cosine of the dominant deviation direction across two halves of the cells),
    and ``reference_alignment`` (|cosine| of the dominant deviation with a supplied reference program,
    e.g. the cell-cycle cascade axis), if given.
    """
    R = residual_heterogeneity(group_samples)
    n = R.shape[0]
    out: dict = {"n_cells": int(n)}
    if n < 4:
        return {**out, "top_var_fraction": float("nan"), "reproducibility": float("nan"),
                "reference_alignment": float("nan")}
    sv = np.linalg.svd(R, compute_uv=False)
    out["top_var_fraction"] = float(sv[0] ** 2 / (sv ** 2).sum()) if sv.sum() > 0 else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    h = n // 2
    v1 = np.linalg.svd(R[idx[:h]], full_matrices=False)[2][0]
    v2 = np.linalg.svd(R[idx[h:2 * h]], full_matrices=False)[2][0]
    out["reproducibility"] = float(abs(np.dot(v1, v2)))

    if reference_program is not None:
        v = np.linalg.svd(R, full_matrices=False)[2][0]
        ref = np.asarray(reference_program, dtype=float)
        out["reference_alignment"] = float(abs(np.dot(v, ref) / ((np.linalg.norm(v) * np.linalg.norm(ref)) + 1e-12)))
    else:
        out["reference_alignment"] = float("nan")
    return out
