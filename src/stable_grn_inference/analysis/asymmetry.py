"""Interventional response asymmetry and whitening tools (experiment 29, Direction A).

The one non-circular, genuinely pairwise quantity in the perturbation-response matrix
is the orientation asymmetry: does perturbing g move h more than perturbing h moves g.
It lives in the off-diagonal pair (M[g,h], M[h,g]) of the square response block, not in
the row difference Delta_g - Delta_h. This module operationalizes it and the one untried
operation on the dominant-mode bottleneck: whiten the response (downweight the dominant
mode) rather than subtract it.

Objects:

* :func:`response_asymmetry` - A = |M| - |M|^T, the antisymmetric magnitude-asymmetry
  matrix. ``net_out`` (experiment 26) is its per-gene row mean.
* :func:`net_out` - per-gene cascade position, mean_h(|M[g,h]| - |M[h,g]|).
* :func:`antisymmetric_lift` / :func:`residualize_asymmetry` - remove the part of the
  asymmetry that is explained by per-gene potentials (net_out, magnitude). The residual
  is the asymmetry NOT recoverable from the two reproducible per-gene axes; that residual,
  not the raw asymmetry, is the non-circular target.
* :func:`fractional_whiten` - SVD-based fractional whitening with a single knob alpha in
  [0, 1]. alpha=0 is the raw matrix, alpha=1 is full ZCA (all singular values equalized).
  Shrinkage between the two downweights the dominant mode without deleting it. Note the
  documented failure mode (BaCoN): whitening amplifies low-variance directions, so a
  strength sweep watching reproducibility is required, not a flip to full whitening.
* :func:`pairwise_reproducibility` - rank agreement of an off-diagonal score across two
  independent cell halves. Reproducibility is consistency, not correctness; correctness
  needs the external anchors (Gate 1).
"""

from __future__ import annotations

import numpy as np


def net_out(matrix) -> np.ndarray:
    """Per-gene cascade position net_out(g) = mean_h(|M[g,h]| - |M[h,g]|).

    High net_out is upstream (perturbing g moves others more than they move g). This is
    the experiment-26 definition and equals ``response_asymmetry(matrix).sum(1) / (n-1)``.
    """
    a = np.abs(np.asarray(matrix, dtype=float))
    n = a.shape[0]
    if n < 2:
        return np.zeros(n)
    return (a.sum(axis=1) - a.sum(axis=0)) / (n - 1)


def response_magnitude(matrix) -> np.ndarray:
    """Per-gene response magnitude ||M[g, :]|| (the experiment-26 severity axis)."""
    return np.linalg.norm(np.asarray(matrix, dtype=float), axis=1)


def response_asymmetry(matrix) -> np.ndarray:
    """Antisymmetric magnitude-asymmetry matrix A = |M| - |M|^T.

    A[g,h] > 0 means perturbing g moves h more than perturbing h moves g. A is
    antisymmetric (A = -A^T) with a zero diagonal.
    """
    a = np.abs(np.asarray(matrix, dtype=float))
    return a - a.T


def antisymmetric_lift(node_scores) -> np.ndarray:
    """The rank-<=2 antisymmetric matrix L[g,h] = u[g] - u[h] from a per-gene potential u."""
    u = np.asarray(node_scores, dtype=float)
    return u[:, None] - u[None, :]


def residualize_asymmetry(asymmetry, node_vectors):
    """Remove the best antisymmetric fit built from per-gene potentials.

    Fits ``A ~ sum_k beta_k * (u_k[g] - u_k[h])`` by least squares over all entries and
    returns ``(residual, beta)``. The residual is the pairwise asymmetry that is NOT a
    function of the supplied per-gene axes (typically net_out and magnitude); it is the
    non-circular signal a pairwise method would have to capture.
    """
    A = np.asarray(asymmetry, dtype=float)
    n = A.shape[0]
    lifts = [antisymmetric_lift(u).ravel() for u in node_vectors]
    if not lifts:
        return A.copy(), np.zeros(0)
    X = np.column_stack(lifts)
    y = A.ravel()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = (y - X @ beta).reshape(n, n)
    return residual, beta


def fractional_whiten(matrix, alpha: float, *, tol: float = 1e-9) -> np.ndarray:
    """Fractional ZCA whitening of a square response matrix, one knob alpha in [0, 1].

    SVD ``M = U diag(s) V^T``; replace ``s -> s**(1-alpha)`` (zeros left at zero, so exact
    null directions are not resurrected), then rescale to preserve the Frobenius norm.
    alpha=0 returns M unchanged; alpha=1 equalizes all nonzero singular values (full
    whitening); intermediate alpha downweights the dominant mode by shrinking the gap
    between its singular value and the rest.
    """
    M = np.asarray(matrix, dtype=float)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return M.copy()
    nz = s > tol * s[0]
    s_new = np.zeros_like(s)
    s_new[nz] = s[nz] ** (1.0 - float(alpha))
    norm_old = np.linalg.norm(s[nz])
    norm_new = np.linalg.norm(s_new[nz])
    if norm_new > 0:
        s_new[nz] *= norm_old / norm_new
    return (U * s_new) @ Vt


def pairwise_reproducibility(score_a, score_b) -> float:
    """Spearman rank agreement of two score matrices over their off-diagonal entries.

    Used to measure split-half reproducibility of an edge/asymmetry score. Returns NaN if
    a side has no variance.
    """
    from scipy.stats import spearmanr

    A = np.asarray(score_a, dtype=float)
    B = np.asarray(score_b, dtype=float)
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    x, y = A[off], B[off]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)
