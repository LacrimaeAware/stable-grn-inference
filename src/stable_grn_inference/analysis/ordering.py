"""Recover a latent order from static data, and capture indirect structure (experiment 34).

The idea this implements: from static, unordered data you can rebuild a one-dimensional order
(a chain) from the geometry of a similarity matrix and its higher powers. The order is recovered
up to reversal (direction is symmetric, order is not); a prior (a root) fixes the direction. And
indirect, multi-step relationships (A to C to D) can be captured by iterating the correlation
matrix (powers / diffusion) rather than reading single pairwise correlations.

This is spectral seriation (Fiedler vector of the Laplacian; Atkins-Boman-Hendrickson) and
diffusion pseudotime (Haghverdi; Palantir), plus higher-order correlation, in one place.

Order recovery:
* :func:`cell_similarity` - similarity between samples (rows), RBF or absolute correlation.
* :func:`spectral_order` - the Fiedler vector (second eigenvector of the normalized Laplacian);
  its sort order is the recovered 1D order, reversal-ambiguous.
* :func:`diffusion_order` - the first non-trivial diffusion component (a powered transition
  operator); the diffusion-pseudotime ordering, also reversal-ambiguous.
* :func:`orient_by_root` - fix the reversal using a root index (the supplied prior).
* :func:`order_recovery_score` - absolute Spearman of a recovered order against a known order
  (absolute, because order is recovered up to reversal).

Indirect / higher-order structure:
* :func:`second_order_correlation` - correlation of correlation profiles (genes related if they
  relate to the same other genes).
* :func:`correlation_power` - the k-step matrix C^k (A to C to D chains).
* :func:`network_propagation` - von Neumann diffusion (I - alpha C)^-1 - I (all chain lengths).
"""

from __future__ import annotations

import numpy as np


def cell_similarity(samples, *, kind: str = "rbf", sigma: float | None = None) -> np.ndarray:
    """Similarity between samples (rows of ``samples``). ``rbf`` Gaussian kernel or absolute
    sample-to-sample correlation. For RBF, ``sigma`` defaults to the median pairwise distance."""
    X = np.asarray(samples, dtype=float)
    if kind == "corr":
        S = np.corrcoef(X)
        return np.nan_to_num(np.abs(np.atleast_2d(S)))
    diff = X[:, None, :] - X[None, :, :]
    d2 = (diff ** 2).sum(axis=2)
    if sigma is None:
        off = np.sqrt(d2[~np.eye(len(X), dtype=bool)])
        sigma = float(np.median(off[off > 0])) if np.any(off > 0) else 1.0
    return np.exp(-d2 / (2.0 * sigma ** 2 + 1e-12))


def _symmetric_normalized(S):
    d = np.asarray(S, dtype=float).sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return dinv[:, None] * S * dinv[None, :], dinv


def spectral_order(samples, *, kind: str = "rbf", sigma: float | None = None) -> np.ndarray:
    """Fiedler-vector coordinate per sample (second eigenvector of the normalized Laplacian).

    Sorting samples by this coordinate gives the recovered 1D order, up to reversal.
    """
    S = cell_similarity(samples, kind=kind, sigma=sigma)
    norm, _ = _symmetric_normalized(S)
    L = np.eye(S.shape[0]) - norm
    w, V = np.linalg.eigh(L)
    return V[:, 1]  # smallest is trivial; the next is the Fiedler vector


def diffusion_order(samples, *, kind: str = "rbf", sigma: float | None = None, t: int = 1) -> np.ndarray:
    """First non-trivial diffusion component (diffusion-pseudotime coordinate), up to reversal.

    Uses the symmetric conjugate of the random-walk transition operator; ``t`` powers the
    diffusion (larger t emphasizes the slow, global coordinate).
    """
    S = cell_similarity(samples, kind=kind, sigma=sigma)
    norm, dinv = _symmetric_normalized(S)
    w, V = np.linalg.eigh(norm)
    order = np.argsort(w)[::-1]  # largest eigenvalue first (trivial), then the slow modes
    idx = order[1]
    return dinv * V[:, idx] * (float(w[idx]) ** t)


def orient_by_root(coord, root_index: int) -> np.ndarray:
    """Flip a reversal-ambiguous coordinate so the root sample is at the low end (the prior step)."""
    coord = np.asarray(coord, dtype=float).copy()
    if coord[root_index] > np.median(coord):
        coord = -coord
    return coord


def order_recovery_score(recovered, true_order) -> float:
    """Absolute Spearman of a recovered order against a known order (absolute: order up to reversal)."""
    from scipy.stats import spearmanr

    r = spearmanr(np.asarray(recovered, dtype=float), np.asarray(true_order, dtype=float)).statistic
    return float(abs(r)) if np.isfinite(r) else float("nan")


def second_order_correlation(corr_matrix) -> np.ndarray:
    """Correlation of correlation profiles: two variables relate if they relate to the same others."""
    C = np.asarray(corr_matrix, dtype=float)
    S = np.corrcoef(C)
    return np.nan_to_num(np.atleast_2d(S))


def correlation_power(corr_matrix, k: int) -> np.ndarray:
    """The k-step matrix C^k, capturing length-k chains A to ... to D."""
    return np.linalg.matrix_power(np.asarray(corr_matrix, dtype=float), int(k))


def network_propagation(corr_matrix, alpha: float = 0.5) -> np.ndarray:
    """Von Neumann diffusion (I - alpha * C_norm)^-1 - I: all chain lengths, geometrically weighted.

    C is rescaled by its spectral radius so the inverse converges for alpha in (0, 1).
    """
    C = np.asarray(corr_matrix, dtype=float)
    n = C.shape[0]
    sr = float(np.max(np.abs(np.linalg.eigvals(C))))
    Cn = C / (sr + 1e-9)
    return np.linalg.inv(np.eye(n) - float(alpha) * Cn) - np.eye(n)
