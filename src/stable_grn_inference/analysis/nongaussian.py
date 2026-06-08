"""Direction from static data via non-Gaussianity, and per-edge detectability (experiment 35).

Correlation is a second-order, symmetric statistic: it gives the skeleton (who relates to whom) but
cannot orient an edge. The arrow lives in the higher moments. When the noise is non-Gaussian (gene
expression is), the causal direction IS identifiable from static observational data (the LiNGAM
result, Shimizu et al. 2006). This module implements a pairwise non-Gaussian orientation measure and
turns it into a directed edge score, plus a detectability map that asks, per edge, how far its
statistic sits from a permutation null (the signal-is-findable-if-it-leaves-the-null idea).

Caveats inherited from LiNGAM: it assumes acyclicity and non-Gaussianity, and degrades under
feedback cycles, latent confounders, and Gaussian technical noise. So expect it to orient acyclic
structure and to struggle on cyclic networks.

* :func:`pairwise_orientation` - antisymmetric matrix M with M[i,j] > 0 favoring i -> j, from the
  tanh likelihood-ratio measure (Hyvarinen-Smith).
* :func:`nongaussian_directed_edges` - the correlation skeleton mass placed on the LiNGAM-chosen
  direction (a directed edge score).
* :func:`edge_detectability` - per-edge z-score of |correlation| against a permutation null.
* :func:`nongaussianity` - per-variable absolute excess kurtosis (0 for Gaussian); reports whether
  the LiNGAM assumption is even met.
"""

from __future__ import annotations

import numpy as np


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def pairwise_orientation(samples) -> np.ndarray:
    """Antisymmetric orientation matrix M (M[i,j] > 0 favors i -> j) from a non-Gaussian measure.

    ``samples`` is observations x variables. For standardized x_i, x_j with correlation rho, the
    tanh likelihood-ratio statistic R = rho * (E[x_i tanh(x_j)] - E[tanh(x_i) x_j]) is positive when
    x_i is the more plausible cause. M is antisymmetric (M[i,j] = -M[j,i]).
    """
    Xs = _standardize(samples)
    g = np.tanh(Xs)
    n = Xs.shape[1]
    # E[x_i tanh(x_j)] over samples -> matrix A[i,j]
    A = (Xs.T @ g) / Xs.shape[0]          # A[i,j] = E[x_i tanh(x_j)]
    C = (Xs.T @ Xs) / Xs.shape[0]         # correlation (standardized)
    M = C * (A - A.T)                      # rho_ij * (E[x_i g(x_j)] - E[g(x_i) x_j])
    np.fill_diagonal(M, 0.0)
    return M


def nongaussian_directed_edges(samples) -> np.ndarray:
    """Directed edge score: the |correlation| skeleton placed on the LiNGAM-chosen direction.

    score[i,j] = |corr(i,j)| if the orientation favors i -> j, else 0. A symmetric-correlation
    baseline (the same |corr| on both directions) cannot beat this at directed recovery if the
    orientation is right.
    """
    C = np.abs(np.nan_to_num(np.corrcoef(np.asarray(samples, dtype=float).T)))
    M = pairwise_orientation(samples)
    score = np.where(M > 0, C, 0.0)
    np.fill_diagonal(score, 0.0)
    return score


def edge_detectability(samples, *, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Per-edge z-score of |correlation| against a per-variable permutation null.

    Each variable is independently shuffled to break all dependence; the null distribution of
    |correlation| gives a mean and spread per edge. A high z means the edge's correlation sits far
    from what randomness produces, i.e. it is detectable; a z near 0 is indistinguishable from noise.
    """
    X = np.asarray(samples, dtype=float)
    n_vars = X.shape[1]
    observed = np.abs(np.nan_to_num(np.corrcoef(X.T)))
    rng = np.random.default_rng(seed)
    acc = np.zeros((n_vars, n_vars))
    acc2 = np.zeros((n_vars, n_vars))
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, k]) for k in range(n_vars)])
        c = np.abs(np.nan_to_num(np.corrcoef(Xp.T)))
        acc += c
        acc2 += c ** 2
    mu = acc / n_perm
    var = np.maximum(acc2 / n_perm - mu ** 2, 0.0)
    z = (observed - mu) / (np.sqrt(var) + 1e-12)
    np.fill_diagonal(z, 0.0)
    return z


def nongaussianity(samples) -> np.ndarray:
    """Per-variable absolute excess kurtosis (0 for Gaussian); whether the LiNGAM assumption holds."""
    from scipy.stats import kurtosis

    return np.abs(kurtosis(np.asarray(samples, dtype=float), axis=0, fisher=True))
