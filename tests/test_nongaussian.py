"""Correctness tests for non-Gaussian orientation and detectability (experiment 35).

The positive control: a planted causal chain with non-Gaussian noise must be ORIENTED from static
data (no time), because the arrow lives in the higher moments. If this fails on planted truth, the
method or the sign convention is wrong, caught here.
"""

import unittest

import numpy as np

from stable_grn_inference.analysis import (
    edge_detectability,
    nongaussian_directed_edges,
    nongaussianity,
    pairwise_orientation,
)


def _nongaussian_chain(n=4000, seed=0, coef=0.9):
    """Linear non-Gaussian chain x0 -> x1 -> x2 -> x3 (Laplace noise, super-Gaussian)."""
    rng = np.random.default_rng(seed)
    e = rng.laplace(size=(n, 4))
    x0 = e[:, 0]
    x1 = coef * x0 + e[:, 1]
    x2 = coef * x1 + e[:, 2]
    x3 = coef * x2 + e[:, 3]
    return np.column_stack([x0, x1, x2, x3])


class OrientationTest(unittest.TestCase):
    def test_pairwise_orientation_is_antisymmetric(self):
        M = pairwise_orientation(_nongaussian_chain())
        np.testing.assert_allclose(M, -M.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-12)

    def test_orients_planted_chain_from_static(self):
        X = _nongaussian_chain(seed=0)
        M = pairwise_orientation(X)
        true_edges = [(0, 1), (1, 2), (2, 3)]
        correct = sum(1 for i, j in true_edges if M[i, j] > 0)
        self.assertGreaterEqual(correct, 2)  # majority of arrows correct (cause -> effect)

    def test_directed_score_beats_symmetric_at_orientation(self):
        X = _nongaussian_chain(seed=1)
        true = {(0, 1), (1, 2), (2, 3)}
        directed = nongaussian_directed_edges(X)
        C = np.abs(np.corrcoef(X.T))
        n = X.shape[1]
        # directed score should put more mass on true (i->j) than on the reverse (j->i)
        fwd = sum(directed[i, j] for i, j in true)
        rev = sum(directed[j, i] for i, j in true)
        self.assertGreater(fwd, rev)

    def test_gaussian_noise_has_low_nongaussianity(self):
        rng = np.random.default_rng(2)
        gauss = rng.normal(size=(5000, 3))
        laplace = rng.laplace(size=(5000, 3))
        self.assertLess(nongaussianity(gauss).mean(), 0.3)
        self.assertGreater(nongaussianity(laplace).mean(), 1.0)


class DetectabilityTest(unittest.TestCase):
    def test_true_edges_are_more_detectable_than_null(self):
        X = _nongaussian_chain(seed=3)
        z = edge_detectability(X, n_perm=100, seed=0)
        true = [(0, 1), (1, 2), (2, 3)]
        true_z = np.mean([z[i, j] for i, j in true])
        # a far-apart, near-independent pair (0,3) should be much less detectable
        self.assertGreater(true_z, z[0, 3])
        self.assertGreater(true_z, 3.0)   # adjacent edges clearly leave the null


if __name__ == "__main__":
    unittest.main()
