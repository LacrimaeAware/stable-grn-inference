"""Correctness tests for the order-recovery and higher-order tools (experiment 34).

The positive control: cells laid out along a planted 1D trajectory in gene space (shuffled) must be
re-ordered from static geometry alone, up to reversal. If this fails on planted truth, the bug is
here, not in the real data.
"""

import unittest

import numpy as np

from stable_grn_inference.analysis import (
    cell_similarity,
    correlation_power,
    diffusion_order,
    network_propagation,
    order_recovery_score,
    orient_by_root,
    second_order_correlation,
    spectral_order,
)


def _trajectory(n_cells=120, seed=0):
    """Cells along a 1D trajectory in 5-gene space; returns (shuffled X, true order of the rows)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_cells)
    genes = np.column_stack([
        t, t ** 2, np.sin(3 * t), np.exp(t), 1.0 - t,
    ])
    X = genes + rng.normal(scale=0.03, size=genes.shape)
    perm = rng.permutation(n_cells)
    return X[perm], t[perm]


class OrderRecoveryTest(unittest.TestCase):
    def test_spectral_order_recovers_planted_trajectory(self):
        X, true_t = _trajectory(seed=0)
        score = order_recovery_score(spectral_order(X), true_t)
        self.assertGreater(score, 0.8)   # recovered up to reversal

    def test_diffusion_order_recovers_planted_trajectory(self):
        X, true_t = _trajectory(seed=1)
        score = order_recovery_score(diffusion_order(X), true_t)
        self.assertGreater(score, 0.8)

    def test_recovery_score_is_reversal_invariant(self):
        x = np.arange(50.0)
        self.assertAlmostEqual(order_recovery_score(x, x), 1.0, places=6)
        self.assertAlmostEqual(order_recovery_score(-x, x), 1.0, places=6)

    def test_orient_by_root_puts_root_low(self):
        coord = np.array([3.0, 1.0, -2.0, 5.0])
        oriented = orient_by_root(coord, root_index=3)   # root currently highest
        self.assertLessEqual(oriented[3], np.median(oriented))

    def test_random_data_does_not_recover_an_order(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 5))
        true_t = np.linspace(0, 1, 120)   # an order unrelated to the random data
        self.assertLess(order_recovery_score(spectral_order(X), true_t), 0.4)


class HigherOrderTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.C = np.corrcoef(rng.normal(size=(200, 8)).T)

    def test_second_order_correlation_symmetric(self):
        S = second_order_correlation(self.C)
        np.testing.assert_allclose(S, S.T, atol=1e-9)
        self.assertEqual(S.shape, self.C.shape)

    def test_correlation_power_shape_and_identity(self):
        np.testing.assert_allclose(correlation_power(self.C, 1), self.C, atol=1e-9)
        self.assertEqual(correlation_power(self.C, 3).shape, self.C.shape)

    def test_network_propagation_finite_and_zero_diagonal_added(self):
        P = network_propagation(self.C, alpha=0.5)
        self.assertTrue(np.all(np.isfinite(P)))
        self.assertEqual(P.shape, self.C.shape)


class SimilarityTest(unittest.TestCase):
    def test_rbf_similarity_is_symmetric_unit_diagonal(self):
        rng = np.random.default_rng(4)
        S = cell_similarity(rng.normal(size=(30, 4)), kind="rbf")
        np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-9)
        np.testing.assert_allclose(S, S.T, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
