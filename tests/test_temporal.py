"""Correctness tests for the dynamical-recovery tools (experiment 30, Direction B).

These pin the claim Direction B rests on: the dynamic-mode operator recovers a known linear
operator from snapshot pairs, it recovers DIRECTED structure where the static symmetric score
cannot, and the dominant input mode does not bias the operator estimate. If the dynamic
operator did not beat the static baseline on clean synthetic data, the bug would be here, not
in the real data.
"""

import unittest

import numpy as np

from stable_grn_inference.dynamics import (
    dmd_edges,
    dmd_operator,
    edges_to_operator,
    make_dynamical_system,
    skeleton_recovery_aupr,
    specific_recovery_aupr,
    static_correlation_edges,
)


class DmdOperatorTest(unittest.TestCase):
    def test_recovers_a_known_operator_on_clean_data(self):
        rng = np.random.default_rng(0)
        g = 6
        A = 0.3 * rng.standard_normal((g, g))
        # make it stable
        A *= 0.6 / max(1e-9, np.max(np.abs(np.linalg.eigvals(A))))
        x = np.zeros(g)
        states = [x]
        for _ in range(4000):
            x = A @ x + 0.1 * rng.standard_normal(g)
            states.append(x)
        states = np.array(states)
        A_hat = dmd_operator(states[:-1], states[1:], ridge=0.0)
        self.assertLess(np.linalg.norm(A_hat - A) / np.linalg.norm(A), 0.1)

    def test_dmd_edges_zero_diagonal(self):
        A = np.arange(9.0).reshape(3, 3)
        score = dmd_edges(A)
        np.testing.assert_allclose(np.diag(score), 0.0)


class StaticScoreTest(unittest.TestCase):
    def test_static_correlation_is_symmetric(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 5))
        C = static_correlation_edges(X)
        np.testing.assert_allclose(C, C.T, atol=1e-12)


class EdgesToOperatorTest(unittest.TestCase):
    def test_orientation_matches_operator_convention(self):
        import pandas as pd

        genes = ["G1", "G2", "G3"]
        edges = pd.DataFrame({"source": ["G1"], "target": ["G2"], "is_true": [1]})
        op = edges_to_operator(edges, genes)
        # edge G1 -> G2 means G2(t+1) depends on G1(t): op[target, source] = op[1, 0]
        self.assertEqual(op[1, 0], 1.0)
        self.assertEqual(op.sum(), 1.0)


class SystemAndContrastTest(unittest.TestCase):
    def test_mode_strength_raises_dominant_fraction(self):
        low = make_dynamical_system(mode_strength=0.0, n_steps=1500, seed=0)
        high = make_dynamical_system(mode_strength=8.0, n_steps=1500, seed=0)
        self.assertGreater(high.realized_mode_fraction, low.realized_mode_fraction)

    def test_system_is_stable(self):
        sys = make_dynamical_system(decay=0.6, coupling=0.3, n_steps=800, seed=0)
        # W is nilpotent (acyclic), so every eigenvalue of A equals decay
        self.assertLess(np.max(np.abs(np.linalg.eigvals(sys.A))), 0.999)
        self.assertTrue(np.all(np.isfinite(sys.X1)))

    def test_time_axis_beats_static_for_direction(self):
        """The positive control: on a clean system the dynamic operator recovers DIRECTED
        structure well above the static symmetric score."""
        sys = make_dynamical_system(
            n_genes=20, density=0.12, coupling=0.22, decay=0.6,
            mode_strength=3.0, noise=0.2, n_steps=4000, seed=0,
        )
        dmd_score = dmd_edges(dmd_operator(sys.X1, sys.X2, ridge=1e-3))
        static_score = static_correlation_edges(sys.X1)
        dmd_dir = specific_recovery_aupr(dmd_score, sys.true_W)
        static_dir = specific_recovery_aupr(static_score, sys.true_W)
        self.assertGreater(dmd_dir, sys.true_edge_density)   # above chance
        self.assertGreater(dmd_dir, static_dir + 0.1)        # and clearly beats static for direction

    def test_static_finds_skeleton_even_when_it_misses_direction(self):
        sys = make_dynamical_system(
            n_genes=20, density=0.12, coupling=0.22, decay=0.6,
            mode_strength=2.0, noise=0.2, n_steps=4000, seed=1,
        )
        static_score = static_correlation_edges(sys.X1)
        skel = skeleton_recovery_aupr(static_score, sys.true_W)
        self.assertGreater(skel, sys.true_edge_density)      # skeleton is found


if __name__ == "__main__":
    unittest.main()
