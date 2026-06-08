"""Correctness tests for the response-asymmetry and whitening tools (experiment 29).

These pin the math the Gate-0 verdict rests on: net_out matches the experiment-26 formula,
the asymmetry is antisymmetric, residualization removes exactly the per-gene-potential part
(so the residual is genuinely the non-circular signal), fractional whitening interpolates
from identity to full whitening, and the pipeline detects reproducible asymmetry when it is
planted (synthetic DAG) so a real null is a property of the data, not a bug here.
"""

import unittest

import numpy as np

from stable_grn_inference.analysis import (
    antisymmetric_lift,
    fractional_whiten,
    net_out,
    pairwise_reproducibility,
    residualize_asymmetry,
    response_asymmetry,
    response_magnitude,
)
from stable_grn_inference.data import (
    load_interventional_frames,
    make_synthetic_interventional,
    perturbation_response_matrix,
)


class AsymmetryMathTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.M = rng.normal(size=(12, 12))

    def test_net_out_matches_manual_formula(self):
        A = np.abs(self.M)
        n = A.shape[0]
        expected = (A.sum(1) - A.sum(0)) / (n - 1)
        np.testing.assert_allclose(net_out(self.M), expected)

    def test_net_out_is_row_mean_of_asymmetry(self):
        A = response_asymmetry(self.M)
        n = A.shape[0]
        np.testing.assert_allclose(A.sum(1) / (n - 1), net_out(self.M), atol=1e-12)

    def test_response_asymmetry_is_antisymmetric(self):
        A = response_asymmetry(self.M)
        np.testing.assert_allclose(A, -A.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(A), 0.0, atol=1e-12)

    def test_antisymmetric_lift_shape_and_values(self):
        u = np.array([1.0, 4.0, 9.0])
        L = antisymmetric_lift(u)
        self.assertEqual(L.shape, (3, 3))
        self.assertAlmostEqual(L[0, 1], 1.0 - 4.0)
        np.testing.assert_allclose(L, -L.T)

    def test_residualize_removes_a_pure_potential_exactly(self):
        u = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
        A = 1.7 * antisymmetric_lift(u)
        residual, beta = residualize_asymmetry(A, [u])
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)
        self.assertAlmostEqual(float(beta[0]), 1.7, places=6)

    def test_residual_is_orthogonal_to_the_removed_potentials(self):
        rng = np.random.default_rng(1)
        u = rng.normal(size=8)
        extra = response_asymmetry(rng.normal(size=(8, 8)))  # asymmetry not from u
        A = 2.0 * antisymmetric_lift(u) + extra
        residual, _ = residualize_asymmetry(A, [u])
        # the residual carries less of u's potential and is smaller than A
        self.assertLess(np.linalg.norm(residual), np.linalg.norm(A))
        self.assertLess(abs(float(np.vdot(residual.ravel(), antisymmetric_lift(u).ravel()))), 1e-6)


class WhiteningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        # a matrix with one dominant mode, to mimic the cascade
        u = rng.normal(size=(20, 1)); v = rng.normal(size=(1, 20))
        self.M = 8.0 * (u @ v) + rng.normal(scale=0.5, size=(20, 20))

    def test_alpha_zero_is_identity(self):
        np.testing.assert_allclose(fractional_whiten(self.M, 0.0), self.M, atol=1e-9)

    def test_full_whitening_equalizes_singular_values(self):
        W = fractional_whiten(self.M, 1.0)
        s = np.linalg.svd(W, compute_uv=False)
        s = s[s > 1e-8]
        self.assertLess(s.std() / s.mean(), 1e-6)   # all nonzero singular values equal

    def test_whitening_shrinks_the_dominant_mode_fraction(self):
        raw = np.linalg.svd(self.M, compute_uv=False)
        white = np.linalg.svd(fractional_whiten(self.M, 0.6), compute_uv=False)
        top_raw = raw[0] ** 2 / (raw ** 2).sum()
        top_white = white[0] ** 2 / (white ** 2).sum()
        self.assertLess(top_white, top_raw)

    def test_whitening_preserves_frobenius_norm(self):
        for a in (0.25, 0.5, 1.0):
            self.assertAlmostEqual(
                np.linalg.norm(fractional_whiten(self.M, a)), np.linalg.norm(self.M), places=6
            )


class ReproducibilityTest(unittest.TestCase):
    def test_identical_scores_reproduce_perfectly(self):
        rng = np.random.default_rng(3)
        A = response_asymmetry(rng.normal(size=(15, 15)))
        self.assertAlmostEqual(pairwise_reproducibility(A, A), 1.0, places=6)

    def test_independent_scores_do_not_reproduce(self):
        rng = np.random.default_rng(4)
        A = response_asymmetry(rng.normal(size=(30, 30)))
        B = response_asymmetry(rng.normal(size=(30, 30)))
        self.assertLess(abs(pairwise_reproducibility(A, B)), 0.3)


class PipelinePositiveControlTest(unittest.TestCase):
    """On a planted DAG, raw asymmetry must be reproducible across cell halves: a real null
    on RPE1 is then a fact about the data, not a failure of this pipeline."""

    def test_synthetic_dag_asymmetry_is_reproducible(self):
        expr, labels, true_edges = make_synthetic_interventional(
            n_genes=30, n_cells_per_condition=120, edge_density=0.2, seed=0
        )
        ds = load_interventional_frames("syn", expr, labels, reference_edges=true_edges)
        P = list(ds.perturbed_genes)
        _, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=0)
        M1 = Da.loc[P, P].to_numpy(float)
        M2 = Db.loc[P, P].to_numpy(float)
        repro = pairwise_reproducibility(response_asymmetry(M1), response_asymmetry(M2))
        self.assertGreater(repro, 0.3)


if __name__ == "__main__":
    unittest.main()
