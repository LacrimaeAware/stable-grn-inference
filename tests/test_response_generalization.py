"""Unit tests for experiment 24's core metric functions (synthetic only; no real data)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP24 = REPO_ROOT / "experiments" / "24_causalbench_response_generalization" / "run_response_generalization.py"


def _load():
    spec = importlib.util.spec_from_file_location("exp24_module", EXP24)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneralizationMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_cosine_identical_and_orthogonal(self):
        self.assertAlmostEqual(self.m.cosine(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])), 1.0, places=6)
        self.assertAlmostEqual(self.m.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, places=6)

    def test_low_rank_project_full_basis_is_identity(self):
        rng = np.random.default_rng(0)
        Q = np.linalg.qr(rng.normal(size=(5, 5)))[0]   # orthonormal rows
        v = rng.normal(size=5)
        proj = self.m.low_rank_project(v, Q)
        self.assertLess(float(np.abs(proj - v).max()), 1e-8)

    def test_low_rank_project_denoises_toward_subspace(self):
        # signal lives in a 2D subspace; projecting a noisy copy onto it should raise cosine
        rng = np.random.default_rng(1)
        basis = np.linalg.qr(rng.normal(size=(10, 2)))[0].T   # 2 x 10 orthonormal rows
        coeff = rng.normal(size=2)
        signal = basis.T @ coeff
        noisy = signal + 0.5 * rng.normal(size=10)
        proj = self.m.low_rank_project(noisy, basis)
        self.assertGreater(self.m.cosine(proj, signal), self.m.cosine(noisy, signal))

    def test_residual_cosine_removes_shared_direction(self):
        p = np.array([1.0, 0.0, 0.0])
        a = np.array([3.0, 1.0, 0.0])
        b = np.array([5.0, 0.0, 1.0])
        # after removing the x-direction, a->(0,1,0), b->(0,0,1) -> orthogonal
        self.assertAlmostEqual(self.m.residual_cosine(a, b, p), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
