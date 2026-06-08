"""Correctness tests for program discovery and heterogeneity (experiments 37-38).

Positive controls on planted truth: planted gene programs must be recovered and reproducible, and a
planted structured per-cell deviation must register as structured, reproducible, and aligned with the
direction it was planted along. If these fail on planted data, the tooling is wrong, caught here.
"""

import unittest

import numpy as np

from stable_grn_inference.analysis import (
    discover_programs,
    heterogeneity_structure,
    match_programs,
    program_reproducibility,
    residual_heterogeneity,
)


def _planted_programs(n_cells=600, n_genes=40, k=4, seed=0):
    rng = np.random.default_rng(seed)
    H = np.zeros((k, n_genes))
    block = n_genes // k
    for p in range(k):
        H[p, p * block:(p + 1) * block] = rng.uniform(0.5, 1.5, size=block)  # disjoint gene blocks
    W = rng.exponential(1.0, size=(n_cells, k))
    X = W @ H + rng.exponential(0.05, size=(n_cells, n_genes))
    return X, H


class ProgramDiscoveryTest(unittest.TestCase):
    def test_nmf_recovers_planted_programs(self):
        X, H = _planted_programs(seed=0)
        _, H_hat = discover_programs(X, k=4, method="nmf", seed=0)
        mean_cos, _ = match_programs(H, H_hat)
        self.assertGreater(mean_cos, 0.9)   # recovered the disjoint gene programs

    def test_planted_programs_are_reproducible(self):
        X, _ = _planted_programs(seed=1)
        repro, _ = program_reproducibility(X, k=4, method="nmf", seed=0)
        self.assertGreater(repro, 0.85)

    def test_random_data_programs_are_less_reproducible(self):
        rng = np.random.default_rng(2)
        X = rng.exponential(1.0, size=(600, 40))
        repro, _ = program_reproducibility(X, k=4, method="nmf", seed=0)
        planted, _ = program_reproducibility(_planted_programs(seed=3)[0], k=4, method="nmf", seed=0)
        self.assertGreater(planted, repro)


class HeterogeneityTest(unittest.TestCase):
    def test_planted_deviation_is_structured_reproducible_aligned(self):
        rng = np.random.default_rng(0)
        n, g = 500, 30
        direction = rng.normal(size=g); direction /= np.linalg.norm(direction)
        mean = rng.normal(size=g)
        # each cell = mean + (scalar along a single direction) + small isotropic noise
        amp = rng.normal(size=(n, 1))
        X = mean[None, :] + amp * direction[None, :] + 0.05 * rng.normal(size=(n, g))
        res = heterogeneity_structure(X, reference_program=direction, seed=0)
        self.assertGreater(res["top_var_fraction"], 0.7)     # deviation is low-rank (one direction)
        self.assertGreater(res["reproducibility"], 0.85)     # the direction recurs across halves
        self.assertGreater(res["reference_alignment"], 0.9)  # it is the planted direction

    def test_isotropic_noise_is_unstructured(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 30))
        res = heterogeneity_structure(X, seed=0)
        self.assertLess(res["top_var_fraction"], 0.2)        # no dominant deviation direction

    def test_residual_is_mean_zero(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 10)) + 5.0
        R = residual_heterogeneity(X)
        np.testing.assert_allclose(R.mean(axis=0), 0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
