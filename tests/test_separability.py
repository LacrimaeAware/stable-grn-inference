"""Tests for the separability phase-diagram module (synthetic only, no data files)."""

import unittest

import numpy as np

from stable_grn_inference.dynamics import (
    make_separable_system,
    recover_specific,
    RECOVERY_METHODS,
    specific_recovery_aupr,
    normalized_recovery,
    separability_grid,
    recoverability_boundary,
)


class GeneratorTest(unittest.TestCase):
    def test_realized_rho_tracks_target(self) -> None:
        low = make_separable_system(n_genes=60, density=0.05, rho=0.1, snr=1.0, seed=0)
        high = make_separable_system(n_genes=60, density=0.05, rho=0.9, snr=1.0, seed=0)
        self.assertLess(low.realized_rho, high.realized_rho)
        # realized fraction is within a reasonable band of the target
        self.assertLess(abs(low.realized_rho - 0.1), 0.2)
        self.assertGreater(high.realized_rho, 0.7)

    def test_true_edge_mask_is_off_diagonal(self) -> None:
        sys = make_separable_system(n_genes=40, density=0.08, rho=0.5, snr=1.0, seed=1)
        mask = sys.true_edge_mask
        self.assertEqual(mask.shape, (40, 40))
        self.assertFalse(np.diag(mask).any())
        self.assertGreater(mask.sum(), 0)
        self.assertTrue(0.0 < sys.true_edge_density < 1.0)

    def test_rejects_bad_parameters(self) -> None:
        for kwargs in ({"rho": 1.0}, {"rho": -0.1}, {"snr": 0.0}, {"entanglement": 1.5}):
            with self.assertRaises(ValueError):
                make_separable_system(n_genes=20, **kwargs)


class RecoveryTest(unittest.TestCase):
    def test_recover_specific_returns_offdiagonal_scores(self) -> None:
        sys = make_separable_system(n_genes=40, density=0.06, rho=0.4, snr=1.0, seed=2)
        for method in RECOVERY_METHODS:
            score = recover_specific(sys.response, method)
            self.assertEqual(score.shape, (40, 40))
            self.assertTrue(np.allclose(np.diag(score), 0.0))
            self.assertTrue(np.all(score >= 0.0))

    def test_unknown_method_raises(self) -> None:
        sys = make_separable_system(n_genes=20, density=0.1, rho=0.3, snr=1.0, seed=0)
        with self.assertRaises(ValueError):
            recover_specific(sys.response, "not_a_method")

    def test_aupr_beats_chance_in_easy_regime(self) -> None:
        sys = make_separable_system(n_genes=80, density=0.04, rho=0.3, snr=4.0, seed=0)
        aupr = specific_recovery_aupr(recover_specific(sys.response, "deflate1"), sys.true_W)
        self.assertGreater(aupr, sys.true_edge_density)  # better than the density prior
        self.assertGreater(normalized_recovery(aupr, sys.true_edge_density), 0.3)

    def test_normalized_recovery_edges(self) -> None:
        self.assertAlmostEqual(normalized_recovery(1.0, 0.02), 1.0)
        self.assertTrue(np.isnan(normalized_recovery(float("nan"), 0.02)))
        self.assertTrue(np.isnan(normalized_recovery(0.5, 1.0)))


class PhaseStructureTest(unittest.TestCase):
    def test_raw_recovery_decreases_with_rho(self) -> None:
        # at fixed snr, the dominant mode increasingly swamps raw recovery
        def nr(rho):
            sys = make_separable_system(n_genes=80, density=0.04, rho=rho, snr=2.0, seed=0)
            return normalized_recovery(
                specific_recovery_aupr(recover_specific(sys.response, "raw"), sys.true_W),
                sys.true_edge_density,
            )

        self.assertGreater(nr(0.3), nr(0.9))

    def test_deflation_is_more_rho_robust_than_raw(self) -> None:
        def nr(method, rho):
            sys = make_separable_system(n_genes=80, density=0.04, rho=rho, snr=2.0, seed=0)
            return normalized_recovery(
                specific_recovery_aupr(recover_specific(sys.response, method), sys.true_W),
                sys.true_edge_density,
            )

        raw_drop = nr("raw", 0.3) - nr("raw", 0.9)
        deflate_drop = nr("deflate1", 0.3) - nr("deflate1", 0.9)
        self.assertGreater(raw_drop, deflate_drop)  # deflation neutralizes the rho axis

    def test_snr_floor_kills_all_methods(self) -> None:
        # at very low snr the specific signal is under the noise floor for every method
        for method in RECOVERY_METHODS:
            sys = make_separable_system(n_genes=80, density=0.04, rho=0.3, snr=0.02, seed=0)
            nr = normalized_recovery(
                specific_recovery_aupr(recover_specific(sys.response, method), sys.true_W),
                sys.true_edge_density,
            )
            self.assertLess(nr, 0.2)


class GridTest(unittest.TestCase):
    def test_grid_shape_and_columns(self) -> None:
        grid = separability_grid(
            [0.3, 0.7], [0.0], snr_values=[2.0, 0.2],
            n_genes=40, density=0.06, n_seeds=2,
        )
        self.assertEqual(len(grid), 2 * 1 * 2 * len(RECOVERY_METHODS))
        for col in ("rho", "entanglement", "snr", "method", "aupr", "normalized_recovery"):
            self.assertIn(col, grid.columns)

    def test_grid_recovery_decreases_with_snr(self) -> None:
        grid = separability_grid(
            [0.4], [0.0], snr_values=[4.0, 0.05],
            n_genes=60, density=0.05, n_seeds=2,
        )
        hi = grid[(grid["method"] == "deflate1") & (grid["snr"] == 4.0)]["normalized_recovery"].iloc[0]
        lo = grid[(grid["method"] == "deflate1") & (grid["snr"] == 0.05)]["normalized_recovery"].iloc[0]
        self.assertGreater(hi, lo)

    def test_boundary_table(self) -> None:
        grid = separability_grid(
            [0.2, 0.5, 0.9], [0.0], snr_values=[2.0, 0.05],
            n_genes=40, density=0.06, n_seeds=2,
        )
        boundary = recoverability_boundary(grid, threshold=0.2, by=("snr",), axis="rho")
        self.assertIn("max_recoverable_rho", boundary.columns)
        self.assertEqual(len(boundary), len(RECOVERY_METHODS) * 2)  # methods x snr levels


if __name__ == "__main__":
    unittest.main()
