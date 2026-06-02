import unittest

import pandas as pd

from stable_grn_inference.data import trajectory_bootstrap_indices
from stable_grn_inference.evaluation import aggregate_per_network_metrics
from stable_grn_inference.inference import (
    build_dynamic_sparse_linear_grid,
    fit_dynamic_linear_coefficients,
    summarize_resampled_dynamic_linear_coefficients,
)


class DynamicSparseCoefficientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "G2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "G3": [0.0, 0.2, 0.5, 0.7, 1.0, 1.2],
            }
        )
        self.y = pd.DataFrame(
            {
                "G1": [0.2, 1.2, 2.1, 3.1, 4.2, 5.1],
                "G2": [4.9, 4.1, 2.9, 2.2, 1.1, 0.2],
                "G3": [0.1, 0.3, 0.7, 0.9, 1.2, 1.4],
            }
        )

    def test_self_predictor_coefficients_are_separate_from_edges(self) -> None:
        edges, self_coefficients = fit_dynamic_linear_coefficients(
            self.x,
            self.y,
            model_kind="lasso",
            alpha=0.03,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertEqual(len(edges), 6)
        self.assertEqual(len(self_coefficients), 3)
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertTrue(edges["score"].ge(0.0).all())
        self.assertTrue(self_coefficients["self_abs_coefficient"].ge(0.0).all())

    def test_selection_frequency_summary_is_bounded(self) -> None:
        metadata = pd.DataFrame({"trajectory_id": [1, 1, 2, 2, 3, 3]})
        resamples = trajectory_bootstrap_indices(metadata, 4, random_seed=7)

        edges, self_coefficients = summarize_resampled_dynamic_linear_coefficients(
            self.x,
            self.y,
            resamples,
            model_kind="lasso",
            alpha=0.03,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertEqual(len(edges), 6)
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertTrue(edges["selection_frequency"].between(0.0, 1.0).all())
        self.assertEqual(len(self_coefficients), 3)
        self.assertTrue(self_coefficients["self_selection_frequency"].between(0.0, 1.0).all())


class DynamicSparseGridTest(unittest.TestCase):
    def test_alpha_grid_shape_matches_validation_design(self) -> None:
        grid = build_dynamic_sparse_linear_grid(
            lasso_alphas=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
            elastic_net_alphas=[0.01, 0.03, 0.1],
            elastic_net_l1_ratios=[0.3, 0.7, 0.95],
        )

        self.assertEqual(len(grid), 42)
        self.assertEqual(len(grid[grid["model_kind"] == "lasso"]), 24)
        self.assertEqual(len(grid[grid["model_kind"] == "elastic_net"]), 18)
        self.assertEqual(grid["method"].nunique(), len(grid))


class MetricAggregationTest(unittest.TestCase):
    def test_per_network_metric_aggregation_returns_mean_std_and_count(self) -> None:
        rows = pd.DataFrame(
            {
                "method": ["a", "a", "b", "b"],
                "alpha": [0.1, 0.1, 0.3, 0.3],
                "network_id": [1, 2, 1, 2],
                "aupr": [0.2, 0.4, 0.5, 0.7],
                "auroc": [0.6, 0.8, 0.7, 0.9],
            }
        )

        aggregated = aggregate_per_network_metrics(
            rows,
            group_columns=["method", "alpha"],
            metric_columns=["aupr", "auroc"],
        )

        a_row = aggregated[aggregated["method"] == "a"].iloc[0]
        self.assertAlmostEqual(a_row["aupr"], 0.3)
        self.assertAlmostEqual(a_row["auroc"], 0.7)
        self.assertEqual(a_row["n_networks"], 2)
        self.assertIn("std_aupr", aggregated.columns)


class TrajectoryBootstrapValidationTest(unittest.TestCase):
    def test_trajectory_bootstrap_indices_remain_reproducible(self) -> None:
        metadata = pd.DataFrame({"trajectory_id": [1, 1, 2, 2, 3, 3]})

        first = trajectory_bootstrap_indices(metadata, 5, random_seed=11)
        second = trajectory_bootstrap_indices(metadata, 5, random_seed=11)

        self.assertEqual([indices.tolist() for indices in first], [indices.tolist() for indices in second])


if __name__ == "__main__":
    unittest.main()
