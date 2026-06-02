import unittest

import pandas as pd

from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import rank_edges_by_correlation, rank_edges_by_lasso
from stable_grn_inference.stability import (
    edge_selection_frequencies,
    generate_resample_indices,
    summarize_resampled_edge_scores,
)


class EdgeSelectionFrequenciesTest(unittest.TestCase):
    def test_counts_unique_edges_across_runs(self) -> None:
        run_one = pd.DataFrame(
            {
                "source": ["G1", "G1", "G2"],
                "target": ["G2", "G2", "G3"],
            }
        )
        run_two = pd.DataFrame(
            {
                "source": ["G1", "G3"],
                "target": ["G2", "G1"],
            }
        )

        frequencies = edge_selection_frequencies([run_one, run_two])
        by_edge = {
            (row.source, row.target): row.frequency
            for row in frequencies.itertuples(index=False)
        }

        self.assertEqual(by_edge[("G1", "G2")], 1.0)
        self.assertEqual(by_edge[("G2", "G3")], 0.5)
        self.assertEqual(by_edge[("G3", "G1")], 0.5)

    def test_empty_runs_return_expected_columns(self) -> None:
        frequencies = edge_selection_frequencies([])

        self.assertEqual(list(frequencies.columns), ["source", "target", "frequency"])
        self.assertTrue(frequencies.empty)


class ResamplingTest(unittest.TestCase):
    def test_bootstrap_indices_are_reproducible_with_seed(self) -> None:
        first = generate_resample_indices(5, 3, method="bootstrap", random_seed=7)
        second = generate_resample_indices(5, 3, method="bootstrap", random_seed=7)

        self.assertEqual([indices.tolist() for indices in first], [indices.tolist() for indices in second])
        self.assertTrue(all(len(indices) == 5 for indices in first))

    def test_subsample_indices_are_reproducible_with_seed(self) -> None:
        first = generate_resample_indices(
            10,
            3,
            method="subsample",
            sample_fraction=0.5,
            random_seed=11,
        )
        second = generate_resample_indices(
            10,
            3,
            method="subsample",
            sample_fraction=0.5,
            random_seed=11,
        )

        self.assertEqual([indices.tolist() for indices in first], [indices.tolist() for indices in second])
        self.assertTrue(all(len(indices) == 5 for indices in first))
        self.assertTrue(all(len(set(indices.tolist())) == 5 for indices in first))


class ResampledEdgeScoreSummaryTest(unittest.TestCase):
    def test_stability_scores_return_one_row_per_directed_non_self_edge(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0],
                "G2": [0.0, 2.0, 4.0, 6.0],
                "G3": [3.0, 2.0, 1.0, 0.0],
            }
        )
        indices = generate_resample_indices(4, 5, method="bootstrap", random_seed=3)

        summary = summarize_resampled_edge_scores(
            expression,
            rank_edges_by_correlation,
            indices,
            top_k=2,
        )

        self.assertEqual(len(summary), 6)
        self.assertEqual(set(summary["source"]), {"G1", "G2", "G3"})
        self.assertTrue(summary["mean_reciprocal_rank"].between(0, 1).all())
        self.assertTrue(summary["top_k_frequency"].between(0, 1).all())

    def test_lasso_selection_frequencies_are_between_zero_and_one(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [0.1, 1.1, 2.1, 3.1, 4.1],
                "G3": [4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )
        indices = generate_resample_indices(5, 5, method="bootstrap", random_seed=5)

        summary = summarize_resampled_edge_scores(
            expression,
            lambda sample: rank_edges_by_lasso(sample, alpha=0.1),
            indices,
        )

        self.assertTrue(summary["selection_frequency"].between(0, 1).all())
        self.assertTrue(summary["mean_score"].ge(0).all())

    def test_metric_computation_accepts_stability_scores(self) -> None:
        scored_edges = pd.DataFrame(
            {
                "is_true": [1, 0, 1, 0],
                "score": [0.9, 0.4, 0.7, 0.1],
            }
        )

        self.assertGreaterEqual(auroc(scored_edges["is_true"], scored_edges["score"]), 0.0)
        self.assertGreaterEqual(aupr(scored_edges["is_true"], scored_edges["score"]), 0.0)
        self.assertEqual(precision_at_k(scored_edges, "is_true", 2), 0.5)


if __name__ == "__main__":
    unittest.main()
