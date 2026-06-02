import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from stable_grn_inference.inference import (
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
)


class Genie3RankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "G2": [0.1, 1.2, 1.8, 3.2, 3.9, 5.1],
                "G3": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "G4": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            }
        )

    def test_random_forest_returns_one_row_per_directed_non_self_edge(self) -> None:
        ranked = rank_edges_by_genie3_random_forest(
            self.expression,
            n_estimators=20,
            random_state=7,
            n_jobs=1,
        )

        self.assertEqual(len(ranked), 12)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)

    def test_extra_trees_excludes_self_edges_and_scores_are_nonnegative(self) -> None:
        ranked = rank_edges_by_genie3_extra_trees(
            self.expression,
            n_estimators=20,
            random_state=7,
            n_jobs=1,
        )

        self.assertEqual(len(ranked), 12)
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].ge(0.0).all())

    def test_random_forest_results_are_reproducible_with_fixed_seed(self) -> None:
        first = rank_edges_by_genie3_random_forest(
            self.expression,
            n_estimators=20,
            random_state=13,
            n_jobs=1,
        )
        second = rank_edges_by_genie3_random_forest(
            self.expression,
            n_estimators=20,
            random_state=13,
            n_jobs=1,
        )

        assert_frame_equal(first, second)

    def test_requires_positive_n_estimators(self) -> None:
        with self.assertRaises(ValueError):
            rank_edges_by_genie3_random_forest(self.expression, n_estimators=0)


if __name__ == "__main__":
    unittest.main()
