import unittest

import pandas as pd

from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_elastic_net,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
)


class RankEdgesByCorrelationTest(unittest.TestCase):
    def test_returns_all_directed_non_self_edges_sorted(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0],
                "G2": [0.0, 2.0, 4.0, 6.0],
                "G3": [3.0, 2.0, 1.0, 0.0],
            }
        )

        ranked = rank_edges_by_correlation(expression)

        self.assertEqual(len(ranked), 6)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)
        self.assertEqual(ranked.iloc[0]["source"], "G1")
        self.assertEqual(ranked.iloc[0]["target"], "G2")


class RankEdgesByLassoTest(unittest.TestCase):
    def test_returns_all_directed_non_self_edges_sorted(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [0.1, 1.1, 2.1, 3.1, 4.1],
                "G3": [4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )

        ranked = rank_edges_by_lasso(expression, alpha=0.01)

        self.assertEqual(len(ranked), 6)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)

    def test_requires_positive_alpha(self) -> None:
        expression = pd.DataFrame({"G1": [0.0, 1.0], "G2": [1.0, 0.0]})

        with self.assertRaises(ValueError):
            rank_edges_by_lasso(expression, alpha=0.0)


class RankEdgesByElasticNetTest(unittest.TestCase):
    def test_returns_all_directed_non_self_edges_sorted(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [0.1, 1.1, 2.1, 3.1, 4.1],
                "G3": [4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )

        ranked = rank_edges_by_elastic_net(expression, alpha=0.01, l1_ratio=0.5)

        self.assertEqual(len(ranked), 6)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)

    def test_requires_valid_l1_ratio(self) -> None:
        expression = pd.DataFrame({"G1": [0.0, 1.0], "G2": [1.0, 0.0]})

        with self.assertRaises(ValueError):
            rank_edges_by_elastic_net(expression, alpha=0.01, l1_ratio=0.0)


class RankEdgesByRandomForestTest(unittest.TestCase):
    def test_returns_all_directed_non_self_edges_sorted(self) -> None:
        expression = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [0.1, 1.1, 2.1, 3.1, 4.1],
                "G3": [4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )

        ranked = rank_edges_by_random_forest(expression, n_estimators=10, random_state=1)

        self.assertEqual(len(ranked), 6)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)

    def test_requires_positive_n_estimators(self) -> None:
        expression = pd.DataFrame({"G1": [0.0, 1.0], "G2": [1.0, 0.0]})

        with self.assertRaises(ValueError):
            rank_edges_by_random_forest(expression, n_estimators=0)


if __name__ == "__main__":
    unittest.main()
