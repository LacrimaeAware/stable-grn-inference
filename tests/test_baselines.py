import unittest

import pandas as pd

from stable_grn_inference.inference import rank_edges_by_correlation


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


if __name__ == "__main__":
    unittest.main()
