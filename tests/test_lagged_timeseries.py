import unittest

import pandas as pd

from stable_grn_inference.data import (
    build_lagged_samples,
    split_trajectories_by_time_reset,
)
from stable_grn_inference.inference import (
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_lasso,
)


class TrajectorySplitTest(unittest.TestCase):
    def test_splits_trajectories_when_time_resets(self) -> None:
        timeseries = pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0, 0.0, 1.0],
                "G1": [1, 2, 3, 10, 11],
                "G2": [4, 5, 6, 12, 13],
            }
        )

        trajectories = split_trajectories_by_time_reset(timeseries)

        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0]["Time"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(trajectories[1]["Time"].tolist(), [0.0, 1.0])

    def test_requires_time_column(self) -> None:
        with self.assertRaises(ValueError):
            split_trajectories_by_time_reset(pd.DataFrame({"G1": [1.0]}))


class LaggedSamplesTest(unittest.TestCase):
    def test_builds_lagged_samples_within_trajectories_only(self) -> None:
        trajectories = split_trajectories_by_time_reset(
            pd.DataFrame(
                {
                    "Time": [0.0, 1.0, 2.0, 0.0, 1.0],
                    "G1": [1, 2, 3, 10, 11],
                    "G2": [4, 5, 6, 12, 13],
                }
            )
        )

        x, y, metadata = build_lagged_samples(trajectories)

        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(metadata["trajectory_id"].tolist(), [1, 1, 2])
        self.assertEqual(metadata["time_t"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(metadata["time_t1"].tolist(), [1.0, 2.0, 1.0])
        self.assertEqual(x["G1"].tolist(), [1, 2, 10])
        self.assertEqual(y["G1"].tolist(), [2, 3, 11])


class LaggedEdgeScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [4.0, 3.0, 2.0, 1.0, 0.0],
                "G3": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        self.y = pd.DataFrame(
            {
                "G1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "G2": [3.0, 2.0, 1.0, 0.0, -1.0],
                "G3": [0.6, 0.7, 0.8, 0.9, 1.0],
            }
        )

    def test_lagged_correlation_returns_all_directed_non_self_edges(self) -> None:
        ranked = rank_edges_by_lagged_correlation(self.x, self.y)

        self.assertEqual(len(ranked), 6)
        self.assertEqual(set(ranked.columns), {"source", "target", "score"})
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].is_monotonic_decreasing)

    def test_lagged_lasso_excludes_self_edges(self) -> None:
        ranked = rank_edges_by_lagged_lasso(self.x, self.y, alpha=0.1)

        self.assertEqual(len(ranked), 6)
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].ge(0.0).all())


if __name__ == "__main__":
    unittest.main()
