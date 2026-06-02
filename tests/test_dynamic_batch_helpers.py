import unittest
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from stable_grn_inference.data import (
    build_dynamic_target,
    moving_average_smooth_trajectories,
    trajectory_bootstrap_indices,
)
from stable_grn_inference.inference import (
    rank_edges_by_dynamic_lasso,
    rank_edges_by_dynamic_mlp_permutation,
    rank_fusion,
)


class DynamicTargetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = pd.DataFrame({"G1": [1.0, 2.0], "G2": [4.0, 6.0]})
        self.y = pd.DataFrame({"G1": [1.5, 3.0], "G2": [5.0, 9.0]})
        self.metadata = pd.DataFrame({"time_t": [0.0, 2.0], "time_t1": [1.0, 4.0]})

    def test_delta_target_construction(self) -> None:
        target = build_dynamic_target(self.x, self.y, self.metadata, target_type="delta")

        self.assertEqual(target["G1"].tolist(), [0.5, 1.0])
        self.assertEqual(target["G2"].tolist(), [1.0, 3.0])

    def test_derivative_target_construction(self) -> None:
        target = build_dynamic_target(self.x, self.y, self.metadata, target_type="derivative")

        self.assertEqual(target["G1"].tolist(), [0.5, 0.5])
        self.assertEqual(target["G2"].tolist(), [1.0, 1.5])


class SelfPredictorModeTest(unittest.TestCase):
    def test_include_and_exclude_self_predictor_return_non_self_edges(self) -> None:
        x = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "G2": [4.0, 3.0, 2.0, 1.0, 0.0],
                "G3": [0.2, 0.3, 0.5, 0.7, 0.9],
            }
        )
        y = x.shift(-1).ffill()

        excluded = rank_edges_by_dynamic_lasso(
            x,
            y,
            alpha=0.1,
            self_predictor_mode="exclude_self_predictor",
        )
        included = rank_edges_by_dynamic_lasso(
            x,
            y,
            alpha=0.1,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertEqual(len(excluded), 6)
        self.assertEqual(len(included), 6)
        self.assertFalse((included["source"] == included["target"]).any())


class TrajectoryBootstrapTest(unittest.TestCase):
    def test_trajectory_bootstrap_is_reproducible(self) -> None:
        metadata = pd.DataFrame({"trajectory_id": [1, 1, 2, 2, 3, 3]})

        first = trajectory_bootstrap_indices(metadata, 3, random_seed=9)
        second = trajectory_bootstrap_indices(metadata, 3, random_seed=9)

        self.assertEqual([indices.tolist() for indices in first], [indices.tolist() for indices in second])
        self.assertTrue(all(len(indices) == 6 for indices in first))


class MlpEdgeScorerTest(unittest.TestCase):
    def test_mlp_returns_nonnegative_directed_non_self_edges(self) -> None:
        x = pd.DataFrame(
            {
                "G1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "G2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "G3": [1.0, 1.5, 1.2, 1.7, 1.4, 1.9],
            }
        )
        y = x.shift(-1).ffill()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ranked = rank_edges_by_dynamic_mlp_permutation(
                x,
                y,
                hidden_layer_sizes=(4,),
                alpha=0.01,
                random_state=4,
                max_iter=80,
                n_repeats=1,
            )

        self.assertEqual(len(ranked), 6)
        self.assertFalse((ranked["source"] == ranked["target"]).any())
        self.assertTrue(ranked["score"].ge(0.0).all())


class RankFusionTest(unittest.TestCase):
    def test_rank_fusion_returns_one_score_per_directed_non_self_edge(self) -> None:
        first = pd.DataFrame(
            {
                "source": ["G1", "G2"],
                "target": ["G2", "G1"],
                "score": [0.9, 0.1],
            }
        )
        second = pd.DataFrame(
            {
                "source": ["G1", "G2"],
                "target": ["G2", "G1"],
                "score": [0.2, 0.8],
            }
        )

        fused = rank_fusion([first, second], method="mean_reciprocal_rank")

        self.assertEqual(len(fused), 2)
        self.assertEqual(set(fused.columns), {"source", "target", "score"})
        self.assertTrue(fused["score"].ge(0.0).all())


class PreprocessingTest(unittest.TestCase):
    def test_moving_average_preserves_trajectory_shape(self) -> None:
        trajectory = pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0],
                "G1": [0.0, 3.0, 6.0],
                "G2": [2.0, 2.0, 8.0],
            }
        )

        smoothed = moving_average_smooth_trajectories([trajectory], window=3)

        self.assertEqual(smoothed[0].shape, trajectory.shape)
        self.assertEqual(smoothed[0]["Time"].tolist(), trajectory["Time"].tolist())


if __name__ == "__main__":
    unittest.main()
