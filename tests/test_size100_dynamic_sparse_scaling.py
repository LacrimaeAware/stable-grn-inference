"""Tests for the DREAM4 Size100 dynamic sparse scaling audit.

These cover the new Size100 path helpers, the 9900 directed non-self candidate
edge contract for 100 genes, within-trajectory lagged construction at Size100
scale, the include-self self-edge exclusion, and the self-coefficient
diagnostic substrate. Tests that need the local DREAM4 Size100 files skip
gracefully when the (git-ignored) data directory is absent.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    build_dynamic_target,
    build_lagged_samples,
    dream4_size100_expression_path,
    dream4_size100_gold_standard_path,
    load_gold_standard_edges,
    split_trajectories_by_time_reset,
)
from stable_grn_inference.evaluation.topology import (
    feed_forward_loop_count,
    topology_metrics_for_cutoff,
)
from stable_grn_inference.inference import fit_dynamic_linear_coefficients

DATA_ROOT = Path("data/raw/dream4")
N_GENES = 100
EXPECTED_CANDIDATE_EDGES = N_GENES * (N_GENES - 1)  # 9900


def _synthetic_size100_lagged(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic 100-gene lagged sample set across trajectories."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(1, N_GENES + 1)]
    frames = []
    for trajectory in range(3):
        times = np.arange(0.0, 7 * 50.0, 50.0)
        values = rng.random((len(times), N_GENES))
        frame = pd.DataFrame(values, columns=genes)
        frame.insert(0, "Time", times)
        frames.append(frame)
    timeseries = pd.concat(frames, ignore_index=True)
    trajectories = split_trajectories_by_time_reset(timeseries)
    x_t, y_t1, _ = build_lagged_samples(trajectories)
    return x_t, y_t1


class Size100PathHelpersTest(unittest.TestCase):
    def test_size100_expression_path_uses_regime_and_network_id(self) -> None:
        path = dream4_size100_expression_path("data/raw/dream4", 4, "timeseries")

        self.assertEqual(
            path,
            Path("data/raw/dream4")
            / "DREAM4_InSilico_Size100"
            / "insilico_size100_4"
            / "insilico_size100_4_timeseries.tsv",
        )

    def test_size100_gold_standard_path_uses_network_id(self) -> None:
        path = dream4_size100_gold_standard_path("data/raw/dream4", 2)

        self.assertEqual(
            path,
            Path("data/raw/dream4")
            / "DREAM4_InSilicoNetworks_GoldStandard"
            / "DREAM4_Challenge2_GoldStandards"
            / "Size 100"
            / "DREAM4_GoldStandard_InSilico_Size100_2.tsv",
        )

    def test_size100_expression_path_rejects_unknown_regime(self) -> None:
        with self.assertRaises(ValueError):
            dream4_size100_expression_path("data/raw/dream4", 1, "multifactorial")

    def test_size100_path_rejects_out_of_range_network(self) -> None:
        with self.assertRaises(ValueError):
            dream4_size100_gold_standard_path("data/raw/dream4", 6)


class Size100GoldStandardShapeTest(unittest.TestCase):
    def test_gold_standard_has_9900_directed_non_self_edges(self) -> None:
        path = dream4_size100_gold_standard_path(DATA_ROOT, 1)
        if not path.exists():
            self.skipTest(f"missing local DREAM4 Size100 gold standard: {path}")

        edges = load_gold_standard_edges(path)

        self.assertEqual(len(edges), EXPECTED_CANDIDATE_EDGES)
        self.assertEqual(list(edges.columns), ["source", "target", "is_true"])
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertEqual(edges[["source", "target"]].drop_duplicates().shape[0], EXPECTED_CANDIDATE_EDGES)
        self.assertTrue(set(edges["is_true"].unique()).issubset({0, 1}))
        self.assertGreater(int(edges["is_true"].sum()), 0)


class Size100LaggedConstructionTest(unittest.TestCase):
    def test_lagged_samples_respect_trajectory_boundaries_at_size100(self) -> None:
        genes = [f"G{i}" for i in range(1, N_GENES + 1)]
        timeseries = pd.DataFrame(
            np.arange(0, 5 * N_GENES, dtype=float).reshape(5, N_GENES), columns=genes
        )
        timeseries.insert(0, "Time", [0.0, 50.0, 100.0, 0.0, 50.0])

        trajectories = split_trajectories_by_time_reset(timeseries)
        x_t, y_t1, metadata = build_lagged_samples(trajectories)

        self.assertEqual(len(trajectories), 2)
        # 2 lagged pairs from the first trajectory + 1 from the second; no cross pairs.
        self.assertEqual(len(x_t), 3)
        self.assertEqual(list(x_t.columns), genes)
        self.assertEqual(x_t.shape[1], N_GENES)
        self.assertEqual(metadata["trajectory_id"].tolist(), [1, 1, 2])

    def test_level_target_matches_next_step_expression(self) -> None:
        x_t, y_t1 = _synthetic_size100_lagged()
        level = build_dynamic_target(
            x_t,
            y_t1,
            pd.DataFrame({"trajectory_id": np.zeros(len(x_t), dtype=int)}),
            target_type="level",
        )
        pd.testing.assert_frame_equal(level, y_t1)


class Size100DynamicScorerTest(unittest.TestCase):
    def test_scorer_returns_9900_directed_non_self_edges_for_100_genes(self) -> None:
        x_t, y_t1 = _synthetic_size100_lagged(seed=1)

        edges, self_coefficients = fit_dynamic_linear_coefficients(
            x_t,
            y_t1,
            model_kind="lasso",
            alpha=0.03,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertEqual(len(edges), EXPECTED_CANDIDATE_EDGES)
        self.assertEqual(
            edges[["source", "target"]].drop_duplicates().shape[0], EXPECTED_CANDIDATE_EDGES
        )
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertTrue(edges["score"].ge(0.0).all())

    def test_include_self_fitting_excludes_self_edges_but_keeps_self_coefficients(self) -> None:
        x_t, y_t1 = _synthetic_size100_lagged(seed=2)

        edges, self_coefficients = fit_dynamic_linear_coefficients(
            x_t,
            y_t1,
            model_kind="lasso",
            alpha=0.03,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertFalse((edges["source"] == edges["target"]).any())
        # one self-coefficient row per gene, even when 100 genes are present
        self.assertEqual(len(self_coefficients), N_GENES)
        self.assertEqual(set(self_coefficients["target"]), {f"G{i}" for i in range(1, N_GENES + 1)})

    def test_self_coefficient_diagnostics_work_for_100_genes(self) -> None:
        x_t, y_t1 = _synthetic_size100_lagged(seed=3)

        edges, self_coefficients = fit_dynamic_linear_coefficients(
            x_t,
            y_t1,
            model_kind="lasso",
            alpha=0.03,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        # The diagnostic substrate used by the audit must be well-formed for 100 genes.
        self_abs = self_coefficients["self_abs_coefficient"].astype(float)
        self.assertEqual(len(self_abs), N_GENES)
        self.assertTrue(self_abs.ge(0.0).all())
        self.assertTrue(self_coefficients["self_selected"].isin([True, False]).all())
        mean_abs_self = float(self_abs.mean())
        mean_abs_nonself = float(edges["score"].mean())
        ratio = mean_abs_self / mean_abs_nonself if mean_abs_nonself else 0.0
        self.assertGreaterEqual(ratio, 0.0)
        self.assertFalse(np.isnan(self_abs.quantile(0.75)))


class FeedForwardLoopVectorizationTest(unittest.TestCase):
    def test_numpy_feed_forward_loop_count_matches_brute_force(self) -> None:
        import itertools

        rng = np.random.default_rng(7)
        for _ in range(5):
            matrix = (rng.random((8, 8)) < 0.3).astype(int)
            np.fill_diagonal(matrix, 0)
            genes = [f"G{i}" for i in range(8)]
            adjacency = pd.DataFrame(matrix, index=genes, columns=genes)
            brute = sum(
                1
                for source, middle, target in itertools.permutations(genes, 3)
                if adjacency.loc[source, middle] == 1
                and adjacency.loc[middle, target] == 1
                and adjacency.loc[source, target] == 1
            )
            self.assertEqual(feed_forward_loop_count(adjacency), brute)

    def test_topology_metrics_include_top5_hub_overlap(self) -> None:
        edges = pd.DataFrame(
            {
                "source": ["G1", "G2", "G3", "G1"],
                "target": ["G2", "G3", "G1", "G3"],
                "is_true": [1, 1, 0, 1],
                "rank": [1, 2, 3, 4],
            }
        )

        metrics = topology_metrics_for_cutoff(edges, cutoff=3, rank_column="rank")

        self.assertIn("top5_out_hub_overlap", metrics)
        self.assertIn("top5_in_hub_overlap", metrics)


if __name__ == "__main__":
    unittest.main()
