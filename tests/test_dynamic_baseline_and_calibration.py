"""Tests for experiment 11 (dynamic baseline + calibration + fusion) and the
experiment 12 GNW sweep design scaffold.

Covers delta/derivative target construction, the dynGENIE3-style tree scorer,
Size10/Size100 candidate edge counts, the alpha-sensitivity table shape, the
nonzero edge-density / self-coefficient diagnostics, the reciprocal-direction
penalty fusion, and the presence of the GNW design sections.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import build_dynamic_target
from stable_grn_inference.inference import (
    fit_dynamic_linear_coefficients,
    rank_edges_by_dynamic_tree_ensemble,
    rank_fusion,
    rank_fusion_with_reciprocal_penalty,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP11_PATH = (
    REPO_ROOT
    / "experiments"
    / "11_dream4_dynamic_baseline_and_calibration"
    / "run_dynamic_baseline_and_calibration.py"
)
GNW_DESIGN_PATH = REPO_ROOT / "experiments" / "12_gnw_sweep_design" / "gnw_sweep_design.md"


def _load_exp11():
    """Import the experiment 11 module by path (it only runs main under __main__)."""
    spec = importlib.util.spec_from_file_location("exp11_module", EXP11_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_lagged(n_genes: int, seed: int = 0):
    """Return small synthetic x_t, y_t1, metadata for n_genes across 2 trajectories."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(1, n_genes + 1)]
    rows_x, rows_y, meta = [], [], []
    for trajectory in range(2):
        times = [0.0, 50.0, 100.0, 150.0]
        values = rng.random((len(times), n_genes))
        for step in range(len(times) - 1):
            rows_x.append(dict(zip(genes, values[step])))
            rows_y.append(dict(zip(genes, values[step + 1])))
            meta.append({"trajectory_id": trajectory + 1, "time_t": times[step], "time_t1": times[step + 1]})
    return pd.DataFrame(rows_x), pd.DataFrame(rows_y), pd.DataFrame(meta)


class TargetConstructionTest(unittest.TestCase):
    def test_delta_target_is_next_minus_current(self) -> None:
        x = pd.DataFrame({"G1": [1.0, 2.0], "G2": [4.0, 8.0]})
        y = pd.DataFrame({"G1": [3.0, 5.0], "G2": [5.0, 11.0]})
        meta = pd.DataFrame({"time_t": [0.0, 50.0], "time_t1": [50.0, 100.0]})

        delta = build_dynamic_target(x, y, meta, target_type="delta")

        self.assertEqual(delta["G1"].tolist(), [2.0, 3.0])
        self.assertEqual(delta["G2"].tolist(), [1.0, 3.0])

    def test_derivative_target_divides_delta_by_time_step(self) -> None:
        x = pd.DataFrame({"G1": [1.0, 2.0], "G2": [4.0, 8.0]})
        y = pd.DataFrame({"G1": [3.0, 5.0], "G2": [5.0, 11.0]})
        meta = pd.DataFrame({"time_t": [0.0, 50.0], "time_t1": [50.0, 100.0]})

        derivative = build_dynamic_target(x, y, meta, target_type="derivative")

        # delta / 50
        self.assertAlmostEqual(derivative["G1"].iloc[0], 2.0 / 50.0)
        self.assertAlmostEqual(derivative["G2"].iloc[1], 3.0 / 50.0)


class DynGenie3StyleScorerTest(unittest.TestCase):
    def test_tree_scorer_returns_directed_non_self_edges_for_delta(self) -> None:
        x, y, meta = _synthetic_lagged(6, seed=1)
        delta = build_dynamic_target(x, y, meta, target_type="delta")

        edges = rank_edges_by_dynamic_tree_ensemble(
            x, delta, ensemble="random_forest", n_estimators=40, random_state=0,
            self_predictor_mode="include_self_predictor_no_self_edge",
        )

        self.assertEqual(len(edges), 6 * 5)  # directed non-self edges for 6 genes
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertEqual(set(edges.columns), {"source", "target", "score"})
        self.assertTrue(edges["score"].ge(0.0).all())

    def test_candidate_edge_counts_for_size10_and_size100(self) -> None:
        for n_genes, expected in [(10, 90), (100, 9900)]:
            x, y, meta = _synthetic_lagged(n_genes, seed=2)
            edges, _ = fit_dynamic_linear_coefficients(
                x, y, model_kind="lasso", alpha=0.1,
                self_predictor_mode="include_self_predictor_no_self_edge",
            )
            self.assertEqual(len(edges), expected)
            self.assertFalse((edges["source"] == edges["target"]).any())


class ReciprocalPenaltyFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        # 3 genes -> 6 directed edges; make (G1,G2) and (G2,G1) the top reciprocal pair.
        def table(mapping):
            return pd.DataFrame(
                [{"source": s, "target": t, "score": v} for (s, t), v in mapping.items()]
            )

        self.t1 = table({("G1", "G2"): 0.9, ("G2", "G1"): 0.8, ("G1", "G3"): 0.7,
                         ("G3", "G1"): 0.1, ("G2", "G3"): 0.6, ("G3", "G2"): 0.2})
        self.t2 = table({("G1", "G2"): 0.85, ("G2", "G1"): 0.75, ("G1", "G3"): 0.65,
                         ("G3", "G1"): 0.15, ("G2", "G3"): 0.55, ("G3", "G2"): 0.25})

    def test_penalty_demotes_weaker_reciprocal_direction(self) -> None:
        base = rank_fusion([self.t1, self.t2], method="mean_reciprocal_rank")
        penalized = rank_fusion_with_reciprocal_penalty(
            [self.t1, self.t2], penalty=0.5, top_fraction=0.5, base_method="mean_reciprocal_rank"
        )

        self.assertEqual(len(penalized), len(base))
        self.assertFalse((penalized["source"] == penalized["target"]).any())
        base_scores = {(r.source, r.target): r.score for r in base.itertuples(index=False)}
        pen_scores = {(r.source, r.target): r.score for r in penalized.itertuples(index=False)}
        # The stronger direction is unchanged; the weaker one is reduced.
        self.assertAlmostEqual(pen_scores[("G1", "G2")], base_scores[("G1", "G2")])
        self.assertLess(pen_scores[("G2", "G1")], base_scores[("G2", "G1")])

    def test_penalty_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            rank_fusion_with_reciprocal_penalty([self.t1, self.t2], penalty=0.0)
        with self.assertRaises(ValueError):
            rank_fusion_with_reciprocal_penalty([self.t1, self.t2], top_fraction=1.5)


class Experiment11ModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp11()

    def test_nonzero_edge_density_diagnostics(self) -> None:
        scored = pd.DataFrame(
            {
                "source": ["G1", "G1", "G2", "G2", "G3", "G3"],
                "target": ["G2", "G3", "G1", "G3", "G1", "G2"],
                "score": [0.5, 0.0, 0.3, 0.0, 0.0, 0.1],
                "selected": [True, False, True, False, False, True],
                "is_true": [1, 0, 1, 0, 0, 0],
            }
        )
        self_coef = pd.DataFrame(
            {
                "target": ["G1", "G2", "G3"],
                "self_coefficient": [0.9, -0.8, 0.7],
                "self_abs_coefficient": [0.9, 0.8, 0.7],
                "self_selected": [True, True, True],
            }
        )

        diag = self.exp.diagnostics_for(scored, self_coef, is_sparse=True)

        self.assertEqual(diag["n_nonzero_nonself_edges"], 3)
        self.assertAlmostEqual(diag["predicted_edge_density"], 3 / 6)
        self.assertAlmostEqual(diag["mean_abs_nonself_coefficient"], scored["score"].mean())
        expected_ratio = (0.9 + 0.8 + 0.7) / 3 / scored["score"].mean()
        self.assertAlmostEqual(diag["self_to_nonself_abs_ratio"], expected_ratio)

    def test_diagnostics_blank_for_non_sparse(self) -> None:
        scored = pd.DataFrame(
            {"source": ["G1"], "target": ["G2"], "score": [0.5], "is_true": [1]}
        )
        diag = self.exp.diagnostics_for(scored, pd.DataFrame(), is_sparse=False)
        self.assertTrue(pd.isna(diag["n_nonzero_nonself_edges"]))
        self.assertTrue(pd.isna(diag["self_to_nonself_abs_ratio"]))

    def test_alpha_sensitivity_table_shape(self) -> None:
        rows = []
        for network_id in (1, 2):
            for alpha in (0.03, 0.1):
                rows.append(
                    {
                        "size": 10,
                        "network_id": network_id,
                        "method": f"dynamic_lasso_level_include_self_a{alpha}",
                        "method_family": "sparse_linear",
                        "model_kind": "lasso",
                        "target_type": "level",
                        "self_predictor_mode": "include_self_predictor_no_self_edge",
                        "alpha": alpha,
                        "aupr": 0.6 if alpha == 0.03 else 0.5,
                        "auroc": 0.8 if alpha == 0.03 else 0.7,
                        "precision_at_10": 0.5,
                        "predicted_edge_density": 0.2 if alpha == 0.03 else 0.1,
                        "true_edge_density": 0.166,
                        "n_nonzero_nonself_edges": 18 if alpha == 0.03 else 9,
                        "self_to_nonself_abs_ratio": 8.0,
                        "topology_top3_out_hub_overlap": 0.4,
                        "topology_reciprocal_false_positive_pair_rate": 0.3,
                    }
                )
        per_network = pd.DataFrame(rows)

        table = self.exp.build_alpha_sensitivity(per_network)

        # one row per (size, model_kind, target, self_mode, alpha): 1 config x 2 alphas
        self.assertEqual(len(table), 2)
        self.assertIn("alpha", table.columns)
        self.assertIn("is_best_alpha_by_aupr", table.columns)
        self.assertEqual(int(table["is_best_alpha_by_aupr"].sum()), 1)
        best = table[table["is_best_alpha_by_aupr"]].iloc[0]
        self.assertAlmostEqual(float(best["alpha"]), 0.03)

    def test_environment_detectors_run(self) -> None:
        official, name = self.exp.detect_official_dyngenie3()
        self.assertIsInstance(official, bool)
        self.assertIsInstance(self.exp.detect_gnw(), str)


class GnwDesignScaffoldTest(unittest.TestCase):
    def test_design_file_exists_with_expected_sections(self) -> None:
        self.assertTrue(GNW_DESIGN_PATH.exists(), f"missing {GNW_DESIGN_PATH}")
        text = GNW_DESIGN_PATH.read_text(encoding="utf-8")
        for heading in (
            "# GeneNetWeaver Simulation Sweep Design",
            "## Sweep Dimensions",
            "## Methods To Run",
            "## Metrics",
            "## Success Questions",
            "## GeneNetWeaver Availability",
        ):
            self.assertIn(heading, text, f"missing section: {heading}")
        # required sweep values / keywords
        for keyword in ("10, 30, 50, 100", "21, 50, 100", "5, 10, 20", "noise", "knockouts", "rank fusion"):
            self.assertIn(keyword, text, f"missing keyword: {keyword}")


if __name__ == "__main__":
    unittest.main()
