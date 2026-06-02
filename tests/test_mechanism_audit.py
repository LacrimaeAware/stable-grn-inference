"""Tests for experiment 13 (mechanism audit) and its helpers.

Covers the residualized-target construction (package), and the analysis
utilities in the experiment module: predicted-density calculation, persistence-
only baseline, reproducible self-permutation, top-k TP/FP overlap, method rank
correlation, and the metric-relationship table schema.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import residualize_target_on_self

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP13_PATH = REPO_ROOT / "experiments" / "13_dream4_mechanism_audit" / "run_mechanism_audit.py"


def _load_exp13():
    spec = importlib.util.spec_from_file_location("exp13_module", EXP13_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scored(pairs_scores, truth_pairs):
    """Build a scored edge table from {(s,t): score} and a set of true pairs."""
    rows = []
    for (s, t), v in pairs_scores.items():
        rows.append({"source": s, "target": t, "score": v, "is_true": int((s, t) in truth_pairs)})
    frame = pd.DataFrame(rows).sort_values(["score", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame


class ResidualizeTargetTest(unittest.TestCase):
    def test_residual_is_orthogonal_to_self_predictor(self) -> None:
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"G1": rng.random(30), "G2": rng.random(30)})
        y = pd.DataFrame({"G1": 0.7 * x["G1"] + 0.05 * rng.random(30) + 2.0, "G2": rng.random(30)})

        residual = residualize_target_on_self(x, y)

        self.assertEqual(list(residual.columns), ["G1", "G2"])
        self.assertAlmostEqual(float(np.corrcoef(residual["G1"], x["G1"])[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(residual["G1"].mean()), 0.0, places=6)

    def test_constant_predictor_returns_mean_centered_target(self) -> None:
        x = pd.DataFrame({"G1": [1.0, 1.0, 1.0, 1.0]})
        y = pd.DataFrame({"G1": [2.0, 4.0, 6.0, 8.0]})

        residual = residualize_target_on_self(x, y)

        self.assertEqual(residual["G1"].tolist(), [-3.0, -1.0, 1.0, 3.0])


class MechanismUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp13()

    def test_predicted_edge_density(self) -> None:
        self.assertAlmostEqual(self.exp.predicted_edge_density(700, 9900), 700 / 9900)
        self.assertAlmostEqual(self.exp.predicted_edge_density(0, 90), 0.0)
        with self.assertRaises(ValueError):
            self.exp.predicted_edge_density(5, 0)

    def test_persistence_only_r2(self) -> None:
        # G1 target is an exact linear function of its own t value -> R^2 ~ 1
        # G2 predictor is constant -> R^2 = 0
        x = pd.DataFrame({"G1": [0.0, 1.0, 2.0, 3.0, 4.0], "G2": [1.0, 1.0, 1.0, 1.0, 1.0]})
        y = pd.DataFrame({"G1": [1.0, 3.0, 5.0, 7.0, 9.0], "G2": [0.0, 1.0, 0.0, 1.0, 0.0]})

        r2 = self.exp.persistence_only_r2(x, y).set_index("target")["self_r2"]

        self.assertAlmostEqual(float(r2["G1"]), 1.0, places=6)
        self.assertAlmostEqual(float(r2["G2"]), 0.0, places=6)

    def test_self_permutation_is_reproducible(self) -> None:
        rng = np.random.default_rng(3)
        genes = [f"G{i}" for i in range(1, 7)]
        x = pd.DataFrame(rng.random((24, 6)), columns=genes)
        y = pd.DataFrame(rng.random((24, 6)), columns=genes)

        first = self.exp.score_with_permuted_self(x, y, alpha=0.1, seed=123)
        second = self.exp.score_with_permuted_self(x, y, alpha=0.1, seed=123)

        self.assertEqual(len(first), 6 * 5)  # non-self edges only
        self.assertFalse((first["source"] == first["target"]).any())
        pd.testing.assert_frame_equal(first, second)

    def test_top_k_overlap(self) -> None:
        truth = {("G1", "G2"), ("G2", "G3")}
        a = _scored({("G1", "G2"): 0.9, ("G2", "G3"): 0.8, ("G1", "G3"): 0.7, ("G3", "G1"): 0.1}, truth)
        b = _scored({("G1", "G2"): 0.9, ("G3", "G1"): 0.8, ("G1", "G3"): 0.2, ("G2", "G3"): 0.1}, truth)

        overlap = self.exp.top_k_overlap(a, b, 2)

        # top-2 of a = {G1G2, G2G3}; top-2 of b = {G1G2, G3G1}; intersection = {G1G2}
        self.assertEqual(overlap["overlap_count"], 1.0)
        self.assertAlmostEqual(overlap["jaccard"], 1 / 3)
        with self.assertRaises(ValueError):
            self.exp.top_k_overlap(a, b, 0)

    def test_rank_correlation(self) -> None:
        a = _scored({("G1", "G2"): 0.9, ("G2", "G3"): 0.6, ("G1", "G3"): 0.3, ("G3", "G1"): 0.1}, set())
        same = a.copy()
        reverse = _scored({("G1", "G2"): 0.1, ("G2", "G3"): 0.3, ("G1", "G3"): 0.6, ("G3", "G1"): 0.9}, set())

        self.assertAlmostEqual(self.exp.rank_correlation(a, same), 1.0, places=6)
        self.assertAlmostEqual(self.exp.rank_correlation(a, reverse), -1.0, places=6)

    def test_metric_relationship_schema(self) -> None:
        rng = np.random.default_rng(1)
        rows = []
        for size in (10, 100):
            for method in ("sparse", "tree", "correlation", "fusion_borda"):
                for nid in range(1, 6):
                    rows.append({
                        "size": size, "network_id": nid, "method": method,
                        "aupr": float(rng.random()), "top_hub_overlap": float(rng.random()),
                        "out_degree_spearman": float(rng.random()), "in_degree_spearman": float(rng.random()),
                        "reciprocal_fp_rate": float(rng.random()), "ffl_abs_error": float(rng.random()),
                    })
        method_metrics = pd.DataFrame(rows)

        table = self.exp.run_h4(method_metrics)

        self.assertEqual(set(table.columns), {"size", "metric_a", "metric_b", "spearman"})
        self.assertEqual(set(table["size"].unique()), {10, 100})
        # the self-correlation aupr vs aupr must be 1.0
        diag = table[(table["metric_a"] == "aupr") & (table["metric_b"] == "aupr")]
        self.assertTrue((abs(diag["spearman"] - 1.0) < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
