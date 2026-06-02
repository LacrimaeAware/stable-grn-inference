"""Tests for experiment 14 (deployable calibrated confidence).

Verifies that alpha-selection rules never see gold labels and are deterministic,
the density-prior selector, the agreement-count and topology-penalty confidence
scores, the calibration-bin computation, and that no self-edges leak into any
scorer output.
"""

import importlib.util
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP14_PATH = REPO_ROOT / "experiments" / "14_dream4_calibrated_confidence" / "run_calibrated_confidence.py"


def _load_exp14():
    spec = importlib.util.spec_from_file_location("exp14_module", EXP14_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_lagged(n_genes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(1, n_genes + 1)]
    x = pd.DataFrame(rng.random((40, n_genes)), columns=genes)
    y = pd.DataFrame(rng.random((40, n_genes)), columns=genes)
    return x, y


def _table(mapping):
    return pd.DataFrame([{"source": s, "target": t, "score": v} for (s, t), v in mapping.items()])


class AlphaSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp14()

    def test_cv_and_bic_selectors_take_no_gold_labels(self) -> None:
        # Structural guarantee: the gold-free routines accept only x/target/config.
        cv_params = set(inspect.signature(self.exp.cv_mse).parameters)
        fit_params = set(inspect.signature(self.exp.fit_targetwise).parameters)
        for forbidden in ("truth", "gold", "is_true", "y_true", "labels"):
            self.assertNotIn(forbidden, cv_params)
            self.assertNotIn(forbidden, fit_params)

    def test_cv_alpha_choice_is_deterministic_and_gold_free(self) -> None:
        x, y = _synthetic_lagged(6, seed=1)
        grid = (0.01, 0.03, 0.1, 0.3)
        cv_first = {a: self.exp.cv_mse(x, y, alpha=a, model_kind="lasso", l1_ratio=None, include_self=True, folds=4, seed=7) for a in grid}
        cv_second = {a: self.exp.cv_mse(x, y, alpha=a, model_kind="lasso", l1_ratio=None, include_self=True, folds=4, seed=7) for a in grid}
        self.assertEqual(self.exp.select_alpha_min(cv_first), self.exp.select_alpha_min(cv_second))
        self.assertIn(self.exp.select_alpha_min(cv_first), grid)

    def test_select_alpha_min_and_max(self) -> None:
        self.assertEqual(self.exp.select_alpha_min({0.01: 5.0, 0.1: 2.0, 1.0: 9.0}), 0.1)
        self.assertEqual(self.exp.select_alpha_max({0.01: 5.0, 0.1: 2.0, 1.0: 9.0}), 1.0)

    def test_density_prior_selection(self) -> None:
        # target_nnz=20: closest nnz among alphas is alpha 0.1 (18)
        nnz_by_alpha = {0.01: 90, 0.03: 50, 0.1: 18, 0.3: 4}
        self.assertEqual(self.exp.select_alpha_by_density_prior(nnz_by_alpha, 20), 0.1)
        self.assertEqual(self.exp.select_alpha_by_density_prior(nnz_by_alpha, 100), 0.01)


class ConfidenceScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp14()

    def test_agreement_count_confidence(self) -> None:
        # 3 genes -> 6 directed edges. (G1,G2) is top in all three tables.
        t1 = _table({("G1", "G2"): 0.9, ("G2", "G1"): 0.1, ("G1", "G3"): 0.2, ("G3", "G1"): 0.15, ("G2", "G3"): 0.12, ("G3", "G2"): 0.05})
        t2 = _table({("G1", "G2"): 0.8, ("G2", "G1"): 0.1, ("G1", "G3"): 0.2, ("G3", "G1"): 0.15, ("G2", "G3"): 0.12, ("G3", "G2"): 0.05})
        t3 = _table({("G1", "G2"): 0.7, ("G2", "G1"): 0.1, ("G1", "G3"): 0.2, ("G3", "G1"): 0.15, ("G2", "G3"): 0.12, ("G3", "G2"): 0.05})

        confidence = self.exp.agreement_count_confidence([t1, t2, t3], top_fraction=0.2)

        self.assertEqual(len(confidence), 6)
        self.assertFalse((confidence["source"] == confidence["target"]).any())
        # G1->G2 is supported by all 3 (top-20% of 6 = top 1 edge each), so it ranks first with score ~3
        top = confidence.iloc[0]
        self.assertEqual((top["source"], top["target"]), ("G1", "G2"))
        self.assertGreaterEqual(top["score"], 3.0)
        with self.assertRaises(ValueError):
            self.exp.agreement_count_confidence([t1], top_fraction=0.0)

    def test_topology_penalty_keeps_non_self_edges(self) -> None:
        t1 = _table({("G1", "G2"): 0.9, ("G2", "G1"): 0.85, ("G1", "G3"): 0.3, ("G3", "G1"): 0.1, ("G2", "G3"): 0.2, ("G3", "G2"): 0.15})
        t2 = _table({("G1", "G2"): 0.8, ("G2", "G1"): 0.75, ("G1", "G3"): 0.35, ("G3", "G1"): 0.1, ("G2", "G3"): 0.25, ("G3", "G2"): 0.15})

        penalized = self.exp.confidence_topology_penalty([t1, t2], penalty=0.5, top_fraction=0.5)

        self.assertEqual(len(penalized), 6)
        self.assertFalse((penalized["source"] == penalized["target"]).any())


class CalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp14()

    def test_calibration_bins_detect_monotone_confidence(self) -> None:
        # 20 edges; higher score => more likely true (perfectly ordered)
        rows = []
        for i in range(20):
            score = 1.0 - i / 20.0
            rows.append({"source": f"S{i}", "target": f"T{i}", "score": score, "is_true": 1 if i < 8 else 0})
        scored = pd.DataFrame(rows)

        bins, summary = self.exp.calibration_bins(scored, n_bins=4)

        self.assertEqual(int(bins["count"].sum()), 20)
        # top bin should have the highest empirical true rate
        self.assertGreaterEqual(bins.iloc[0]["empirical_true_rate"], bins.iloc[-1]["empirical_true_rate"])
        # confidence and true rate move together
        self.assertGreater(summary["confidence_true_rate_spearman"], 0.0)
        self.assertGreaterEqual(summary["ece"], 0.0)


class NoSelfEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp = _load_exp14()

    def test_fit_targetwise_emits_no_self_edges(self) -> None:
        x, y = _synthetic_lagged(8, seed=2)
        edges, _, _, _, _ = self.exp.fit_targetwise(x, y, alpha=0.1, model_kind="lasso", l1_ratio=None, include_self=True)
        self.assertEqual(len(edges), 8 * 7)
        self.assertFalse((edges["source"] == edges["target"]).any())


if __name__ == "__main__":
    unittest.main()
