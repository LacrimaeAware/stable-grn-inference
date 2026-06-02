"""Tests for experiment 17 diagnostics (stability + orientation).

Synthetic / unit-level only; no DREAM4 data, GNW, Java, or network access.
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP17_PATH = REPO_ROOT / "experiments" / "17_dream4_stability_orientation_diagnostics" / "run_stability_orientation_diagnostics.py"


def _load():
    spec = importlib.util.spec_from_file_location("exp17_module", EXP17_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scored(rows):
    """rows: list of (source, target, score, is_true)."""
    df = pd.DataFrame(rows, columns=["source", "target", "score", "is_true"])
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


class CollapseAndOrientationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load()

    def test_collapse_to_undirected_max_and_labels(self):
        scored = _scored([
            ("G1", "G2", 0.9, 1), ("G2", "G1", 0.2, 0),
            ("G1", "G3", 0.3, 0), ("G3", "G1", 0.1, 0),
            ("G2", "G3", 0.4, 1), ("G3", "G2", 0.5, 0),
        ])
        und = self.exp.collapse_to_undirected(scored, how="max")
        self.assertEqual(len(und), 3)  # C(3,2)
        as_map = {p: (s, t) for p, s, t in zip(und["pair"], und["score"], und["is_true"])}
        self.assertAlmostEqual(as_map[("G1", "G2")][0], 0.9)  # max(0.9, 0.2)
        self.assertEqual(as_map[("G1", "G2")][1], 1)          # true if either direction true
        self.assertEqual(as_map[("G2", "G3")][1], 1)
        self.assertEqual(as_map[("G1", "G3")][1], 0)

    def test_orientation_accuracy_symmetric_is_half(self):
        # symmetric scores -> orientation accuracy 0.5
        scored = _scored([
            ("G1", "G2", 0.5, 1), ("G2", "G1", 0.5, 0),
            ("G1", "G3", 0.5, 1), ("G3", "G1", 0.5, 0),
            ("G2", "G3", 0.1, 0), ("G3", "G2", 0.1, 0),
        ])
        out = self.exp.orientation_accuracy(scored)
        self.assertEqual(out["n_orientable"], 2)
        self.assertAlmostEqual(out["orientation_accuracy"], 0.5)

    def test_orientation_accuracy_perfect_is_one(self):
        scored = _scored([
            ("G1", "G2", 0.9, 1), ("G2", "G1", 0.1, 0),
            ("G2", "G3", 0.8, 1), ("G3", "G2", 0.2, 0),
            ("G1", "G3", 0.05, 0), ("G3", "G1", 0.04, 0),
        ])
        out = self.exp.orientation_accuracy(scored)
        self.assertEqual(out["n_orientable"], 2)
        self.assertAlmostEqual(out["orientation_accuracy"], 1.0)

    def test_orientation_excludes_reciprocal_true_pairs(self):
        # both directions true -> not orientable
        scored = _scored([
            ("G1", "G2", 0.9, 1), ("G2", "G1", 0.8, 1),
            ("G1", "G3", 0.7, 1), ("G3", "G1", 0.1, 0),
        ])
        out = self.exp.orientation_accuracy(scored)
        self.assertEqual(out["n_orientable"], 1)  # only G1->G3 is orientable
        self.assertAlmostEqual(out["orientation_accuracy"], 1.0)


class AlphaAndSqrtLassoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load()

    def test_theory_alpha_formula_monotonicity(self):
        # alpha proportional to sqrt(2 log p / n): increases with p, decreases with n
        f = lambda p, n: np.sqrt(2.0 * np.log(p) / n)
        self.assertGreater(f(100, 200), f(10, 200))   # larger p -> larger penalty
        self.assertGreater(f(100, 50), f(100, 500))    # smaller n -> larger penalty
        self.assertGreater(f(100, 200), 0.0)

    def test_sqrt_lasso_converges_no_self_edges(self):
        rng = np.random.default_rng(0)
        genes = [f"G{i}" for i in range(1, 6)]
        x = pd.DataFrame(rng.standard_normal((40, 5)), columns=genes)
        y = pd.DataFrame(rng.standard_normal((40, 5)), columns=genes)
        edges, alphas = self.exp.sqrt_lasso_edges(x, y, include_self=True)
        self.assertEqual(len(edges), 5 * 4)
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertEqual(len(alphas), 5)
        self.assertTrue(all(a > 0 and np.isfinite(a) for a in alphas))

    def test_fit_targetwise_no_self_edges(self):
        rng = np.random.default_rng(1)
        genes = [f"G{i}" for i in range(1, 5)]
        x = pd.DataFrame(rng.standard_normal((30, 4)), columns=genes)
        y = pd.DataFrame(rng.standard_normal((30, 4)), columns=genes)
        edges = self.exp.fit_targetwise(x, y, alpha=0.1, include_self=True)
        self.assertEqual(len(edges), 4 * 3)
        self.assertFalse((edges["source"] == edges["target"]).any())
        self.assertTrue(set(edges.columns) >= {"source", "target", "score", "selected"})


class StabilityBoundAndSubsampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load()

    def test_mb_bound_formula(self):
        # E[V] <= q^2 / ((2 pi - 1) p)
        self.assertAlmostEqual(self.exp.meinshausen_buhlmann_bound(10.0, 0.8, 99), 100.0 / (0.6 * 99))
        self.assertEqual(self.exp.meinshausen_buhlmann_bound(10.0, 0.5, 99), float("inf"))  # invalid pi
        self.assertEqual(self.exp.meinshausen_buhlmann_bound(10.0, 0.8, 0), float("inf"))

    def test_trajectory_subsamples_respect_boundaries(self):
        meta = pd.DataFrame({"trajectory_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]})
        rows_by_traj = {t: set(meta.index[meta["trajectory_id"] == t]) for t in [1, 2, 3, 4]}
        samples = self.exp.trajectory_subsamples(meta, 2, seed=0, fraction=0.5, complementary=True)
        self.assertTrue(len(samples) >= 2)
        for s in samples:
            sset = set(int(i) for i in s)
            # every trajectory is wholly in or wholly out
            for t, rows in rows_by_traj.items():
                inter = sset & rows
                self.assertIn(len(inter), (0, len(rows)))

    def test_calibration_bins_monotone(self):
        scores = np.linspace(1.0, 0.0, 20)
        labels = np.array([1] * 8 + [0] * 12)  # top scores are true
        bins, ece = self.exp.calibration_bins(scores, labels, n_bins=4)
        self.assertEqual(int(bins["count"].sum()), 20)
        self.assertGreaterEqual(bins.iloc[0]["empirical_true_rate"], bins.iloc[-1]["empirical_true_rate"])
        self.assertGreaterEqual(ece, 0.0)


class PairedComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load()

    def test_paired_comparison_detects_consistent_winner(self):
        rows = []
        for nid, (a, b) in enumerate(zip([0.5, 0.6, 0.55, 0.7, 0.65], [0.4, 0.5, 0.45, 0.6, 0.55]), start=1):
            rows.append({"network_id": nid, "method": "A", "aupr": a})
            rows.append({"network_id": nid, "method": "B", "aupr": b})
        pn = pd.DataFrame(rows)
        res = self.exp.paired_network_comparison(pn, "aupr", "A", "B")
        self.assertEqual(res["n"], 5)
        self.assertAlmostEqual(res["mean_delta"], 0.1, places=6)
        self.assertGreater(res["ci_low"], 0.0)
        self.assertEqual(res["verdict"], "a>b")
        self.assertEqual(res["wins_a"], 5)

    def test_paired_comparison_calls_tie_when_ci_crosses_zero(self):
        rows = []
        for nid, (a, b) in enumerate(zip([0.5, 0.4, 0.6, 0.45, 0.55], [0.45, 0.5, 0.5, 0.5, 0.5]), start=1):
            rows.append({"network_id": nid, "method": "A", "aupr": a})
            rows.append({"network_id": nid, "method": "B", "aupr": b})
        pn = pd.DataFrame(rows)
        res = self.exp.paired_network_comparison(pn, "aupr", "A", "B")
        self.assertTrue(res["verdict"].startswith("tie"))


if __name__ == "__main__":
    unittest.main()
