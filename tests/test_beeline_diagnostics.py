"""Tests for experiment 18 (BEELINE diagnostics) and the adapter's GroundTruth
reference fallback. Synthetic temp fixtures only; no real BEELINE data needed.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import load_beeline_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP18_PATH = REPO_ROOT / "experiments" / "18_beeline_diagnostics" / "run_beeline_diagnostics.py"


def _load_exp18():
    spec = importlib.util.spec_from_file_location("exp18_module", EXP18_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroundTruthFallbackTest(unittest.TestCase):
    def test_loads_model_level_groundtruth_from_parent(self) -> None:
        # mimic BEELINE Curated layout: model-level GroundTruthNetwork.csv shared by replicates
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Model"
            rep = model / "Model-2000-1"
            rep.mkdir(parents=True)
            genes = ["G1", "G2", "G3", "G4"]
            expr = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), index=genes,
                                columns=["C1", "C2", "C3"])  # genes x cells
            expr.to_csv(rep / "ExpressionData.csv")
            pd.DataFrame([("G1", "G2"), ("G2", "G3")], columns=["Gene1", "Gene2"]).to_csv(
                model / "GroundTruthNetwork.csv", index=False)  # at MODEL level, not in rep

            ds = load_beeline_dataset(model, "Model-2000-1", reference="exact")

            self.assertEqual(ds.expression.shape, (3, 4))            # cells x genes
            self.assertTrue(ds.metadata["has_reference"])
            self.assertEqual(ds.metadata["reference_kind"], "exact")
            self.assertEqual(int(ds.edge_labels["is_true"].sum()), 2)
            labels = {(s, t): v for s, t, v in zip(ds.edge_labels["source"], ds.edge_labels["target"], ds.edge_labels["is_true"])}
            self.assertEqual(labels[("G1", "G2")], 1)
            self.assertEqual(labels[("G2", "G3")], 1)
            self.assertEqual(labels[("G1", "G4")], 0)

    def test_no_reference_anywhere_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = Path(tmp) / "ds"
            rep.mkdir()
            pd.DataFrame(np.ones((3, 3)), index=["G1", "G2", "G3"], columns=["C1", "C2", "C3"]).to_csv(rep / "ExpressionData.csv")
            ds = load_beeline_dataset(tmp, "ds")
            self.assertFalse(ds.metadata["has_reference"])
            self.assertEqual(int(ds.edge_labels["is_true"].sum()), 0)


class CellSubsampleAndEprTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load_exp18()

    def test_cell_subsamples_complementary_disjoint_and_cover(self):
        samples = self.exp.cell_subsamples(10, 2, seed=0, fraction=0.5, complementary=True)
        self.assertEqual(len(samples), 4)  # 2 draws x (chosen + complement)
        for s in samples:
            self.assertEqual(len(set(s.tolist())), len(s))     # no replacement within a subsample
            self.assertTrue(set(s.tolist()) <= set(range(10)))
        # consecutive pairs are complementary halves of all cells
        for i in range(0, 4, 2):
            chosen, comp = set(samples[i].tolist()), set(samples[i + 1].tolist())
            self.assertEqual(chosen & comp, set())
            self.assertEqual(chosen | comp, set(range(10)))

    def test_epr_formula(self):
        # top-n_true are all true -> precision@n_true = 1 -> EPR = 1 / density
        rows = [("G1", "G2", 0.9, 1), ("G2", "G3", 0.8, 1), ("G1", "G3", 0.1, 0), ("G3", "G1", 0.05, 0)]
        scored = pd.DataFrame(rows, columns=["source", "target", "score", "is_true"]).sort_values("score", ascending=False).reset_index(drop=True)
        scored["rank"] = range(1, len(scored) + 1)
        n_true, n_candidate = 2, 4
        density = n_true / n_candidate
        self.assertAlmostEqual(self.exp.epr(scored, n_true, n_candidate), 1.0 / density)


if __name__ == "__main__":
    unittest.main()
