"""Tests for the interventional (perturbation) adapter shape (experiment 19).

Synthetic fixtures only; no real CausalBench/Replogle download required.
"""

import unittest

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    build_candidate_edges_from_perturbations,
    interventional_effect_matrix,
    interventional_orientation_asymmetry,
    load_interventional_frames,
    make_synthetic_interventional,
)


class CandidateEdgeTest(unittest.TestCase):
    def test_sources_restricted_to_perturbed(self):
        genes = ["A", "B", "C"]
        edges = build_candidate_edges_from_perturbations(genes, ["A"])
        self.assertEqual(set(edges["source"]), {"A"})
        self.assertEqual(set(zip(edges["source"], edges["target"])), {("A", "B"), ("A", "C")})
        self.assertNotIn("A", set(edges["target"]))  # no self edges

    def test_tf_list_further_restricts_sources(self):
        genes = ["A", "B", "C"]
        edges = build_candidate_edges_from_perturbations(genes, ["A", "B"], tf_list=["B"])
        self.assertEqual(set(edges["source"]), {"B"})


class LoadFramesTest(unittest.TestCase):
    def test_control_mask_and_metadata(self):
        expr = pd.DataFrame(
            np.arange(12, dtype=float).reshape(4, 3),
            index=["c0", "c1", "c2", "c3"],
            columns=["A", "B", "C"],
        )
        perturb = pd.Series(["control", "A", "B", "control"], index=expr.index)
        ref = pd.DataFrame([("A", "B")], columns=["source", "target"])
        ds = load_interventional_frames("d", expr, perturb, reference_edges=ref, reference_kind="exact")
        self.assertEqual(ds.metadata["n_control_cells"], 2)
        self.assertEqual(ds.perturbed_genes, ["A", "B"])
        self.assertEqual(ds.metadata["n_true_edges"], 1)
        self.assertTrue(ds.is_control.loc["c0"])
        self.assertFalse(ds.is_control.loc["c1"])
        # only perturbed genes are sources
        self.assertEqual(set(ds.candidate_edges["source"]), {"A", "B"})

    def test_no_reference_gives_zero_true(self):
        expr = pd.DataFrame(np.ones((3, 2)), index=["c0", "c1", "c2"], columns=["A", "B"])
        perturb = pd.Series(["control", "A", "A"], index=expr.index)
        ds = load_interventional_frames("d", expr, perturb)
        self.assertFalse(ds.metadata["has_reference"])
        self.assertEqual(int(ds.edge_labels["is_true"].sum()), 0)


class SyntheticSemTest(unittest.TestCase):
    def test_fixture_schema(self):
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=5, n_cells_per_condition=40, seed=0
        )
        # control + one block per gene
        self.assertEqual(expr.shape, (40 * (5 + 1), 5))
        self.assertEqual(set(perturb.unique()) - {"control"}, set(expr.columns))
        self.assertTrue(set(zip(true_edges["source"], true_edges["target"])))

    def test_interventional_effect_separates_true_from_false(self):
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=6, n_cells_per_condition=100, edge_density=0.4, seed=2
        )
        ds = load_interventional_frames("d", expr, perturb, reference_edges=true_edges, reference_kind="exact")
        eff = interventional_effect_matrix(ds).merge(ds.edge_labels, on=["source", "target"])
        mean_true = eff.loc[eff["is_true"] == 1, "effect"].mean()
        mean_false = eff.loc[eff["is_true"] == 0, "effect"].mean()
        self.assertGreater(mean_true, mean_false)

    def test_orientation_asymmetry_recovers_direction(self):
        # On the acyclic SEM fixture, intervention asymmetry should orient ~perfectly,
        # while a symmetric observational score cannot (this is the whole point of exp19).
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=7, n_cells_per_condition=120, edge_density=0.4, seed=1
        )
        ds = load_interventional_frames("d", expr, perturb, reference_edges=true_edges, reference_kind="exact")
        result = interventional_orientation_asymmetry(ds)
        self.assertGreaterEqual(result["n_pairs_both_perturbed"], 1)
        self.assertGreaterEqual(result["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
