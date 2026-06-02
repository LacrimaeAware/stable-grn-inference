"""Tests for the BEELINE single-cell GRN benchmark adapter.

Uses a tiny synthetic fixture written to a temp directory; no real BEELINE
download is required. Genes are stored in rows (BEELINE convention) so the
orientation-correction path is exercised.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    build_tf_to_gene_candidate_edges,
    infer_expression_orientation,
    label_candidate_edges,
    load_beeline_dataset,
    read_beeline_expression,
    read_beeline_reference_edges,
)
from stable_grn_inference.evaluation import aupr, auroc
from stable_grn_inference.inference import rank_edges_by_correlation

GENES = ["G1", "G2", "G3", "G4"]
CELLS = ["C1", "C2", "C3"]
# genes x cells (BEELINE orientation), raw integer counts so log1p applies
EXPRESSION_ROWS = {
    "G1": [1, 2, 3],
    "G2": [2, 4, 6],
    "G3": [5, 0, 1],
    "G4": [0, 3, 2],
}
# directed reference; G1->G5 references a gene absent from expression (should drop)
REFERENCE_ROWS = [("G1", "G2"), ("G1", "G3"), ("G2", "G4"), ("G1", "G5")]


def _write_fixture(directory: Path, *, with_tf_file: bool, with_pseudotime: bool) -> Path:
    base = directory / "tinyDataset"
    base.mkdir(parents=True, exist_ok=True)
    expr = pd.DataFrame(EXPRESSION_ROWS, index=CELLS).T  # genes (rows) x cells (cols)
    expr.index.name = ""
    expr.to_csv(base / "ExpressionData.csv")
    pd.DataFrame(REFERENCE_ROWS, columns=["Gene1", "Gene2"]).to_csv(base / "refNetwork.csv", index=False)
    if with_tf_file:
        pd.DataFrame({"TF": ["G1", "G2"]}).to_csv(base / "TFs.csv", index=False)
    if with_pseudotime:
        pd.DataFrame({"PseudoTime": [0.0, 0.5, 1.0]}, index=CELLS).to_csv(base / "PseudoTime.csv")
    return base


class OrientationAndExpressionTest(unittest.TestCase):
    def test_infer_orientation_uses_gene_hint(self) -> None:
        genes_in_rows = pd.DataFrame(EXPRESSION_ROWS, index=CELLS).T
        genes_in_cols = genes_in_rows.T
        hint = set(GENES)
        self.assertEqual(infer_expression_orientation(genes_in_rows, gene_hint=hint), "genes_in_rows")
        self.assertEqual(infer_expression_orientation(genes_in_cols, gene_hint=hint), "genes_in_columns")
        # no hint -> BEELINE default
        self.assertEqual(infer_expression_orientation(genes_in_rows), "genes_in_rows")

    def test_read_expression_corrects_orientation_and_applies_log1p(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = _write_fixture(Path(tmp), with_tf_file=False, with_pseudotime=False)
            expr = read_beeline_expression(base / "ExpressionData.csv", log1p=True, gene_hint=set(GENES))
            self.assertEqual(expr.shape, (3, 4))  # cells x genes
            self.assertEqual(list(expr.columns), GENES)
            self.assertEqual(list(expr.index), CELLS)
            self.assertAlmostEqual(float(expr.loc["C1", "G1"]), float(np.log1p(1)))

            raw = read_beeline_expression(base / "ExpressionData.csv", log1p=False, gene_hint=set(GENES))
            self.assertAlmostEqual(float(raw.loc["C1", "G1"]), 1.0)


class ReferenceAndCandidateTest(unittest.TestCase):
    def test_read_reference_edges_dedups_and_drops_self(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = _write_fixture(Path(tmp), with_tf_file=False, with_pseudotime=False)
            ref = read_beeline_reference_edges(base / "refNetwork.csv")
            self.assertEqual(set(ref.columns), {"source", "target"})
            self.assertIn(("G1", "G2"), set(zip(ref["source"], ref["target"])))
            self.assertFalse((ref["source"] == ref["target"]).any())

    def test_candidate_edges_exclude_self_and_respect_tf_restriction(self) -> None:
        candidates = build_tf_to_gene_candidate_edges(GENES, tf_list=["G1", "G2"])
        self.assertFalse((candidates["source"] == candidates["target"]).any())
        self.assertEqual(set(candidates["source"]), {"G1", "G2"})
        self.assertEqual(len(candidates), 2 * 3)  # 2 TFs x (4-1) targets
        # no TF list -> all directed non-self pairs
        all_pairs = build_tf_to_gene_candidate_edges(GENES, tf_list=None)
        self.assertEqual(len(all_pairs), 4 * 3)

    def test_label_candidate_edges(self) -> None:
        candidates = build_tf_to_gene_candidate_edges(GENES, tf_list=["G1", "G2"])
        reference = pd.DataFrame([("G1", "G2"), ("G2", "G4")], columns=["source", "target"])
        labels = label_candidate_edges(candidates, reference)
        as_map = {(s, t): v for s, t, v in zip(labels["source"], labels["target"], labels["is_true"])}
        self.assertEqual(as_map[("G1", "G2")], 1)
        self.assertEqual(as_map[("G2", "G4")], 1)
        self.assertEqual(as_map[("G1", "G4")], 0)
        self.assertEqual(int(labels["is_true"].sum()), 2)


class LoadBeelineDatasetTest(unittest.TestCase):
    def test_tf_restricted_load_labels_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), with_tf_file=True, with_pseudotime=True)
            ds = load_beeline_dataset(tmp, "tinyDataset", reference="cell_type_specific")

            # orientation corrected to cells x genes
            self.assertEqual(ds.expression.shape, (3, 4))
            self.assertEqual(list(ds.expression.columns), GENES)
            self.assertEqual(ds.samples, CELLS)
            # TF-restricted candidates from the TFs.csv file
            self.assertTrue(ds.metadata["tf_restricted"])
            self.assertEqual(set(ds.candidate_edges["source"]), {"G1", "G2"})
            self.assertFalse((ds.candidate_edges["source"] == ds.candidate_edges["target"]).any())
            # labels: G1->G2, G1->G3, G2->G4 true; G1->G5 dropped (G5 absent)
            self.assertEqual(int(ds.edge_labels["is_true"].sum()), 3)
            self.assertEqual(ds.metadata["n_reference_dropped_missing_genes"], 1)
            self.assertEqual(ds.metadata["reference_kind"], "chip_seq_proxy")
            self.assertTrue(ds.metadata["log1p_applied"])
            self.assertTrue(ds.metadata["has_pseudotime"])
            self.assertIsNotNone(ds.pseudotime)

    def test_gold_labels_only_live_in_edge_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), with_tf_file=True, with_pseudotime=False)
            ds = load_beeline_dataset(tmp, "tinyDataset")
            self.assertEqual(set(ds.candidate_edges.columns), {"source", "target"})
            self.assertIn("is_true", ds.edge_labels.columns)
            self.assertNotIn("is_true", ds.expression.columns)
            self.assertIsNone(ds.perturbation_labels)

    def test_missing_optional_files_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), with_tf_file=False, with_pseudotime=False)
            ds = load_beeline_dataset(tmp, "tinyDataset", tf_list=None)
            self.assertIsNone(ds.pseudotime)
            self.assertFalse(ds.metadata["tf_restricted"])
            self.assertEqual(len(ds.candidate_edges), 4 * 3)  # all directed non-self pairs
            self.assertIsNone(ds.tf_list)

    def test_output_consumable_by_existing_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), with_tf_file=True, with_pseudotime=False)
            ds = load_beeline_dataset(tmp, "tinyDataset")
            scored = rank_edges_by_correlation(ds.expression)
            merged = ds.edge_labels.merge(scored, on=["source", "target"], how="left")
            self.assertEqual(len(merged), len(ds.candidate_edges))
            self.assertFalse(merged["score"].isna().any())
            # metrics run on the densified labels + scores
            self.assertTrue(np.isfinite(auroc(merged["is_true"], merged["score"])))
            self.assertTrue(np.isfinite(aupr(merged["is_true"], merged["score"])))


if __name__ == "__main__":
    unittest.main()
