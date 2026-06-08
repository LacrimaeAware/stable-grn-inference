"""Tests for the external-anchor loaders (STRING), used to grade inferred structure."""

import io
import unittest

import numpy as np
import pandas as pd

from stable_grn_inference.data import load_string_network, skeleton_truth_matrix


STRING_TSV = (
    "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\tscore\n"
    "1\t2\tSOX2\tPOU5F1\t9606\t0.99\n"
    "1\t3\tSOX2\tNANOG\t9606\t0.95\n"
    "2\t4\tPOU5F1\tMYC\t9606\t0.30\n"
)


class StringAnchorTest(unittest.TestCase):
    def setUp(self):
        self.edges = load_string_network(io.StringIO(STRING_TSV))

    def test_load_columns(self):
        self.assertEqual(list(self.edges.columns), ["source", "target", "score"])
        self.assertEqual(len(self.edges), 3)

    def test_skeleton_truth_is_symmetric_and_thresholded(self):
        genes = ["SOX2", "POU5F1", "NANOG", "MYC"]
        T = skeleton_truth_matrix(self.edges, genes, min_score=0.4)
        # SOX2-POU5F1 (0.99) and SOX2-NANOG (0.95) pass; POU5F1-MYC (0.30) does not
        self.assertEqual(T[0, 1], 1.0)
        self.assertEqual(T[1, 0], 1.0)   # symmetric
        self.assertEqual(T[0, 2], 1.0)
        self.assertEqual(T[1, 3], 0.0)   # below threshold
        np.testing.assert_allclose(np.diag(T), 0.0)
        np.testing.assert_allclose(T, T.T)

    def test_genes_outside_network_are_zero(self):
        T = skeleton_truth_matrix(self.edges, ["SOX2", "UNRELATED"], min_score=0.4)
        self.assertEqual(T.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
