"""Correctness tests for the RENGE time-resolved Perturb-seq loader (experiment 32).

These pin the guide-to-target assignment and the 10x read path against a tiny written fixture,
so the real-data loader's behavior (dominant guide -> target gene, controls -> CONTROL_LABEL,
ambiguous cells dropped, per-cell normalization over the full gene block) is verified offline.
"""

import gzip
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stable_grn_inference.data import (
    assign_guides,
    load_renge_day,
    parse_guide_target,
)
from stable_grn_inference.data.interventional import CONTROL_LABEL


class GuideParsingTest(unittest.TestCase):
    def test_parse_guide_target(self):
        self.assertEqual(parse_guide_target("SOX2_1"), "SOX2")
        self.assertEqual(parse_guide_target("ZNF398_2"), "ZNF398")
        self.assertEqual(parse_guide_target("AAVS1_1"), "AAVS1")

    def test_assign_guides_confident_ambiguous_and_control(self):
        # cell 0: clear TFA; cell 1: ambiguous (tie); cell 2: clear control (AAVS1)
        counts = np.array([
            [10.0, 0.0, 0.0],   # guide0 dominant
            [5.0, 5.0, 0.0],    # tie -> unassigned
            [0.0, 0.0, 8.0],    # control guide dominant
        ])
        targets = np.array(["TFA", "TFB", "AAVS1"])
        labels = assign_guides(counts, targets, min_umi=1, dominance=2.0)
        self.assertEqual(labels[0], "TFA")
        self.assertEqual(labels[1], "unassigned")
        self.assertEqual(labels[2], CONTROL_LABEL)


def _write_10x(day_dir: Path, matrix, feature_rows, n_cells):
    from scipy.io import mmwrite
    from scipy.sparse import coo_matrix

    day_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    mmwrite(buf, coo_matrix(matrix))
    with gzip.open(day_dir / "matrix.mtx.gz", "wb") as fh:
        fh.write(buf.getvalue())
    import pandas as pd

    pd.DataFrame(feature_rows).to_csv(
        day_dir / "features.tsv.gz", sep="\t", header=False, index=False, compression="gzip"
    )
    pd.DataFrame({"bc": [f"cell{i}" for i in range(n_cells)]}).to_csv(
        day_dir / "barcodes.tsv.gz", sep="\t", header=False, index=False, compression="gzip"
    )


class LoadRengeDayTest(unittest.TestCase):
    def test_end_to_end_assignment_and_shape(self):
        # 3 genes + 3 guides (TFA x2, control x1); features x cells
        feature_rows = [
            ("ENSG1", "TFA", "Gene Expression"),    # TFA is both perturbed and measured
            ("ENSG2", "GENEB", "Gene Expression"),
            ("ENSG3", "GENEC", "Gene Expression"),
            ("gTFA1", "TFA_1", "CRISPR Guide Capture"),
            ("gTFA2", "TFA_2", "CRISPR Guide Capture"),
            ("gCTRL", "CTRL_1", "CRISPR Guide Capture"),
        ]
        # 4 cells: 0,1 perturbed TFA; 2 control; 3 ambiguous (dropped)
        gene_block = np.array([
            [5, 6, 7, 5],   # TFA
            [1, 1, 1, 1],   # GENEB
            [2, 2, 2, 2],   # GENEC
        ], dtype=float)
        guide_block = np.array([
            [9, 0, 0, 3],   # TFA_1
            [0, 8, 0, 3],   # TFA_2
            [0, 0, 9, 0],   # CTRL_1
        ], dtype=float)
        matrix = np.vstack([gene_block, guide_block])  # features x cells

        with TemporaryDirectory() as tmp:
            day_dir = Path(tmp) / "day"
            _write_10x(day_dir, matrix, feature_rows, n_cells=4)
            expr, labels = load_renge_day(day_dir, min_umi=1, dominance=2.0)

        # cell 3 is ambiguous (tie 3 vs 3) -> dropped, so 3 cells remain
        self.assertEqual(expr.shape[0], 3)
        self.assertEqual(list(expr.columns), ["TFA"])    # only the perturbed TF is measured
        self.assertEqual(labels.iloc[0], "TFA")
        self.assertEqual(labels.iloc[1], "TFA")
        self.assertEqual(labels.iloc[2], CONTROL_LABEL)
        self.assertTrue(np.all(np.isfinite(expr.to_numpy())))


if __name__ == "__main__":
    unittest.main()
