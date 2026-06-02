import tempfile
import unittest
from pathlib import Path

from stable_grn_inference.data import (
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
)


class Dream4PathHelpersTest(unittest.TestCase):
    def test_size10_expression_path_uses_regime_and_network_id(self) -> None:
        path = dream4_size10_expression_path("data/raw/dream4", 3, "knockdowns")

        self.assertEqual(
            path,
            Path("data/raw/dream4")
            / "DREAM4_InSilico_Size10"
            / "insilico_size10_3"
            / "insilico_size10_3_knockdowns.tsv",
        )

    def test_size10_gold_standard_path_uses_network_id(self) -> None:
        path = dream4_size10_gold_standard_path("data/raw/dream4", 2)

        self.assertEqual(
            path,
            Path("data/raw/dream4")
            / "DREAM4_InSilicoNetworks_GoldStandard"
            / "DREAM4_Challenge2_GoldStandards"
            / "Size 10"
            / "DREAM4_GoldStandard_InSilico_Size10_2.tsv",
        )

    def test_size10_expression_path_rejects_unknown_regime(self) -> None:
        with self.assertRaises(ValueError):
            dream4_size10_expression_path("data/raw/dream4", 1, "steady_state")


class Dream4ExpressionLoaderTest(unittest.TestCase):
    def test_load_expression_matrix_drops_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "timeseries.tsv"
            path.write_text('"Time"\t"G1"\t"G2"\n0.0\t1.0\t2.0\n1.0\t3.0\t4.0\n')

            expression = load_expression_matrix(path)

        self.assertEqual(list(expression.columns), ["G1", "G2"])
        self.assertEqual(expression.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
