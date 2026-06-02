import unittest

import pandas as pd

from stable_grn_inference.evaluation import precision_at_k


class PrecisionAtKTest(unittest.TestCase):
    def test_precision_at_k_uses_top_k_rows(self) -> None:
        scored_edges = pd.DataFrame(
            {
                "source": ["G1", "G2", "G3"],
                "target": ["G2", "G3", "G1"],
                "score": [0.9, 0.8, 0.1],
                "is_true": [1, 0, 1],
            }
        )

        self.assertEqual(precision_at_k(scored_edges, "is_true", 2), 0.5)

    def test_precision_at_k_requires_positive_k(self) -> None:
        scored_edges = pd.DataFrame({"is_true": [1]})

        with self.assertRaises(ValueError):
            precision_at_k(scored_edges, "is_true", 0)


if __name__ == "__main__":
    unittest.main()
