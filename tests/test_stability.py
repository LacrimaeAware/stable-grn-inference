import unittest

import pandas as pd

from stable_grn_inference.stability import edge_selection_frequencies


class EdgeSelectionFrequenciesTest(unittest.TestCase):
    def test_counts_unique_edges_across_runs(self) -> None:
        run_one = pd.DataFrame(
            {
                "source": ["G1", "G1", "G2"],
                "target": ["G2", "G2", "G3"],
            }
        )
        run_two = pd.DataFrame(
            {
                "source": ["G1", "G3"],
                "target": ["G2", "G1"],
            }
        )

        frequencies = edge_selection_frequencies([run_one, run_two])
        by_edge = {
            (row.source, row.target): row.frequency
            for row in frequencies.itertuples(index=False)
        }

        self.assertEqual(by_edge[("G1", "G2")], 1.0)
        self.assertEqual(by_edge[("G2", "G3")], 0.5)
        self.assertEqual(by_edge[("G3", "G1")], 0.5)

    def test_empty_runs_return_expected_columns(self) -> None:
        frequencies = edge_selection_frequencies([])

        self.assertEqual(list(frequencies.columns), ["source", "target", "frequency"])
        self.assertTrue(frequencies.empty)


if __name__ == "__main__":
    unittest.main()
