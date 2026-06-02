import unittest

import pandas as pd

from stable_grn_inference.evaluation import (
    degree_by_node,
    directed_adjacency,
    reciprocal_false_positive_pair_count,
    reciprocal_pair_count,
    top_hub_overlap,
)


class DirectedAdjacencyTest(unittest.TestCase):
    def test_builds_directed_adjacency_from_ranked_edges_and_cutoff(self) -> None:
        edges = pd.DataFrame(
            {
                "source": ["G1", "G2", "G3"],
                "target": ["G2", "G3", "G1"],
                "rank": [1, 3, 2],
            }
        )

        adjacency = directed_adjacency(edges, genes=["G1", "G2", "G3"], cutoff=2)

        self.assertEqual(adjacency.loc["G1", "G2"], 1)
        self.assertEqual(adjacency.loc["G3", "G1"], 1)
        self.assertEqual(adjacency.loc["G2", "G3"], 0)

    def test_self_edges_are_excluded(self) -> None:
        edges = pd.DataFrame(
            {
                "source": ["G1", "G1"],
                "target": ["G1", "G2"],
                "rank": [1, 2],
            }
        )

        adjacency = directed_adjacency(edges, genes=["G1", "G2"], cutoff=2)

        self.assertEqual(adjacency.loc["G1", "G1"], 0)
        self.assertEqual(adjacency.loc["G1", "G2"], 1)


class DegreeByNodeTest(unittest.TestCase):
    def test_computes_in_degree_and_out_degree(self) -> None:
        adjacency = pd.DataFrame(
            {
                "G1": [0, 1, 0],
                "G2": [1, 0, 1],
                "G3": [1, 0, 0],
            },
            index=["G1", "G2", "G3"],
        )

        out_degree = degree_by_node(adjacency, direction="out")
        in_degree = degree_by_node(adjacency, direction="in")

        self.assertEqual(out_degree.to_dict(), {"G1": 2, "G2": 1, "G3": 1})
        self.assertEqual(in_degree.to_dict(), {"G1": 1, "G2": 2, "G3": 1})


class HubOverlapTest(unittest.TestCase):
    def test_hub_overlap_on_known_degrees(self) -> None:
        true_degree = pd.Series({"G1": 3, "G2": 2, "G3": 1, "G4": 0})
        predicted_degree = pd.Series({"G1": 1, "G2": 3, "G3": 0, "G4": 2})

        self.assertEqual(top_hub_overlap(true_degree, predicted_degree, top_n=1), 0.0)
        self.assertEqual(top_hub_overlap(true_degree, predicted_degree, top_n=3), 2 / 3)


class ReciprocalEdgeTest(unittest.TestCase):
    def test_counts_reciprocal_pairs(self) -> None:
        adjacency = pd.DataFrame(
            {
                "G1": [0, 1, 0],
                "G2": [1, 0, 1],
                "G3": [0, 0, 0],
            },
            index=["G1", "G2", "G3"],
        )

        self.assertEqual(reciprocal_pair_count(adjacency), 1)

    def test_counts_reciprocal_false_positive_pairs(self) -> None:
        predicted = pd.DataFrame(
            {
                "G1": [0, 1, 0],
                "G2": [1, 0, 0],
                "G3": [0, 0, 0],
            },
            index=["G1", "G2", "G3"],
        )
        truth = pd.DataFrame(
            {
                "G1": [0, 0, 0],
                "G2": [1, 0, 0],
                "G3": [0, 0, 0],
            },
            index=["G1", "G2", "G3"],
        )

        self.assertEqual(reciprocal_false_positive_pair_count(predicted, truth), 1)


if __name__ == "__main__":
    unittest.main()
