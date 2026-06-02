"""Run a tiny synthetic pipeline check without DREAM4 data."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import rank_edges_by_correlation


def main() -> None:
    """Rank synthetic edges and print basic recovery scores."""
    expression = pd.DataFrame(
        {
            "G1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "G2": [0.1, 1.1, 2.1, 3.1, 4.1],
            "G3": [4.0, 3.0, 2.0, 1.0, 0.0],
            "G4": [0.0, 0.5, 0.0, 0.5, 0.0],
        }
    )
    gold_edges = pd.DataFrame(
        {
            "source": ["G1", "G2", "G4"],
            "target": ["G2", "G1", "G3"],
            "is_true": [1, 1, 1],
        }
    )

    ranked_edges = rank_edges_by_correlation(expression)
    scored_edges = ranked_edges.merge(
        gold_edges,
        on=["source", "target"],
        how="left",
    )
    scored_edges["is_true"] = scored_edges["is_true"].fillna(0).astype(int)

    scores = {
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_2": precision_at_k(scored_edges, "is_true", k=2),
    }

    for name, value in scores.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
