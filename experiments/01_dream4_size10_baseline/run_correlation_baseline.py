"""Run the first DREAM4 Size10 correlation baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import load_expression_matrix, load_gold_standard_edges
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import rank_edges_by_correlation


EXPRESSION_PATH = (
    ROOT
    / "data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_1/"
    / "insilico_size10_1_multifactorial.tsv"
)
GOLD_STANDARD_PATH = (
    ROOT
    / "data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/"
    / "DREAM4_Challenge2_GoldStandards/Size 10/"
    / "DREAM4_GoldStandard_InSilico_Size10_1.tsv"
)
RESULTS_PATH = ROOT / "results/tables/dream4_size10_1_correlation_baseline.csv"


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join ranked edge scores to DREAM4 gold-standard truth labels."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        missing = scored.loc[scored["is_true"].isna(), ["source", "target"]]
        raise ValueError(f"Predicted edges missing from gold standard: {len(missing)}")
    scored["is_true"] = scored["is_true"].astype(int)
    return scored


def main() -> None:
    """Load data, run correlation ranking, compute metrics, and save results."""
    expression = load_expression_matrix(EXPRESSION_PATH)
    truth_edges = load_gold_standard_edges(GOLD_STANDARD_PATH)
    ranked_edges = rank_edges_by_correlation(expression)
    scored_edges = score_edges(ranked_edges, truth_edges)

    metrics = {
        "dataset": "insilico_size10_1",
        "expression_file": EXPRESSION_PATH.as_posix(),
        "gold_standard_file": GOLD_STANDARD_PATH.as_posix(),
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": int(scored_edges["is_true"].sum()),
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(RESULTS_PATH, index=False)

    print("DREAM4 Size10 network 1 correlation baseline")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"saved_results: {RESULTS_PATH.as_posix()}")


if __name__ == "__main__":
    main()
