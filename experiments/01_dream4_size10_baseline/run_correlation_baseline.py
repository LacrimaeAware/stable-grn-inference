"""Run DREAM4 Size10 multifactorial baseline comparisons."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import load_expression_matrix, load_gold_standard_edges
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import rank_edges_by_correlation, rank_edges_by_lasso


RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_size10_multifactorial_baseline_summary.csv"
AGGREGATE_PATH = RESULTS_DIR / "dream4_size10_multifactorial_baseline_mean_metrics.csv"

Ranker = Callable[[pd.DataFrame], pd.DataFrame]


def expression_path(network_id: int) -> Path:
    """Return the Size10 multifactorial expression path for one network."""
    return (
        ROOT
        / f"data/raw/dream4/DREAM4_InSilico_Size10/insilico_size10_{network_id}/"
        / f"insilico_size10_{network_id}_multifactorial.tsv"
    )


def gold_standard_path(network_id: int) -> Path:
    """Return the matching Size10 gold-standard topology path."""
    return (
        ROOT
        / "data/raw/dream4/DREAM4_InSilicoNetworks_GoldStandard/"
        / "DREAM4_Challenge2_GoldStandards/Size 10/"
        / f"DREAM4_GoldStandard_InSilico_Size10_{network_id}.tsv"
    )


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join ranked edge scores to DREAM4 gold-standard truth labels."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        missing = scored.loc[scored["is_true"].isna(), ["source", "target"]]
        raise ValueError(f"Predicted edges missing from gold standard: {len(missing)}")
    scored["is_true"] = scored["is_true"].astype(int)
    return scored.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_scored_edges(scored_edges: pd.DataFrame) -> dict[str, float | int]:
    """Compute edge-recovery metrics for one scored edge table."""
    return {
        "n_candidate_edges": len(scored_edges),
        "n_true_edges": int(scored_edges["is_true"].sum()),
        "auroc": auroc(scored_edges["is_true"], scored_edges["score"]),
        "aupr": aupr(scored_edges["is_true"], scored_edges["score"]),
        "precision_at_5": precision_at_k(scored_edges, "is_true", 5),
        "precision_at_10": precision_at_k(scored_edges, "is_true", 10),
        "precision_at_20": precision_at_k(scored_edges, "is_true", 20),
    }


def run_one_network(network_id: int, method_name: str, ranker: Ranker) -> dict[str, float | int | str]:
    """Run one ranking method on one DREAM4 Size10 network."""
    expression_file = expression_path(network_id)
    gold_file = gold_standard_path(network_id)

    expression = load_expression_matrix(expression_file)
    truth_edges = load_gold_standard_edges(gold_file)
    ranked_edges = ranker(expression)
    scored_edges = score_edges(ranked_edges, truth_edges)

    edge_table_path = RESULTS_DIR / f"dream4_size10_{network_id}_{method_name}_scored_edges.csv"
    scored_edges.to_csv(edge_table_path, index=False)

    metrics: dict[str, float | int | str] = {
        "network": f"insilico_size10_{network_id}",
        "method": method_name,
        "expression_file": expression_file.as_posix(),
        "gold_standard_file": gold_file.as_posix(),
        "scored_edges_file": edge_table_path.as_posix(),
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
    }
    metrics.update(evaluate_scored_edges(scored_edges))
    return metrics


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics across networks for each method."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    return summary.groupby("method", as_index=False)[metric_columns].mean()


def print_summary(summary: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    """Print a compact baseline comparison."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]

    print("DREAM4 Size10 multifactorial baseline comparison")
    print()
    print("Mean metrics")
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Per-network metrics")
    print(
        summary[["network", "method", *metric_columns]].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_aggregate: {AGGREGATE_PATH.as_posix()}")


def main() -> None:
    """Run correlation and LASSO baselines on Size10 networks 1-5."""
    methods: list[tuple[str, Ranker]] = [
        ("correlation", rank_edges_by_correlation),
        ("lasso_alpha_0_01", lambda expression: rank_edges_by_lasso(expression, alpha=0.01)),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        run_one_network(network_id, method_name, ranker)
        for network_id in range(1, 6)
        for method_name, ranker in methods
    ]
    summary = pd.DataFrame(rows)
    aggregate = aggregate_metrics(summary)

    summary.to_csv(SUMMARY_PATH, index=False)
    aggregate.to_csv(AGGREGATE_PATH, index=False)
    print_summary(summary, aggregate)


if __name__ == "__main__":
    main()
