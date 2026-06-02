"""Run a GENIE3-style baseline audit on DREAM4 Size10 data regimes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import (
    SIZE10_DATA_REGIMES,
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
)
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
)
from stable_grn_inference.stability import (
    generate_resample_indices,
    summarize_resampled_edge_scores,
)


DATA_ROOT = ROOT / "data/raw/dream4"
RESULTS_DIR = ROOT / "results/tables"
SUMMARY_PATH = RESULTS_DIR / "dream4_genie3_baseline_summary.csv"
EDGE_AUDIT_PATH = RESULTS_DIR / "dream4_genie3_baseline_edges.csv"
DEBUG_REPORT_PATH = RESULTS_DIR / "dream4_genie3_baseline_debug_report.md"

Ranker = Callable[[pd.DataFrame], pd.DataFrame]


def score_edges(predicted_edges: pd.DataFrame, truth_edges: pd.DataFrame) -> pd.DataFrame:
    """Join edge scores to DREAM4 gold-standard truth labels and assign ranks."""
    scored = predicted_edges.merge(truth_edges, on=["source", "target"], how="left")
    if scored["is_true"].isna().any():
        missing = scored.loc[scored["is_true"].isna(), ["source", "target"]]
        raise ValueError(f"Predicted edges missing from gold standard: {len(missing)}")

    scored = scored.sort_values(["score", "source", "target"], ascending=[False, True, True])
    scored = scored.reset_index(drop=True)
    scored["is_true"] = scored["is_true"].astype(int)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


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


def one_shot_methods(
    *,
    n_estimators: int,
    random_seed: int,
    n_jobs: int,
    include_extra_trees: bool,
) -> list[tuple[str, Ranker, str]]:
    """Return one-shot rankers for this GENIE3 audit."""
    methods: list[tuple[str, Ranker, str]] = [
        ("one_shot_correlation", rank_edges_by_correlation, "absolute correlation"),
        (
            "genie3_random_forest",
            lambda expression: rank_edges_by_genie3_random_forest(
                expression,
                n_estimators=n_estimators,
                random_state=random_seed + 101,
                n_jobs=n_jobs,
            ),
            f"GENIE3-style target-wise RandomForestRegressor feature importance, {n_estimators} trees",
        ),
    ]
    if include_extra_trees:
        methods.append(
            (
                "genie3_extra_trees",
                lambda expression: rank_edges_by_genie3_extra_trees(
                    expression,
                    n_estimators=n_estimators,
                    random_state=random_seed + 202,
                    n_jobs=n_jobs,
                ),
                f"GENIE3-style target-wise ExtraTreesRegressor feature importance, {n_estimators} trees",
            )
        )
    return methods


def available_regimes() -> list[str]:
    """Return Size10 data regimes present for all five networks."""
    regimes: list[str] = []
    for data_regime in SIZE10_DATA_REGIMES:
        paths = [
            dream4_size10_expression_path(DATA_ROOT, network_id, data_regime)
            for network_id in range(1, 6)
        ]
        if all(path.exists() for path in paths):
            regimes.append(data_regime)
    return regimes


def run_one_shot_method(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    data_regime: str,
    network_id: int,
    method: str,
    ranker: Ranker,
    score_definition: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run and evaluate one one-shot method."""
    scored_edges = score_edges(ranker(expression), truth_edges)
    metric_row: dict[str, float | int | str] = base_metric_row(
        data_regime,
        network_id,
        method,
        "one_shot",
        score_definition,
        expression,
    )
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges


def run_stability_correlation(
    expression: pd.DataFrame,
    truth_edges: pd.DataFrame,
    data_regime: str,
    network_id: int,
    resample_indices: list,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    """Run and evaluate bootstrap-stability correlation."""
    stability_summary = summarize_resampled_edge_scores(
        expression,
        rank_edges_by_correlation,
        resample_indices,
        top_k=20,
        selection_threshold=0.0,
    )
    predicted = stability_summary[["source", "target", "mean_reciprocal_rank"]].rename(
        columns={"mean_reciprocal_rank": "score"}
    )
    scored_edges = score_edges(predicted, truth_edges)
    metric_row: dict[str, float | int | str] = base_metric_row(
        data_regime,
        network_id,
        "stability_correlation",
        "stability",
        "mean reciprocal rank across bootstrap resamples",
        expression,
    )
    metric_row.update(evaluate_scored_edges(scored_edges))
    return metric_row, scored_edges, stability_summary


def base_metric_row(
    data_regime: str,
    network_id: int,
    method: str,
    variant: str,
    score_definition: str,
    expression: pd.DataFrame,
) -> dict[str, int | str]:
    """Create common metric metadata for one network-level result."""
    return {
        "row_type": "network",
        "data_regime": data_regime,
        "network_id": network_id,
        "network": f"insilico_size10_{network_id}",
        "method": method,
        "variant": variant,
        "score_definition": score_definition,
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
    }


def run_regime_network(
    data_regime: str,
    network_id: int,
    *,
    n_estimators: int,
    n_resamples: int,
    resampling_method: str,
    sample_fraction: float,
    random_seed: int,
    n_jobs: int,
    include_extra_trees: bool,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    """Run all GENIE3 audit methods for one data regime and one network."""
    expression = load_expression_matrix(
        dream4_size10_expression_path(DATA_ROOT, network_id, data_regime),
        drop_time=True,
    )
    truth_edges = load_gold_standard_edges(dream4_size10_gold_standard_path(DATA_ROOT, network_id))
    edge_audit = truth_edges.sort_values(["source", "target"]).reset_index(drop=True)
    edge_audit.insert(0, "network_id", network_id)
    edge_audit.insert(0, "data_regime", data_regime)
    metric_rows: list[dict[str, float | int | str]] = []

    method_seed = random_seed + (100 * network_id) + regime_seed_offset(data_regime)
    for method, ranker, score_definition in one_shot_methods(
        n_estimators=n_estimators,
        random_seed=method_seed,
        n_jobs=n_jobs,
        include_extra_trees=include_extra_trees,
    ):
        metric_row, scored_edges = run_one_shot_method(
            expression,
            truth_edges,
            data_regime,
            network_id,
            method,
            ranker,
            score_definition,
        )
        metric_rows.append(metric_row)
        edge_audit = merge_score_columns(edge_audit, scored_edges, method)

    indices = generate_resample_indices(
        len(expression),
        n_resamples,
        method=resampling_method,
        sample_fraction=sample_fraction,
        random_seed=method_seed + 303,
    )
    metric_row, scored_edges, stability_summary = run_stability_correlation(
        expression,
        truth_edges,
        data_regime,
        network_id,
        indices,
    )
    metric_row["n_resamples"] = n_resamples
    metric_row["resampling_method"] = resampling_method
    metric_rows.append(metric_row)
    edge_audit = merge_score_columns(edge_audit, scored_edges, "stability_correlation")
    edge_audit = merge_stability_columns(edge_audit, stability_summary, "stability_correlation")

    return metric_rows, edge_audit


def regime_seed_offset(data_regime: str) -> int:
    """Return a deterministic seed offset for a data regime."""
    return SIZE10_DATA_REGIMES.index(data_regime) * 1000


def merge_score_columns(edge_audit: pd.DataFrame, scored_edges: pd.DataFrame, method: str) -> pd.DataFrame:
    """Merge score and rank columns for one method into the edge audit table."""
    method_scores = scored_edges[["source", "target", "score", "rank"]].rename(
        columns={"score": f"score_{method}", "rank": f"rank_{method}"}
    )
    return edge_audit.merge(method_scores, on=["source", "target"], how="left")


def merge_stability_columns(
    edge_audit: pd.DataFrame,
    stability_summary: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Merge detailed stability summaries into the edge audit table."""
    details = stability_summary.rename(
        columns={
            "mean_score": f"mean_score_{method}",
            "mean_reciprocal_rank": f"mean_reciprocal_rank_{method}",
            "top_k_frequency": f"top20_frequency_{method}",
            "selection_frequency": f"selection_frequency_{method}",
        }
    )
    columns = [
        "source",
        "target",
        f"mean_score_{method}",
        f"mean_reciprocal_rank_{method}",
        f"top20_frequency_{method}",
        f"selection_frequency_{method}",
    ]
    return edge_audit.merge(details[columns], on=["source", "target"], how="left")


def aggregate_metrics(network_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return network-level rows plus mean rows by regime and method."""
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    grouped = network_metrics.groupby(["data_regime", "method"], as_index=False)
    mean_rows = grouped[metric_columns].mean()
    std_rows = grouped[metric_columns].std().rename(
        columns={column: f"std_{column}" for column in metric_columns}
    )
    mean_rows = mean_rows.merge(std_rows, on=["data_regime", "method"], how="left")
    mean_rows = mean_rows.merge(grouped.size().rename(columns={"size": "n_networks"}), on=["data_regime", "method"])
    mean_rows.insert(0, "row_type", "mean")
    mean_rows["network_id"] = pd.NA
    mean_rows["network"] = "mean_across_size10_networks"
    mean_rows["variant"] = mean_rows["method"].map(method_variant)
    mean_rows["score_definition"] = mean_rows["method"].map(method_score_definition)

    summary = pd.concat([network_metrics, mean_rows], ignore_index=True, sort=False)
    columns = [
        "row_type",
        "data_regime",
        "network_id",
        "network",
        "method",
        "variant",
        "score_definition",
        "n_samples",
        "n_genes",
        "n_candidate_edges",
        "n_true_edges",
        "n_resamples",
        "resampling_method",
        "n_networks",
        "auroc",
        "aupr",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "std_auroc",
        "std_aupr",
        "std_precision_at_5",
        "std_precision_at_10",
        "std_precision_at_20",
    ]
    return summary.reindex(columns=columns)


def method_variant(method: str) -> str:
    """Return broad method variant for aggregate rows."""
    if method.startswith("stability"):
        return "stability"
    return "one_shot"


def method_score_definition(method: str) -> str:
    """Return compact score definitions for aggregate rows."""
    definitions = {
        "one_shot_correlation": "absolute correlation",
        "stability_correlation": "mean reciprocal rank across bootstrap resamples",
        "genie3_random_forest": "GENIE3-style target-wise RandomForestRegressor feature importance",
        "genie3_extra_trees": "GENIE3-style target-wise ExtraTreesRegressor feature importance",
    }
    return definitions.get(method, "")


def mean_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate rows from the summary table."""
    return summary[summary["row_type"] == "mean"].copy()


def make_debug_report(summary: pd.DataFrame, edge_audit: pd.DataFrame) -> str:
    """Build a human-readable GENIE3 baseline report."""
    means = mean_rows(summary)
    best_aupr = best_methods_by_metric(means, "aupr")
    best_auroc = best_methods_by_metric(means, "auroc")
    comparison = compare_genie3_to_correlation(means)

    lines = [
        "# DREAM4 Size10 GENIE3 Baseline Debug Report",
        "",
        "This report compares correlation, stability correlation, and GENIE3-style tree ensemble rankings.",
        "",
        "Time-series rows are treated as same-time expression observations after dropping `Time`; no lagged inference is used here.",
        "",
        "## Best Method By Mean AUPR",
        "",
        to_markdown_table(best_aupr[["data_regime", "method", "aupr"]]),
        "",
        "## Best Method By Mean AUROC",
        "",
        to_markdown_table(best_auroc[["data_regime", "method", "auroc"]]),
        "",
        "## GENIE3 Versus Correlation",
        "",
        to_markdown_table(comparison),
        "",
        "## Interpretation",
        "",
        interpretation_text(comparison, best_aupr, best_auroc),
        "",
    ]

    for data_regime in ["multifactorial", most_informative_regime(comparison)]:
        if data_regime not in set(edge_audit["data_regime"]):
            continue
        regime_edges = edge_audit[
            (edge_audit["data_regime"] == data_regime)
            & (edge_audit["network_id"] == 1)
        ].copy()
        lines.extend(["", f"## Network 1 Top Edges: {data_regime}", ""])
        for method in [
            "one_shot_correlation",
            "stability_correlation",
            "genie3_random_forest",
            "genie3_extra_trees",
        ]:
            score_column = f"score_{method}"
            rank_column = f"rank_{method}"
            if score_column not in regime_edges.columns:
                continue
            top_edges = regime_edges.sort_values(rank_column).head(10).copy()
            top_edges["result"] = top_edges["is_true"].map({1: "true_positive", 0: "false_positive"})
            lines.extend(
                [
                    f"### Top 10 By {method}",
                    "",
                    to_markdown_table(
                        top_edges[["source", "target", "is_true", "result", score_column, rank_column]]
                    ),
                    "",
                ]
            )

    return "\n".join(lines)


def best_methods_by_metric(means: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the best method per data regime for one aggregate metric."""
    idx = means.groupby("data_regime")[metric].idxmax()
    return means.loc[idx, ["data_regime", "method", metric]].sort_values("data_regime")


def compare_genie3_to_correlation(means: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether the best GENIE3 variant beats correlation baselines."""
    rows: list[dict[str, float | str | bool]] = []
    for data_regime in sorted(means["data_regime"].unique()):
        regime = means[means["data_regime"] == data_regime].set_index("method")
        genie3 = regime.loc[[method for method in regime.index if method.startswith("genie3_")]]
        best_genie3_aupr_method = str(genie3["aupr"].idxmax())
        best_genie3_auroc_method = str(genie3["auroc"].idxmax())
        best_genie3_aupr = float(genie3.loc[best_genie3_aupr_method, "aupr"])
        best_genie3_auroc = float(genie3.loc[best_genie3_auroc_method, "auroc"])
        corr_aupr = float(regime.loc["one_shot_correlation", "aupr"])
        corr_auroc = float(regime.loc["one_shot_correlation", "auroc"])
        stability_aupr = float(regime.loc["stability_correlation", "aupr"])
        stability_auroc = float(regime.loc["stability_correlation", "auroc"])
        rows.append(
            {
                "data_regime": data_regime,
                "best_genie3_aupr_method": best_genie3_aupr_method,
                "best_genie3_aupr_minus_correlation": best_genie3_aupr - corr_aupr,
                "best_genie3_aupr_minus_stability_correlation": best_genie3_aupr - stability_aupr,
                "genie3_beats_correlation_aupr": best_genie3_aupr > corr_aupr,
                "genie3_beats_stability_correlation_aupr": best_genie3_aupr > stability_aupr,
                "best_genie3_auroc_method": best_genie3_auroc_method,
                "best_genie3_auroc_minus_correlation": best_genie3_auroc - corr_auroc,
                "best_genie3_auroc_minus_stability_correlation": best_genie3_auroc - stability_auroc,
                "genie3_beats_correlation_auroc": best_genie3_auroc > corr_auroc,
                "genie3_beats_stability_correlation_auroc": best_genie3_auroc > stability_auroc,
            }
        )
    return pd.DataFrame(rows)


def interpretation_text(comparison: pd.DataFrame, best_aupr: pd.DataFrame, best_auroc: pd.DataFrame) -> str:
    """Return concise interpretation text for the generated debug report."""
    aupr_genie3_regimes = best_aupr[best_aupr["method"].str.startswith("genie3_")]["data_regime"].tolist()
    auroc_genie3_regimes = best_auroc[best_auroc["method"].str.startswith("genie3_")]["data_regime"].tolist()
    stability_losses = comparison[
        ~comparison["genie3_beats_stability_correlation_aupr"]
    ]["data_regime"].tolist()

    lines = [
        f"GENIE3 has the best mean AUPR in: {format_regime_list(aupr_genie3_regimes)}.",
        f"GENIE3 has the best mean AUROC in: {format_regime_list(auroc_genie3_regimes)}.",
        f"Stability correlation remains ahead of the best GENIE3 AUPR in: {format_regime_list(stability_losses)}.",
        "This audit is not a final method claim; it is a baseline check before adding stability-aware tree ensembles or lagged time-series models.",
        "Recommended next branch: stability-GENIE3 if tree ensembles win several regimes; otherwise proper lagged time-series inference should come before Size100 scaling.",
    ]
    return "\n".join(lines)


def format_regime_list(regimes: list[str]) -> str:
    """Format a list of regimes for prose."""
    if not regimes:
        return "none"
    return ", ".join(regimes)


def most_informative_regime(comparison: pd.DataFrame) -> str:
    """Pick one non-multifactorial regime with the largest GENIE3 AUPR contrast."""
    candidates = comparison[comparison["data_regime"] != "multifactorial"].copy()
    if candidates.empty:
        return "multifactorial"
    candidates["abs_contrast"] = candidates["best_genie3_aupr_minus_stability_correlation"].abs()
    return str(candidates.sort_values("abs_contrast", ascending=False).iloc[0]["data_regime"])


def to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional packages."""
    columns = [str(column) for column in frame.columns]
    rows = [[format_cell(value) for value in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_cell(value: object) -> str:
    """Format a Markdown table cell."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_summary(summary: pd.DataFrame) -> None:
    """Print a compact aggregate summary."""
    means = mean_rows(summary)
    metric_columns = ["auroc", "aupr", "precision_at_5", "precision_at_10", "precision_at_20"]
    print("DREAM4 Size10 GENIE3 baseline audit")
    print()
    print(means[["data_regime", "method", *metric_columns]].to_string(index=False, float_format=format_float))
    print()
    print(f"saved_summary: {SUMMARY_PATH.as_posix()}")
    print(f"saved_edge_audit: {EDGE_AUDIT_PATH.as_posix()}")
    print(f"saved_debug_report: {DEBUG_REPORT_PATH.as_posix()}")


def format_float(value: float) -> str:
    """Format console metric values."""
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the GENIE3 audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--resampling-method", choices=["bootstrap", "subsample"], default="bootstrap")
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=20260602)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-extra-trees", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the Size10 GENIE3 baseline audit and write summary artifacts."""
    args = parse_args()
    regimes = available_regimes()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | int | str]] = []
    edge_tables: list[pd.DataFrame] = []
    for data_regime in regimes:
        for network_id in range(1, 6):
            rows, edge_audit = run_regime_network(
                data_regime,
                network_id,
                n_estimators=args.n_estimators,
                n_resamples=args.n_resamples,
                resampling_method=args.resampling_method,
                sample_fraction=args.sample_fraction,
                random_seed=args.random_seed,
                n_jobs=args.n_jobs,
                include_extra_trees=not args.no_extra_trees,
            )
            metric_rows.extend(rows)
            edge_tables.append(edge_audit)

    network_metrics = pd.DataFrame(metric_rows)
    summary = aggregate_metrics(network_metrics)
    edge_audit_all = pd.concat(edge_tables, ignore_index=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    edge_audit_all.to_csv(EDGE_AUDIT_PATH, index=False)
    DEBUG_REPORT_PATH.write_text(make_debug_report(summary, edge_audit_all), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
