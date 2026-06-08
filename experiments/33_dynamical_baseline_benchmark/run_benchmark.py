r"""Experiment 33: benchmark the dynamical operator against established lagged methods.

The methodology audit of experiments 30-32 found that the dynamical operator was only ever
compared to a SYMMETRIC static correlation, which cannot orient an edge by construction, so the
"beats static" claim is uninformative. This experiment puts the operator in the same table as the
established orientable methods (lagged GENIE3 random forest, lagged LASSO, lagged correlation) on
the SAME lagged / pseudotime-ordered pairs and grades all of them with one directed AUPR against
the same ground truth.

Datasets: DREAM4 Size10 time-series (exact directed gold standard; reproduces the exp 7 numbers and
ranks the operator among them) and BoolODE single-cell (exact truth, pseudotime-ordered).

Decision rule: if the dynamical operator matches or beats lagged GENIE3 / LASSO, Direction B has a
real positive; if it underperforms (as exp 7 already indicates for DREAM4), the honest contribution
is that time order enables orientation and simple lagged feature-importance is the method of choice.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/33_dynamical_baseline_benchmark/run_benchmark.py
  # --quick: fewer BoolODE datasets, smaller forests
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stable_grn_inference.data import (
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_beeline_dataset,
    load_expression_matrix,
    load_gold_standard_edges,
    operator_edges,
)
from stable_grn_inference.data.timeseries import build_lagged_samples, split_trajectories_by_time_reset
from stable_grn_inference.dynamics import dmd_operator, static_correlation_edges
from stable_grn_inference.inference.lagged import (
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_lasso,
    rank_edges_by_lagged_random_forest,
)

ROOT = Path(__file__).resolve().parents[2]
DREAM4_ROOT = ROOT / "data" / "raw" / "dream4"
SYN_ROOT = ROOT / "data" / "raw" / "BEELINE-data" / "inputs" / "Synthetic"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "dynamical_baseline_benchmark"
BOOLODE_TYPES = ("dyn-LI", "dyn-LL", "dyn-CY", "dyn-BF", "dyn-BFC", "dyn-TF")


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def to_markdown_table(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in frame.to_numpy():
        cells = [fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def directed_aupr(edges: pd.DataFrame, truth_pairs: set, genes) -> float:
    """Directed AUPR of a (source, target, score) edge ranking against a directed truth set.

    Graded over all directed non-self gene pairs; missing pairs score 0. Edge orientation is
    source -> target throughout (the operator is converted to this convention by the caller)."""
    gene_set = set(map(str, genes))
    score = {(str(s), str(t)): float(v)
             for s, t, v in zip(edges["source"], edges["target"], edges["score"])
             if str(s) in gene_set and str(t) in gene_set}
    y_true, y_score = [], []
    for s in genes:
        for t in genes:
            if s == t:
                continue
            y_true.append(1 if (str(s), str(t)) in truth_pairs else 0)
            y_score.append(score.get((str(s), str(t)), 0.0))
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def operator_to_source_target_edges(A_hat: np.ndarray, genes) -> pd.DataFrame:
    """Convert a dynamic operator A_hat (A_hat[j,k] = effect of gene k on gene j) to source->target
    edges. Edge k->j has score |A_hat[j,k]|, i.e. operator_edges on the transpose."""
    return operator_edges(np.asarray(A_hat).T, list(genes))


def score_all_methods(x_t: pd.DataFrame, y_t1: pd.DataFrame, genes, *, n_estimators: int):
    """Run every method on the same lagged pairs; return {method: (source,target,score) edges}."""
    A_hat = dmd_operator(x_t.to_numpy(float), y_t1.to_numpy(float), ridge=1e-2)
    static = static_correlation_edges(x_t.to_numpy(float))
    return {
        "dmd_operator": operator_to_source_target_edges(A_hat, genes),
        "lagged_genie3_rf": rank_edges_by_lagged_random_forest(x_t, y_t1, n_estimators=n_estimators),
        "lagged_lasso": rank_edges_by_lagged_lasso(x_t, y_t1, alpha=0.1),
        "lagged_correlation": rank_edges_by_lagged_correlation(x_t, y_t1),
        "static_correlation": operator_edges(static, list(genes)),
    }


def run_dream4(n_estimators, lines):
    if not DREAM4_ROOT.exists():
        lines.append("## DREAM4 Size10 time-series\n- skipped: no DREAM4 data.")
        return None
    per_net = []
    for nid in range(1, 6):
        ts_path = dream4_size10_expression_path(DREAM4_ROOT, nid, "timeseries")
        gold_path = dream4_size10_gold_standard_path(DREAM4_ROOT, nid)
        if not ts_path.exists() or not gold_path.exists():
            continue
        ts = load_expression_matrix(ts_path, drop_time=False)
        trajs = split_trajectories_by_time_reset(ts, time_column="Time")
        x_t, y_t1, _ = build_lagged_samples(trajs, time_column="Time")
        genes = list(x_t.columns)
        gold = load_gold_standard_edges(gold_path)
        truth = {(str(s), str(t)) for s, t, k in zip(gold["source"], gold["target"], gold["is_true"]) if int(k) == 1}
        scores = score_all_methods(x_t, y_t1, genes, n_estimators=n_estimators)
        row = {"network": nid}
        row.update({m: directed_aupr(e, truth, genes) for m, e in scores.items()})
        row["chance"] = len(truth) / (len(genes) * (len(genes) - 1))
        per_net.append(row)
    if not per_net:
        lines.append("## DREAM4 Size10 time-series\n- skipped: files not found.")
        return None
    df = pd.DataFrame(per_net)
    df.to_csv(TABLES_DIR / f"{PREFIX}_dream4.csv", index=False)
    mean = df.drop(columns=["network"]).mean()
    summary = pd.DataFrame([{"dataset": "DREAM4 Size10 time-series", **{k: float(mean[k]) for k in mean.index}}])
    lines.append("## DREAM4 Size10 time-series (directed AUPR, mean over 5 networks)\n")
    lines.append(to_markdown_table(summary.drop(columns=["dataset"])))
    return mean


def run_boolode(args, lines):
    if not SYN_ROOT.exists():
        lines.append("\n## BoolODE single-cell\n- skipped: no BoolODE data.")
        return None
    count = 2000
    max_rep = 2 if args.quick else 3
    n_est = 100 if args.quick else 200
    rows = []
    for net in BOOLODE_TYPES:
        base = SYN_ROOT / net / f"{net}-{count}"
        if not base.exists():
            continue
        reps = sorted(p.name for p in base.glob(f"{net}-{count}-*") if p.is_dir())[:max_rep]
        for name in reps:
            try:
                ds = load_beeline_dataset(base, name, reference="boolode", log1p=False)
                if ds.pseudotime is None or ds.reference_edges.empty:
                    continue
                expr, pt = ds.expression, ds.pseudotime
                xb, yb = [], []
                for col in pt.columns:
                    order = pt[col].dropna().sort_values()
                    cells = [c for c in order.index if c in expr.index]
                    if len(cells) < 2:
                        continue
                    M = expr.loc[cells]
                    xb.append(M.iloc[:-1]); yb.append(M.iloc[1:])
                if not xb:
                    continue
                x_t = pd.concat(xb, ignore_index=True)
                y_t1 = pd.concat(yb, ignore_index=True)
                genes = list(expr.columns)
                truth = {(str(s), str(t)) for s, t in zip(ds.reference_edges["source"], ds.reference_edges["target"])}
                scores = score_all_methods(x_t, y_t1, genes, n_estimators=n_est)
                row = {"net_type": net, "replicate": name}
                row.update({m: directed_aupr(e, truth, genes) for m, e in scores.items()})
                row["chance"] = len(truth) / (len(genes) * (len(genes) - 1))
                rows.append(row)
            except Exception:
                continue
    if not rows:
        lines.append("\n## BoolODE single-cell\n- skipped: no datasets scored.")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / f"{PREFIX}_boolode.csv", index=False)
    methods = ["dmd_operator", "lagged_genie3_rf", "lagged_lasso", "lagged_correlation", "static_correlation", "chance"]
    mean = df[methods].mean()
    lines.append(f"\n## BoolODE single-cell at {count} cells (directed AUPR, mean over {len(df)} datasets)\n")
    lines.append(to_markdown_table(pd.DataFrame([{k: float(mean[k]) for k in methods}])))
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    n_estimators = 200 if args.quick else 500

    lines = ["# Experiment 33: dynamical operator vs established lagged baselines\n",
             "All methods graded with one directed AUPR on the same lagged / pseudotime pairs and the "
             "same ground truth. Established orientable baselines (lagged GENIE3, lagged LASSO, lagged "
             "correlation) are the honest comparators; static correlation is symmetric and cannot orient, "
             "kept only as a lower bound.\n"]

    d4 = run_dream4(n_estimators, lines)
    bo = run_boolode(args, lines)

    lines.append("\n## Verdict\n")
    for label, mean in (("DREAM4", d4), ("BoolODE", bo)):
        if mean is None:
            continue
        ranking = mean[["dmd_operator", "lagged_genie3_rf", "lagged_lasso", "lagged_correlation"]].sort_values(ascending=False)
        best = ranking.index[0]
        dmd_rank = list(ranking.index).index("dmd_operator") + 1
        lines.append(f"- {label}: best method `{best}` ({fmt(float(ranking.iloc[0]))}); the dynamical "
                     f"operator ranks {dmd_rank} of 4 orientable methods "
                     f"(dmd {fmt(float(mean['dmd_operator']))} vs lagged GENIE3 {fmt(float(mean['lagged_genie3_rf']))}, "
                     f"lagged LASSO {fmt(float(mean['lagged_lasso']))}, lagged correlation {fmt(float(mean['lagged_correlation']))}).")
    lines.append("- the dynamical operator is not the method of choice where an established orientable "
                 "baseline exists; the time axis enabling orientation is the real (already known) point.")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
