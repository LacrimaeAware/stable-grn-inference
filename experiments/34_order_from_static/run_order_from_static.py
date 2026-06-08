r"""Experiment 34: recover an order from static data, and ask whether it helps (BoolODE).

The idea (the user's): from static, unordered single-cell data you can rebuild a 1D order from the
geometry of a similarity matrix and its higher powers (spectral seriation / diffusion pseudotime).
The order is recovered up to reversal (direction is symmetric, order is not); a root prior orients
it. And indirect, multi-step relationships (A to C to D) might be captured by iterating the
correlation matrix rather than reading single pairwise correlations.

BoolODE is the honest testbed: it has BOTH a true pseudotime AND a true network, so we can measure
order-recovery accuracy against truth (which the trajectory-inference papers do not report) and ask
whether the recovered order improves network recovery over static co-expression.

Three parts, run together:
  A. order recovery: recover a cell order from static geometry (spectral, diffusion), score it
     against the true pseudotime (absolute Spearman, since order is reversal-ambiguous). Baselines:
     ordering by the top principal component, and a random order.
  B. does the order help the network: infer directed edges by lagged scoring along the RECOVERED
     order (oriented by the true earliest cell as the root prior), vs static correlation (no order)
     and vs the TRUE-order oracle. Directed AUPR against the true network.
  C. indirect / higher-order correlation: direct |C| vs |C^2| vs second-order correlation vs
     network propagation, skeleton AUPR against the true network. Does capturing chains help or add
     spurious transitive edges?

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/34_order_from_static/run_order_from_static.py
  # --quick: fewer datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stable_grn_inference.analysis import (
    correlation_power,
    diffusion_order,
    network_propagation,
    order_recovery_score,
    orient_by_root,
    second_order_correlation,
    spectral_order,
)
from stable_grn_inference.data import load_beeline_dataset, operator_edges
from stable_grn_inference.dynamics import (
    edges_to_operator,
    skeleton_recovery_aupr,
    specific_recovery_aupr,
    static_correlation_edges,
)
from stable_grn_inference.inference.lagged import (
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_random_forest,
)

ROOT = Path(__file__).resolve().parents[2]
SYN_ROOT = ROOT / "data" / "raw" / "BEELINE-data" / "inputs" / "Synthetic"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "order_from_static"
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
    if sum(y_true) in (0, len(y_true)):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def pc1_order(X):
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, 0] * s[0]


def lagged_pairs_from_order(expr_df, coord):
    order = np.argsort(coord)
    M = expr_df.to_numpy(float)[order]
    cols = list(expr_df.columns)
    return pd.DataFrame(M[:-1], columns=cols), pd.DataFrame(M[1:], columns=cols)


def score_one(base, name, *, max_cells, seed):
    ds = load_beeline_dataset(base, name, reference="boolode", log1p=False)
    if ds.pseudotime is None or ds.reference_edges.empty:
        return None
    expr = ds.expression
    genes = list(expr.columns)
    truth_pairs = {(str(s), str(t)) for s, t in zip(ds.reference_edges["source"], ds.reference_edges["target"])}
    truth_op = edges_to_operator(ds.reference_edges, genes)
    # true order = first pseudotime column; subsample cells for the eigendecomposition
    true_t = ds.pseudotime.iloc[:, 0].to_numpy(float)
    keep = np.where(np.isfinite(true_t))[0]
    rng = np.random.default_rng(seed)
    if keep.size > max_cells:
        keep = np.sort(rng.choice(keep, size=max_cells, replace=False))
    Xkeep = expr.iloc[keep]
    t_keep = true_t[keep]
    X = Xkeep.to_numpy(float)

    # Part A: order recovery accuracy (absolute Spearman vs true pseudotime)
    spec = spectral_order(X)
    diff = diffusion_order(X)
    a = {
        "spectral": order_recovery_score(spec, t_keep),
        "diffusion": order_recovery_score(diff, t_keep),
        "pc1": order_recovery_score(pc1_order(X), t_keep),
        "random": order_recovery_score(rng.permutation(len(t_keep)), t_keep),
    }

    # Part B: does the recovered order help the network (root = true earliest cell)
    root = int(np.argmin(t_keep))
    rec = orient_by_root(spec, root)
    true_coord = t_keep  # oracle order
    xr, yr = lagged_pairs_from_order(Xkeep, rec)
    xo, yo = lagged_pairs_from_order(Xkeep, true_coord)
    static = operator_edges(static_correlation_edges(X), genes)
    b = {
        "static_corr": directed_aupr(static, truth_pairs, genes),
        "recovered_order_corr": directed_aupr(rank_edges_by_lagged_correlation(xr, yr), truth_pairs, genes),
        "recovered_order_rf": directed_aupr(rank_edges_by_lagged_random_forest(xr, yr, n_estimators=100), truth_pairs, genes),
        "true_order_corr_oracle": directed_aupr(rank_edges_by_lagged_correlation(xo, yo), truth_pairs, genes),
    }

    # Part C: indirect / higher-order correlation (skeleton AUPR vs true network)
    C = np.nan_to_num(np.corrcoef(X.T))
    c = {
        "direct_corr": skeleton_recovery_aupr(np.abs(C), truth_op),
        "corr_squared": skeleton_recovery_aupr(np.abs(correlation_power(C, 2)), truth_op),
        "second_order": skeleton_recovery_aupr(np.abs(second_order_correlation(C)), truth_op),
        "propagation": skeleton_recovery_aupr(np.abs(network_propagation(C, alpha=0.5)), truth_op),
        "chance": float(((np.abs(truth_op) + np.abs(truth_op).T) > 0)[~np.eye(len(genes), dtype=bool)].mean()),
    }
    return {"net_type": base.name.split("-")[0] + "-" + base.name.split("-")[1], "replicate": name,
            "n_cells": len(t_keep), "n_genes": len(genes), **{f"A_{k}": v for k, v in a.items()},
            **{f"B_{k}": v for k, v in b.items()}, **{f"C_{k}": v for k, v in c.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--max-replicates", type=int, default=3)
    args = ap.parse_args()
    if not SYN_ROOT.exists():
        raise SystemExit(f"No BoolODE data at {SYN_ROOT}.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    count = 200
    max_rep = 2 if args.quick else args.max_replicates

    rows = []
    for net in BOOLODE_TYPES:
        base = SYN_ROOT / net / f"{net}-{count}"
        if not base.exists():
            continue
        reps = sorted(p.name for p in base.glob(f"{net}-{count}-*") if p.is_dir())[:max_rep]
        for i, name in enumerate(reps):
            try:
                r = score_one(base, name, max_cells=args.cells, seed=i)
            except Exception:
                r = None
            if r is not None:
                rows.append(r)
    if not rows:
        raise SystemExit("No datasets scored.")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / f"{PREFIX}_all.csv", index=False)

    def block(prefix, label):
        cols = [c for c in df.columns if c.startswith(prefix)]
        mean = df[cols].mean()
        return label, pd.DataFrame([{c[len(prefix):]: float(mean[c]) for c in cols}])

    lines = ["# Experiment 34: recover an order from static data, and ask whether it helps\n",
             f"BoolODE, {len(df)} datasets ({count}-cell, up to {max_rep} replicates per topology), "
             f"true order and true network known.\n"]

    for pref, label, note in [
        ("A_", "## Part A: order-recovery accuracy (absolute Spearman vs true pseudotime)",
         "Can static geometry recover the order? Baselines: top principal component, random."),
        ("B_", "## Part B: does the recovered order help the network (directed AUPR vs truth)",
         "Lagged scoring along the recovered order (root = true earliest cell) vs static correlation vs the true-order oracle."),
        ("C_", "## Part C: indirect / higher-order correlation (skeleton AUPR vs truth)",
         "Direct correlation vs its square, second-order correlation, and network propagation."),
    ]:
        _, tbl = block(pref, label)
        lines.append(label + "\n")
        lines.append(note + "\n")
        lines.append(to_markdown_table(tbl))
        lines.append("")

    # per-topology order recovery (where does it work)
    by_type = df.groupby("net_type")[["A_spectral", "A_diffusion"]].mean().reset_index()
    lines.append("## Part A by topology (spectral / diffusion order recovery)\n")
    lines.append(to_markdown_table(by_type))

    a_spec = float(df["A_spectral"].mean()); a_pc1 = float(df["A_pc1"].mean())
    b_rec = float(df["B_recovered_order_corr"].mean()); b_static = float(df["B_static_corr"].mean())
    b_oracle = float(df["B_true_order_corr_oracle"].mean())
    c_direct = float(df["C_direct_corr"].mean()); c_best_indirect = float(max(
        df["C_corr_squared"].mean(), df["C_second_order"].mean(), df["C_propagation"].mean()))
    lines.append("\n## Verdict\n")
    lines.append(f"- Part A: static geometry recovers the order at absolute Spearman {fmt(a_spec)} "
                 f"(spectral) vs {fmt(a_pc1)} (PC1 baseline). The order is recoverable where the "
                 f"trajectory is orderable.")
    lines.append(f"- Part B: recovered-order edges score directed AUPR {fmt(b_rec)} vs static "
                 f"correlation {fmt(b_static)} and the true-order oracle {fmt(b_oracle)}. Recovered "
                 f"order {'helps' if b_rec > b_static + 0.02 else 'does not clearly help'} over static.")
    lines.append(f"- Part C: best indirect correlation {fmt(c_best_indirect)} vs direct correlation "
                 f"{fmt(c_direct)}; indirect {'helps' if c_best_indirect > c_direct + 0.02 else 'does not help'} "
                 f"(skeleton recovery).")

    pd.DataFrame([{
        "n_datasets": len(df), "A_spectral": a_spec, "A_pc1": a_pc1,
        "B_recovered_order_corr": b_rec, "B_static_corr": b_static, "B_oracle": b_oracle,
        "C_direct": c_direct, "C_best_indirect": c_best_indirect,
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
