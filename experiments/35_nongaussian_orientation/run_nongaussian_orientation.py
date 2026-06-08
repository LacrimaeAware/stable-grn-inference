r"""Experiment 35: direction from static data via non-Gaussianity, and a detectability map.

Two of the project's candidate directions, together (directions 1 and 2 in docs/research_directions.md):

A. Non-Gaussian orientation (LiNGAM idea). Correlation is symmetric and cannot orient an edge; the
   arrow lives in the higher moments. When noise is non-Gaussian (gene expression is), direction is
   identifiable from static data. Question: does a non-Gaussian directed score beat the symmetric
   correlation baseline at DIRECTED recovery, on static data with no time axis? Acyclic vs cyclic is
   reported, because LiNGAM assumes acyclicity and should degrade on cycles.
B. Detectability map. Per edge, how far does its correlation sit from a permutation null? Reports how
   many true edges are detectable above the null versus how many false edges leak in, i.e. the
   per-edge version of the SNR floor.

BoolODE is the testbed (static single-cell expression, exact network truth). Baselines: symmetric
correlation (cannot orient) and GENIE3 (the established method). Standing question from the research:
no published method has been shown to beat simple baselines on real directed-GRN recovery, so this is
a head-to-head against them on truth.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/35_nongaussian_orientation/run_nongaussian_orientation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stable_grn_inference.analysis import (
    edge_detectability,
    nongaussian_directed_edges,
    nongaussianity,
    pairwise_orientation,
)
from stable_grn_inference.data import load_beeline_dataset, operator_edges

ROOT = Path(__file__).resolve().parents[2]
SYN_ROOT = ROOT / "data" / "raw" / "BEELINE-data" / "inputs" / "Synthetic"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "nongaussian_orientation"
ACYCLIC = ("dyn-LI", "dyn-LL")
CYCLIC = ("dyn-CY", "dyn-BF", "dyn-BFC", "dyn-TF")


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


def directed_aupr_matrix(score, truth_pairs, genes) -> float:
    n = len(genes)
    yt, ys = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            yt.append(1 if (str(genes[i]), str(genes[j])) in truth_pairs else 0)
            ys.append(float(score[i, j]))
    if sum(yt) in (0, len(yt)):
        return float("nan")
    return float(average_precision_score(yt, ys))


def genie3_directed_aupr(expr_df, truth_pairs, genes):
    try:
        from stable_grn_inference.inference.genie3 import rank_edges_by_genie3_random_forest
        edges = rank_edges_by_genie3_random_forest(expr_df)
        score = {(str(s), str(t)): float(v) for s, t, v in zip(edges["source"], edges["target"], edges["score"])}
        mat = np.zeros((len(genes), len(genes)))
        idx = {g: i for i, g in enumerate(genes)}
        for (s, t), v in score.items():
            if s in idx and t in idx:
                mat[idx[s], idx[t]] = v
        return directed_aupr_matrix(mat, truth_pairs, genes)
    except Exception:
        return float("nan")


def score_one(base, name):
    ds = load_beeline_dataset(base, name, reference="boolode", log1p=False)
    if ds.reference_edges.empty:
        return None
    expr = ds.expression
    genes = list(expr.columns)
    X = expr.to_numpy(float)
    truth_pairs = {(str(s), str(t)) for s, t in zip(ds.reference_edges["source"], ds.reference_edges["target"])}
    n = len(genes)

    C = np.abs(np.nan_to_num(np.corrcoef(X.T)))
    sym = C.copy(); np.fill_diagonal(sym, 0.0)
    directed = nongaussian_directed_edges(X)
    z = edge_detectability(X, n_perm=100, seed=0)

    # detectability: true vs false edge z, and detectable-true-edge rate (z > 2)
    off = ~np.eye(n, dtype=bool)
    true_mask = np.zeros((n, n), dtype=bool)
    for s, t in truth_pairs:
        if s in genes and t in genes:
            true_mask[genes.index(s), genes.index(t)] = True
    z_true = z[true_mask].mean() if true_mask.any() else float("nan")
    z_false = z[off & ~true_mask & ~true_mask.T].mean()
    detectable_true_rate = float((z[true_mask] > 2.0).mean()) if true_mask.any() else float("nan")

    return {
        "net_type": base.name.rsplit("-", 1)[0], "replicate": name, "n_genes": n,
        "nongaussianity": float(nongaussianity(X).mean()),
        "aupr_symmetric_corr": directed_aupr_matrix(sym, truth_pairs, genes),
        "aupr_nongaussian_directed": directed_aupr_matrix(directed, truth_pairs, genes),
        "aupr_genie3": genie3_directed_aupr(expr, truth_pairs, genes),
        "z_true_edges": float(z_true), "z_false_edges": float(z_false),
        "detectable_true_rate": detectable_true_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--max-replicates", type=int, default=3)
    args = ap.parse_args()
    if not SYN_ROOT.exists():
        raise SystemExit(f"No BoolODE data at {SYN_ROOT}.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    max_rep = 2 if args.quick else args.max_replicates

    rows = []
    for net in ACYCLIC + CYCLIC:
        base = SYN_ROOT / net / f"{net}-{args.cells}"
        if not base.exists():
            continue
        reps = sorted(p.name for p in base.glob(f"{net}-{args.cells}-*") if p.is_dir())[:max_rep]
        for name in reps:
            try:
                r = score_one(base, name)
            except Exception:
                r = None
            if r is not None:
                rows.append(r)
    if not rows:
        raise SystemExit("No datasets scored.")
    df = pd.DataFrame(rows)
    df["regime"] = np.where(df["net_type"].isin(ACYCLIC), "acyclic", "cyclic")
    df.to_csv(TABLES_DIR / f"{PREFIX}_all.csv", index=False)

    metrics = ["aupr_symmetric_corr", "aupr_nongaussian_directed", "aupr_genie3"]
    overall = df[metrics].mean()
    by_regime = df.groupby("regime")[metrics].mean().reset_index()

    lines = ["# Experiment 35: non-Gaussian orientation from static data + detectability\n",
             f"BoolODE, {len(df)} datasets ({args.cells}-cell), exact network truth. Directed AUPR; "
             f"symmetric correlation cannot orient by construction.\n",
             f"- mean non-Gaussianity (abs excess kurtosis) of the data: {fmt(float(df['nongaussianity'].mean()))} "
             f"(0 = Gaussian; LiNGAM needs this above 0).\n",
             "## Part A: directed recovery (AUPR vs true network)\n",
             to_markdown_table(pd.DataFrame([{m: float(overall[m]) for m in metrics}])),
             "\n### By regime (LiNGAM assumes acyclicity)\n",
             to_markdown_table(by_regime),
             "\n## Part B: detectability (per-edge z vs permutation null)\n",
             to_markdown_table(pd.DataFrame([{
                 "z_true_edges": float(df["z_true_edges"].mean()),
                 "z_false_edges": float(df["z_false_edges"].mean()),
                 "detectable_true_rate_z>2": float(df["detectable_true_rate"].mean()),
             }]))]

    sym = float(overall["aupr_symmetric_corr"]); ng = float(overall["aupr_nongaussian_directed"])
    g3 = float(overall["aupr_genie3"])
    ng_acyc = float(by_regime[by_regime["regime"] == "acyclic"]["aupr_nongaussian_directed"].iloc[0]) if (by_regime["regime"] == "acyclic").any() else float("nan")
    sym_acyc = float(by_regime[by_regime["regime"] == "acyclic"]["aupr_symmetric_corr"].iloc[0]) if (by_regime["regime"] == "acyclic").any() else float("nan")
    lines.append("\n## Verdict\n")
    lines.append(f"- non-Gaussian orientation directed AUPR {fmt(ng)} vs symmetric correlation {fmt(sym)} "
                 f"and GENIE3 {fmt(g3)}.")
    lines.append(f"- on ACYCLIC networks (where LiNGAM applies): non-Gaussian {fmt(ng_acyc)} vs symmetric "
                 f"{fmt(sym_acyc)}.")
    lines.append(f"- non-Gaussian orientation {'beats' if ng > sym + 0.02 else 'does not beat'} the symmetric "
                 f"baseline overall; {'beats' if ng_acyc > sym_acyc + 0.02 else 'does not beat'} it on acyclic nets.")
    lines.append(f"- detectability: true edges sit at z {fmt(float(df['z_true_edges'].mean()))} vs false edges "
                 f"{fmt(float(df['z_false_edges'].mean()))}; {fmt(float(df['detectable_true_rate'].mean()))} of true "
                 f"edges clear z>2 (the per-edge SNR map).")

    pd.DataFrame([{
        "n_datasets": len(df), "nongaussianity": float(df["nongaussianity"].mean()),
        "aupr_symmetric": sym, "aupr_nongaussian": ng, "aupr_genie3": g3,
        "aupr_nongaussian_acyclic": ng_acyc, "aupr_symmetric_acyclic": sym_acyc,
        "z_true": float(df["z_true_edges"].mean()), "z_false": float(df["z_false_edges"].mean()),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
