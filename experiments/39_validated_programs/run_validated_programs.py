r"""Experiment 39: depth-controlled, externally-validated gene programs (RENGE day 5).

Exp 38's lesson: internal reproducibility is not evidence of biology (library size is reproducible).
A program counts as real here only if it is (a) not driven by sequencing depth, (b) not just the
ribosomal/housekeeping axis, and (c) externally coherent (its genes are more STRING-connected than
random gene sets). This is the confound-proof version of program discovery.

Method: NMF programs on the RENGE day-5 high-variance genes; per program report the cell-loading
correlation with log library size (depth), the ribosomal fraction of its top genes, and a STRING
within-set connectivity z-score against a permutation null. STRING is fetched once for the gene set
and cached. A "real" program clears all three: depth corr < 0.3, ribosomal fraction < 0.3, STRING
z > 2.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/39_validated_programs/run_validated_programs.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.analysis import discover_programs, program_reproducibility
from stable_grn_inference.data import load_renge_day_hvg

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"
STRING_DIR = ROOT / "data" / "raw" / "string"
PREFIX = "validated_programs"
RIBO = ("RPL", "RPS", "RPLP", "MRPL", "MRPS", "RACK1")


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def fetch_string(genes, path):
    if path.exists():
        return pd.read_csv(path, sep="\t")
    STRING_DIR.mkdir(parents=True, exist_ok=True)
    ids = "%0d".join(genes)
    url = f"https://string-db.org/api/tsv/network?identifiers={ids}&species=9606"
    print(f"Fetching STRING for {len(genes)} genes...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    df = pd.read_csv(io.BytesIO(raw), sep="\t")
    df.to_csv(path, sep="\t", index=False)
    return df


def string_edges(df, genes):
    gs = set(genes)
    a = "preferredName_A" if "preferredName_A" in df.columns else df.columns[2]
    b = "preferredName_B" if "preferredName_B" in df.columns else df.columns[3]
    sc = "score" if "score" in df.columns else df.columns[-1]
    edges = set()
    for x, y, s in zip(df[a], df[b], df[sc]):
        if float(s) >= 0.4 and str(x) in gs and str(y) in gs and x != y:
            edges.add(frozenset((str(x), str(y))))
    return edges


def within_set_edges(gene_set, edges):
    gs = list(gene_set)
    return sum(1 for i in range(len(gs)) for j in range(i + 1, len(gs)) if frozenset((gs[i], gs[j])) in edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hvg", type=int, default=250)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    day_dir = ROOT / "data" / "raw" / "renge" / "day5" / "day5"
    if not (day_dir / "matrix.mtx.gz").exists():
        raise SystemExit(f"No RENGE day5 10x at {day_dir}.")
    print(f"Loading RENGE day5 ({args.n_hvg} HVGs)...", flush=True)
    expr_df, _, depth = load_renge_day_hvg(day_dir, n_hvg=args.n_hvg, return_total_umi=True)
    X = expr_df.to_numpy(float)
    genes = [str(g) for g in expr_df.columns]
    log_umi = np.log1p(depth.to_numpy(float))

    W, H = discover_programs(X, args.k, method="nmf", seed=args.random_seed)
    repro, _ = program_reproducibility(X, args.k, method="nmf", seed=args.random_seed)

    string_df = fetch_string(genes, STRING_DIR / f"renge_hvg{args.n_hvg}_string.tsv")
    edges = string_edges(string_df, genes)

    rng = np.random.default_rng(args.random_seed)
    rows = []
    for p in range(args.k):
        top_idx = np.argsort(H[p])[::-1][:args.top]
        top = [genes[i] for i in top_idx]
        obs = within_set_edges(top, edges)
        null = [within_set_edges([genes[i] for i in rng.choice(len(genes), args.top, replace=False)], edges)
                for _ in range(1000)]
        mu, sd = float(np.mean(null)), float(np.std(null) + 1e-9)
        ribo_frac = np.mean([any(t.startswith(r) for r in RIBO) for t in top])
        depth_corr = abs(np.corrcoef(W[:, p], log_umi)[0, 1])
        rows.append({
            "program": p, "depth_corr": float(depth_corr), "ribosomal_frac": float(ribo_frac),
            "string_obs_edges": obs, "string_z": (obs - mu) / sd,
            "top_genes": ", ".join(top[:6]),
        })
    df = pd.DataFrame(rows)
    df["real"] = (df["depth_corr"] < 0.3) & (df["ribosomal_frac"] < 0.3) & (df["string_z"] > 2.0)
    df.to_csv(TABLES_DIR / f"{PREFIX}_programs.csv", index=False)

    n_real = int(df["real"].sum())
    lines = [f"# Experiment 39: depth-controlled, externally-validated programs (RENGE day5)\n",
             f"- {len(genes)} genes; {X.shape[0]} cells; k={args.k} NMF programs; overall program "
             f"reproducibility {fmt(repro)}; {len(edges)} STRING edges among the genes.\n",
             "## Programs (depth correlation, ribosomal fraction, STRING enrichment z)\n",
             "| program | depth_corr | ribosomal_frac | string_z | real | top genes |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, r in df.sort_values("string_z", ascending=False).iterrows():
        lines.append(f"| {int(r['program'])} | {fmt(r['depth_corr'])} | {fmt(r['ribosomal_frac'])} | "
                     f"{fmt(r['string_z'])} | {'yes' if r['real'] else 'no'} | {r['top_genes']} |")
    lines.append(f"\n## Verdict\n")
    lines.append(f"- {n_real} of {args.k} programs are real by all three controls (depth corr < 0.3, "
                 f"ribosomal fraction < 0.3, STRING z > 2).")
    if n_real > 0:
        real = df[df["real"]].sort_values("string_z", ascending=False)
        lines.append(f"- real programs (top genes): " +
                     "; ".join(f"[{r['top_genes']}] (z={fmt(r['string_z'])})" for _, r in real.iterrows()))
        lines.append("- so after controlling for depth and excluding housekeeping, externally-coherent "
                     "gene programs DO exist in this data.")
    else:
        lines.append("- no program survives all three controls: the externally-coherent reproducible "
                     "structure is the housekeeping/depth axis, and nothing biological-and-specific "
                     "survives the confound controls. A clean negative.")

    pd.DataFrame([{"n_genes": len(genes), "k": args.k, "program_reproducibility": repro,
                   "n_string_edges": len(edges), "n_real_programs": n_real}]).to_csv(
        TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
