r"""Experiment 27: do cascade-adjacent gene pairs give more direct edges? (RPE1)

Experiment 26 produced a reproducible cascade ordering (net_out: how much perturbing a gene
moves others minus how much they move it). Hypothesis: a direct edge A->B connects genes that
are ADJACENT in that ordering, while the cascade connects DISTANT genes (far-upstream to
far-downstream). If so, ordering-adjacent pairs should be:
  - less explainable as a chain through an intermediate gene (more direct), and
  - more reproducible as edges across independent cell halves.

Tests (no external ground truth):
  1. ordering distance vs mediation: for each interacting pair, mediation_ratio =
     (strongest 2-step path through any middle gene) / |D[A,B]|. High ratio = the pair is
     explained by a chain (indirect). Does mediation_ratio increase with ordering distance?
  2. reproducibility: compute the ordering and an ordering-local, direction-consistent edge
     score on each cell half independently; compare the top-edge overlap to the overlap of a
     raw |D| ranking and a raw correlation ranking.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/27_causalbench_local_edges/run_local_edges.py
  # --quick caps the gene set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_grn_inference.data import load_replogle_raw_h5ad, perturbation_response_matrix

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "causalbench_local_edges"


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def net_out_rank(D):
    A = np.abs(D)
    score = (A.sum(1) - A.sum(0)) / (A.shape[0] - 1)   # high = upstream
    return score.argsort()[::-1].argsort()             # rank 0 = most upstream


def best_two_step(A):
    best = np.zeros_like(A)
    for c in range(A.shape[0]):
        best = np.maximum(best, np.minimum(A[:, c][:, None], A[c, :][None, :]))
    return best


def control_null_threshold(dataset, *, n_splits=10, seed=0, pct=95.0):
    X = dataset.expression.to_numpy(dtype=float)
    ctrl = np.where(dataset.is_control.to_numpy())[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        sel = rng.permutation(ctrl)
        h = sel.size // 2
        vals.append(np.abs(X[sel[:h]].mean(0) - X[sel[h:2 * h]].mean(0)))
    return float(np.percentile(np.concatenate(vals), pct))


def local_edge_topk(D, k, *, local_frac):
    """Top-k edges restricted to ordering-adjacent, upstream->downstream pairs."""
    n = D.shape[0]
    rank = net_out_rank(D)
    A = np.abs(D).copy()
    np.fill_diagonal(A, 0.0)
    span = max(1, int(local_frac * n))
    keep = np.zeros((n, n), dtype=bool)
    ri = rank[:, None]; rj = rank[None, :]
    keep = (rj > ri) & (rj - ri <= span)              # j just downstream of i
    score = np.where(keep, A, 0.0)
    idx = np.argsort(score, axis=None)[::-1][:k]
    return set(map(int, idx))


def raw_topk(M, k):
    A = np.abs(M).copy(); np.fill_diagonal(A, 0.0)
    return set(map(int, np.argsort(A, axis=None)[::-1][:k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--local-frac", type=float, default=0.05)
    args = ap.parse_args()
    max_perts = 200 if args.quick else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if raw_path is None:
        raise SystemExit(f"No raw RPE1 h5ad in {CB_DIR}.")
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    P = list(ds.perturbed_genes)
    Dfull, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
    D = Dfull.loc[P, P].to_numpy(float)
    D1 = Da.loc[P, P].to_numpy(float)
    D2 = Db.loc[P, P].to_numpy(float)
    n = len(P)
    thr = control_null_threshold(ds, seed=args.random_seed)

    rank = net_out_rank(D)
    A = np.abs(D)
    best = best_two_step(A)

    # Part 1: ordering distance vs mediation, over interacting off-diagonal pairs
    dist, med, mag = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j or A[i, j] <= thr:
                continue
            dist.append(abs(rank[i] - rank[j]))
            med.append(best[i, j] / (A[i, j] + 1e-9))
            mag.append(A[i, j])
    dist, med, mag = np.array(dist), np.array(med), np.array(mag)
    rho_dist_med = spearmanr(dist, med).statistic

    lines = ["# Experiment 27: cascade-adjacent edges (RPE1)\n"]
    lines.append(f"- genes {n}; interacting pairs {len(dist)}; control-null threshold {fmt(thr)}\n")
    lines.append("## Part 1: ordering distance vs mediation\n")
    lines.append(f"- Spearman(ordering distance, mediation ratio) = {fmt(rho_dist_med)} "
                 f"(positive = distant pairs are more chain-explained, supporting the hypothesis)")
    # quartile comparison
    qs = np.quantile(dist, [0.25, 0.5, 0.75])
    near = med[dist <= qs[0]]; far = med[dist >= qs[2]]
    lines.append(f"- mean mediation ratio: nearest-quartile pairs {fmt(near.mean())} vs "
                 f"farthest-quartile {fmt(far.mean())} (lower = more direct)")
    lines.append(f"- fraction of pairs where direct effect exceeds best chain (ratio<1): "
                 f"near {fmt(np.mean(near < 1))}, far {fmt(np.mean(far < 1))}\n")

    # Part 2: reproducibility of local edges vs raw rankings
    k = 200
    ctrl = ds.expression.loc[ds.is_control.values, P].to_numpy(float)
    half = len(ctrl) // 2
    C1 = np.corrcoef(ctrl[:half].T); C2 = np.corrcoef(ctrl[half:].T)
    local1 = local_edge_topk(D1, k, local_frac=args.local_frac)
    local2 = local_edge_topk(D2, k, local_frac=args.local_frac)
    rawD1, rawD2 = raw_topk(D1, k), raw_topk(D2, k)
    corr1, corr2 = raw_topk(C1, k), raw_topk(C2, k)
    lines.append("## Part 2: split-half top-edge reproducibility (overlap of top-200)\n")
    lines.append("| edge score | top-200 overlap across halves |")
    lines.append("| --- | --- |")
    lines.append(f"| raw |D| (total effect) | {fmt(len(rawD1 & rawD2) / k)} |")
    lines.append(f"| observational correlation | {fmt(len(corr1 & corr2) / k)} |")
    lines.append(f"| cascade-local |D| (ordering-adjacent) | {fmt(len(local1 & local2) / k)} |")

    # top local edges (from full data)
    locfull = local_edge_topk(D, 25, local_frac=args.local_frac)
    pairs = [(P[idx // n], P[idx % n]) for idx in sorted(locfull, key=lambda x: -A.flatten()[x])]
    lines.append("\n- top cascade-local edges (source -> target): "
                 + ", ".join(f"{s}->{t}" for s, t in pairs[:15]))

    lines.append("\n## Summary\n")
    supports = rho_dist_med > 0.1 and near.mean() < far.mean()
    lines.append(f"- {'ordering distance relates to mediation as hypothesized' if supports else 'ordering distance shows little relation to mediation'} "
                 f"(Spearman {fmt(rho_dist_med)}).")
    repro_local = len(local1 & local2) / k
    repro_raw = len(rawD1 & rawD2) / k
    lines.append(f"- cascade-local edges are {'more' if repro_local > repro_raw else 'not more'} "
                 f"reproducible than raw |D| ({fmt(repro_local)} vs {fmt(repro_raw)}).")
    lines.append("- no external ground truth is used; reproducibility and mediation are internal checks.")

    pd.DataFrame([{
        "n_genes": n, "spearman_dist_mediation": rho_dist_med,
        "mean_mediation_near": float(near.mean()), "mean_mediation_far": float(far.mean()),
        "repro_local": repro_local, "repro_rawD": repro_raw,
        "repro_correlation": len(corr1 & corr2) / k,
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
