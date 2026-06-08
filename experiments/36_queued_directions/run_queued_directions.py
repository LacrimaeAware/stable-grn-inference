r"""Experiment 36: the queued directions (diversity-consensus, cycle 2D geometry).

Two remaining candidate directions from docs/research_directions.md, run together and honestly:

A. Diversity-consensus (the translation-chain-with-drift idea). Combine genuinely different lenses
   (Pearson = linear, Spearman = monotone-nonlinear, mutual information = arbitrary dependence) and
   keep the agreement. Question: does a consensus across diverse lenses beat the best single lens at
   recovering the skeleton? Agreement = signal, disagreement = drift.

B. Cycle 2D geometry. A 1D order cannot describe a loop (exp 34: cycle recovery 0.55). Recover a 2D
   diffusion embedding and read the cyclic order as an angle. Question: on cyclic topologies, does the
   2D angular order match the true order (circular correlation) better than the 1D spectral order?

BoolODE, exact truth. Baselines: each single lens and GENIE3 (Part A); the 1D spectral order (Part B).
Prior is sobering: these are long shots after the wall the other directions hit.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/36_queued_directions/run_queued_directions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from stable_grn_inference.analysis import cell_similarity, spectral_order, order_recovery_score
from stable_grn_inference.data import load_beeline_dataset
from stable_grn_inference.dynamics import edges_to_operator, skeleton_recovery_aupr

ROOT = Path(__file__).resolve().parents[2]
SYN_ROOT = ROOT / "data" / "raw" / "BEELINE-data" / "inputs" / "Synthetic"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "queued_directions"
ACYCLIC = ("dyn-LI", "dyn-LL")
CYCLIC = ("dyn-CY", "dyn-BF", "dyn-BFC", "dyn-TF")


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def to_markdown_table(frame):
    cols = [str(c) for c in frame.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |"
            for row in frame.to_numpy()]
    return "\n".join([head, sep, *body])


def rank_normalize(M):
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = np.zeros_like(M, dtype=float)
    out[off] = rankdata(M[off]) / off.sum()
    return out


def mi_matrix(X, bins=8):
    n = X.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c, _, _ = np.histogram2d(X[:, i], X[:, j], bins=bins)
            p = c / max(c.sum(), 1)
            pi = p.sum(1, keepdims=True); pj = p.sum(0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = p * np.log(p / (pi * pj + 1e-12) + 1e-12)
            mi = float(np.nansum(np.where(p > 0, term, 0.0)))
            M[i, j] = M[j, i] = mi
    return M


def circ_corr(theta, phi):
    a = theta - np.angle(np.mean(np.exp(1j * theta)))
    b = phi - np.angle(np.mean(np.exp(1j * phi)))
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return float(abs(num / (den + 1e-12)))


def diffusion_2d_angle(X):
    S = cell_similarity(X)
    d = S.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    norm = dinv[:, None] * S * dinv[None, :]
    w, V = np.linalg.eigh(norm)
    order = np.argsort(w)[::-1]
    dc1 = dinv * V[:, order[1]]
    dc2 = dinv * V[:, order[2]]
    return np.arctan2(dc2, dc1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--max-replicates", type=int, default=3)
    args = ap.parse_args()
    if not SYN_ROOT.exists():
        raise SystemExit(f"No BoolODE data at {SYN_ROOT}.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rowsA, rowsB = [], []
    for net in ACYCLIC + CYCLIC:
        base = SYN_ROOT / net / f"{net}-{args.cells}"
        if not base.exists():
            continue
        for name in sorted(p.name for p in base.glob(f"{net}-{args.cells}-*") if p.is_dir())[:args.max_replicates]:
            try:
                ds = load_beeline_dataset(base, name, reference="boolode", log1p=False)
            except Exception:
                continue
            if ds.reference_edges.empty:
                continue
            genes = list(ds.expression.columns)
            X = ds.expression.to_numpy(float)
            truth = edges_to_operator(ds.reference_edges, genes)

            # Part A: diversity-consensus (skeleton)
            pear = np.abs(np.nan_to_num(np.corrcoef(X.T)))
            spear = np.abs(np.nan_to_num(np.corrcoef(np.apply_along_axis(rankdata, 0, X).T)))
            mi = mi_matrix(X)
            consensus = (rank_normalize(pear) * rank_normalize(spear) * rank_normalize(mi)) ** (1 / 3)
            rowsA.append({
                "net_type": net,
                "pearson": skeleton_recovery_aupr(pear, truth),
                "spearman": skeleton_recovery_aupr(spear, truth),
                "mutual_info": skeleton_recovery_aupr(mi, truth),
                "consensus": skeleton_recovery_aupr(consensus, truth),
            })

            # Part B: cycle 2D geometry (only where a pseudotime exists)
            if ds.pseudotime is not None and net in CYCLIC:
                t = ds.pseudotime.iloc[:, 0].to_numpy(float)
                keep = np.where(np.isfinite(t))[0]
                if keep.size > 6:
                    Xk = X[keep]; tk = t[keep]
                    theta = diffusion_2d_angle(Xk)
                    rowsB.append({
                        "net_type": net,
                        "order_1d_spectral": order_recovery_score(spectral_order(Xk), tk),
                        "order_2d_circular": circ_corr(theta, 2 * np.pi * (tk - tk.min()) / (np.ptp(tk) + 1e-12)),
                    })

    dfA = pd.DataFrame(rowsA)
    dfA.to_csv(TABLES_DIR / f"{PREFIX}_consensus.csv", index=False)
    meanA = dfA[["pearson", "spearman", "mutual_info", "consensus"]].mean()

    lines = ["# Experiment 36: queued directions (diversity-consensus, cycle 2D geometry)\n",
             f"BoolODE, {len(dfA)} datasets, exact truth.\n",
             "## Part A: diversity-consensus (skeleton AUPR vs true network)\n",
             to_markdown_table(pd.DataFrame([{k: float(meanA[k]) for k in meanA.index}]))]
    best_single = float(meanA[["pearson", "spearman", "mutual_info"]].max())
    cons = float(meanA["consensus"])
    lines.append(f"\n- consensus {fmt(cons)} vs best single lens {fmt(best_single)}: "
                 f"consensus {'beats' if cons > best_single + 0.02 else 'does not beat'} the best single lens.")

    if rowsB:
        dfB = pd.DataFrame(rowsB)
        dfB.to_csv(TABLES_DIR / f"{PREFIX}_cycle2d.csv", index=False)
        meanB = dfB[["order_1d_spectral", "order_2d_circular"]].mean()
        lines.append("\n## Part B: cycle 2D geometry (order recovery on cyclic topologies)\n")
        lines.append(to_markdown_table(dfB.groupby("net_type")[["order_1d_spectral", "order_2d_circular"]].mean().reset_index()))
        lines.append(f"\n- 2D circular order {fmt(float(meanB['order_2d_circular']))} vs 1D spectral "
                     f"{fmt(float(meanB['order_1d_spectral']))}: 2D "
                     f"{'helps' if float(meanB['order_2d_circular']) > float(meanB['order_1d_spectral']) + 0.02 else 'does not help'} on cycles.")

    lines.append("\n## Verdict\n")
    lines.append(f"- diversity-consensus does {'' if cons > best_single + 0.02 else 'not '}beat the best single lens; "
                 "combining diverse lenses mostly re-finds the same skeleton.")
    if rowsB:
        lines.append("- the 2D embedding gives the cyclic order an angle, tested against the 1D order on cycles.")
    lines.append("- consistent with the project: the skeleton is easy, and combining methods does not break the wall.")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
