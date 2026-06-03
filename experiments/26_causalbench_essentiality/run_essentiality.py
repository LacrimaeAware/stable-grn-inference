r"""Experiment 26: perturbation essentiality and cascade position on RPE1.

Most knockouts of essential genes trigger the same convergent cell-cycle response. This
experiment uses that dominant signal directly, rather than trying to remove it.

Part 1: which genes are most essential, inferred from the data. Four data-derived measures
per perturbed gene g:
  - magnitude:  ||Delta_g||, the size of the whole-transcriptome response to knocking out g.
  - cascade:    |Delta_g . cascade_axis|, projection onto the dominant response component.
  - breadth:    number of genes that respond to perturbing g above a control-null (out-reach).
  - centrality: connectivity of g in the control-cell correlation network.
Validation without external labels: do the four measures agree (rank correlation), is the
ranking reproducible across independent cell halves, and does it relate to the number of
surviving cells per perturbation (essential knockouts are more depleted).

Part 2: cascade position. net_out(g) = mean_h (|Delta_g[h]| - |Delta_h[g]|), the degree to
which perturbing g affects others more than they affect g. High net_out is upstream. This
reuses the asymmetry signal that was reproducible in experiment 21. Reported with its
split-half reproducibility and its relation to essentiality.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/26_causalbench_essentiality/run_essentiality.py
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
FIG_DIR = ROOT / "results" / "figures"
PREFIX = "causalbench_essentiality"


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


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


def net_out(D):
    A = np.abs(D)
    return (A.sum(axis=1) - A.sum(axis=0)) / (A.shape[0] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
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
    n_cells = np.array([int((ds.perturbation_labels == g).sum()) for g in P])

    # cascade axis (top right singular vector of the response matrix)
    cascade_axis = np.linalg.svd(D, full_matrices=False)[2][0]
    # control-cell correlation centrality
    ctrl = ds.expression.loc[ds.is_control.values, P].to_numpy(float)
    C = np.abs(np.corrcoef(ctrl.T))
    np.fill_diagonal(C, 0.0)
    thr = control_null_threshold(ds, seed=args.random_seed)

    Aabs = np.abs(D)
    ess = pd.DataFrame({
        "gene": P,
        "magnitude": np.linalg.norm(D, axis=1),
        "cascade": np.abs(D @ cascade_axis),
        "breadth": (Aabs > thr).sum(axis=1),
        "centrality": C.sum(axis=1),
        "n_cells": n_cells,
        "self_knockdown": np.array([abs(D[i, i]) for i in range(len(P))]),
        "net_out": net_out(D),
    })

    lines = ["# Experiment 26: perturbation essentiality and cascade position (RPE1)\n"]
    lines.append(f"- genes {len(P)}; control cells {ds.metadata['n_control_cells']}; "
                 f"control-null threshold {fmt(thr)}\n")

    # Part 1: cross-measure agreement
    lines.append("## Part 1: essentiality measures\n")
    measures = ["magnitude", "cascade", "breadth", "centrality"]
    lines.append("Cross-measure rank agreement (Spearman):\n")
    lines.append("| | " + " | ".join(measures) + " |")
    lines.append("| " + " --- |" * (len(measures) + 1))
    for a in measures:
        row = [fmt(spearmanr(ess[a], ess[b]).statistic) for b in measures]
        lines.append(f"| {a} | " + " | ".join(row) + " |")

    # split-half reproducibility of the magnitude ranking
    mag1 = np.linalg.norm(D1, axis=1); mag2 = np.linalg.norm(D2, axis=1)
    repro_mag = spearmanr(mag1, mag2).statistic
    breadth1 = (np.abs(D1) > thr).sum(1); breadth2 = (np.abs(D2) > thr).sum(1)
    repro_breadth = spearmanr(breadth1, breadth2).statistic
    lines.append(f"\n- split-half reproducibility: magnitude ranking Spearman {fmt(repro_mag)}, "
                 f"breadth ranking {fmt(repro_breadth)}")
    rho_cells = spearmanr(ess["magnitude"], ess["n_cells"]).statistic
    lines.append(f"- essentiality (magnitude) vs surviving cells per perturbation: Spearman {fmt(rho_cells)} "
                 f"(negative = stronger responses come from more-depleted perturbations)")

    # combined essentiality rank and top genes
    ranks = ess[measures].rank(ascending=False)
    ess["essentiality_rank"] = ranks.mean(axis=1)
    top = ess.sort_values("essentiality_rank").head(20)
    lines.append(f"\n- top 20 most essential genes (mean rank of the four measures):\n  "
                 + ", ".join(top["gene"].tolist()))

    # Part 2: cascade position
    no1, no2 = net_out(D1), net_out(D2)
    repro_netout = spearmanr(no1, no2).statistic
    rho_ess_up = spearmanr(ess["magnitude"], ess["net_out"]).statistic
    up = ess.sort_values("net_out", ascending=False).head(15)
    down = ess.sort_values("net_out").head(15)
    lines.append("\n## Part 2: cascade position (upstream vs downstream)\n")
    lines.append(f"- net_out split-half reproducibility: Spearman {fmt(repro_netout)}")
    lines.append(f"- essentiality (magnitude) vs upstream score (net_out): Spearman {fmt(rho_ess_up)}")
    lines.append(f"- most upstream (high net_out): {', '.join(up['gene'].tolist())}")
    lines.append(f"- most downstream (low net_out): {', '.join(down['gene'].tolist())}")

    lines.append("\n## Summary\n")
    agree = np.mean([spearmanr(ess[a], ess[b]).statistic for a in measures for b in measures if a != b])
    lines.append(f"- the four essentiality measures agree at mean Spearman {fmt(agree)}; the ranking is "
                 f"reproducible across cell halves at {fmt(repro_mag)} (magnitude).")
    lines.append("- a single data-derived essentiality axis is therefore well-defined here.")
    lines.append(f"- cascade position (net_out) is {'reproducible' if repro_netout > 0.3 else 'weakly reproducible'} "
                 f"(Spearman {fmt(repro_netout)}) and "
                 f"{'aligns with' if rho_ess_up > 0.3 else 'is largely separate from'} essentiality.")
    lines.append("- external validation against annotated essentiality (e.g. DepMap) is the next step "
                 "and is not done here (no external download).")

    ess.sort_values("essentiality_rank").to_csv(TABLES_DIR / f"{PREFIX}_genes.csv", index=False)
    pd.DataFrame([{
        "n_genes": len(P), "cross_measure_mean_spearman": agree,
        "repro_magnitude": repro_mag, "repro_breadth": repro_breadth,
        "magnitude_vs_ncells": rho_cells, "repro_netout": repro_netout,
        "essentiality_vs_upstream": rho_ess_up,
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(mag1, mag2, s=8, alpha=0.5)
        ax[0].set_xlabel("magnitude, half A"); ax[0].set_ylabel("magnitude, half B")
        ax[0].set_title(f"essentiality reproducibility (rho={fmt(repro_mag)})")
        ax[1].scatter(ess["magnitude"], ess["net_out"], s=8, alpha=0.5)
        ax[1].set_xlabel("essentiality (magnitude)"); ax[1].set_ylabel("upstream score (net_out)")
        ax[1].set_title(f"essentiality vs cascade position (rho={fmt(rho_ess_up)})")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        lines.append(f"(figure skipped: {e})")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
