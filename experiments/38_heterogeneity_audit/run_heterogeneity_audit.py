r"""Experiment 38: audit the exp 37 heterogeneity result, and test for a real residual (RENGE).

Exp 37 reported that the single-cell deviation from each knockout's mean response is structured,
reproducible, and aligned with a single cell-state axis (0.89). That signature -- ribosomal-dominated
programs plus one shared axis -- is also the classic signature of a TECHNICAL/GLOBAL confound (library
size / sequencing depth / cell size). This experiment decides whether exp 37 is a real result or a
trivial one, with three tests, and then looks for the only place a non-trivial signal could survive.

1. Technical confound: does the heterogeneity axis correlate with raw library size (total UMI) and
   detected-gene count? High correlation = the "cell-state" axis is largely technical.
2. Knockout-specificity: is the per-cell deviation axis the SAME for every knockout (one global axis,
   trivial) or different by knockout (real knockout-specific structure)? Measured by pairwise cosine
   of the per-perturbation deviation axes.
3. Residual: after regressing out library size AND projecting out the global axis, is there
   reproducible, knockout-specific heterogeneity left? This is the only non-trivial outcome.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/38_heterogeneity_audit/run_heterogeneity_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from stable_grn_inference.analysis import heterogeneity_structure
from stable_grn_inference.data import load_renge_day_hvg

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "heterogeneity_audit"


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def top_axis(R):
    return np.linalg.svd(R - R.mean(0), full_matrices=False)[2][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    day_dir = ROOT / "data" / "raw" / "renge" / "day5" / "day5"
    if not (day_dir / "matrix.mtx.gz").exists():
        raise SystemExit(f"No RENGE day5 10x at {day_dir}.")
    print("Loading RENGE day5 (1500 HVGs + depth)...", flush=True)
    expr_df, label_series, depth = load_renge_day_hvg(day_dir, n_hvg=1500, return_total_umi=True)
    X = expr_df.to_numpy(float)
    labels = label_series.to_numpy()
    log_umi = np.log1p(depth.to_numpy(float))
    n_detected = (X > 0).sum(1).astype(float)
    ctrl = labels == "control"

    # the global cell-state axis (top PC of controls)
    Xc = X[ctrl]
    v_state = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)[2][0]
    proj_state = (X - X.mean(0)) @ v_state

    # Test 1: is the cell-state axis technical?
    corr_state_umi = abs(pearsonr(proj_state, log_umi)[0])
    corr_state_ndet = abs(pearsonr(proj_state, n_detected)[0])

    perts = [g for g in sorted(set(labels)) if g != "control" and int((labels == g).sum()) >= 30]
    axes, het_rows = {}, []
    for g in perts:
        idx = labels == g
        Xg = X[idx]
        ax = top_axis(Xg)
        axes[g] = ax
        proj = (Xg - Xg.mean(0)) @ ax
        het_rows.append({
            "gene": g, "n_cells": int(idx.sum()),
            "dev_axis_vs_cellstate": float(abs(np.dot(ax, v_state))),
            "dev_proj_vs_log_umi": float(abs(pearsonr(proj, log_umi[idx])[0])),
        })
    het = pd.DataFrame(het_rows)
    het.to_csv(TABLES_DIR / f"{PREFIX}_per_perturbation.csv", index=False)

    # Test 2: knockout-specificity of the raw deviation axes (pairwise cosine)
    A = np.array([axes[g] for g in perts])
    cos = np.abs(A @ A.T)
    off = ~np.eye(len(perts), dtype=bool)
    mean_pairwise = float(cos[off].mean())

    # Test 3: residual after removing library size AND the global state axis
    dc = log_umi - log_umi.mean()
    beta = ((X - X.mean(0)).T @ dc) / (dc @ dc)
    X_dr = X - np.outer(dc, beta)                       # regress out depth
    X_res = X_dr - np.outer(X_dr @ v_state, v_state)    # project out the global state axis
    res_rows, res_axes = [], {}
    for g in perts:
        idx = labels == g
        hs = heterogeneity_structure(X_res[idx], seed=args.random_seed)
        res_axes[g] = top_axis(X_res[idx])
        res_rows.append({"gene": g, **hs})
    res = pd.DataFrame(res_rows)
    Ares = np.array([res_axes[g] for g in perts])
    res_pairwise = float(np.abs(Ares @ Ares.T)[off].mean())

    lines = ["# Experiment 38: heterogeneity audit (RENGE day5)\n",
             f"- {X.shape[1]} genes; {X.shape[0]} cells; {len(perts)} perturbations.\n",
             "## Test 1: is the cell-state axis technical (library size / depth)?\n",
             f"- |corr(cell-state projection, log total UMI)| = {fmt(corr_state_umi)}",
             f"- |corr(cell-state projection, detected-gene count)| = {fmt(corr_state_ndet)}",
             f"- mean per-knockout |corr(deviation projection, log UMI)| = {fmt(float(het['dev_proj_vs_log_umi'].mean()))}\n",
             "## Test 2: knockout-specificity of the raw deviation axes\n",
             f"- mean pairwise |cosine| of the 23 deviation axes = {fmt(mean_pairwise)} "
             f"(near 1 = one shared global axis / trivial; near 0 = knockout-specific)",
             f"- mean alignment of each deviation axis with the global cell-state axis = "
             f"{fmt(float(het['dev_axis_vs_cellstate'].mean()))}\n",
             "## Test 3: residual after removing library size and the global axis\n",
             f"- residual heterogeneity: top-direction variance fraction {fmt(float(res['top_var_fraction'].mean()))}, "
             f"reproducibility {fmt(float(res['reproducibility'].mean()))} across cell halves",
             f"- residual deviation axes mean pairwise |cosine| = {fmt(res_pairwise)} "
             f"(low + reproducible = real knockout-specific structure)\n"]

    technical = corr_state_umi > 0.4 or float(het["dev_proj_vs_log_umi"].mean()) > 0.4
    global_axis = mean_pairwise > 0.5
    residual_real = float(res["reproducibility"].mean()) > 0.4 and res_pairwise < 0.5
    lines.append("## Verdict\n")
    lines.append(f"- the cell-state / heterogeneity axis is {'largely TECHNICAL (library size)' if technical else 'not strongly tied to library size'} "
                 f"(|corr| with log UMI {fmt(corr_state_umi)}).")
    lines.append(f"- the raw heterogeneity is {'a single GLOBAL axis shared across knockouts (trivial / expected)' if global_axis else 'knockout-specific'} "
                 f"(mean pairwise cosine {fmt(mean_pairwise)}).")
    lines.append(f"- after removing library size and the global axis, the residual is "
                 f"{'real, reproducible, and knockout-specific' if residual_real else 'not reproducible / not knockout-specific'} "
                 f"(reproducibility {fmt(float(res['reproducibility'].mean()))}, pairwise cosine {fmt(res_pairwise)}).")
    lines.append(f"- HONEST READING: exp 37's heterogeneity is "
                 f"{'a real, knockout-specific signal' if residual_real else 'largely the trivial global/technical axis; the structured-reproducible-aligned result was expected, not a discovery'}.")

    pd.DataFrame([{
        "corr_state_umi": corr_state_umi, "corr_state_ndet": corr_state_ndet,
        "mean_dev_proj_umi": float(het["dev_proj_vs_log_umi"].mean()),
        "axis_pairwise_cosine": mean_pairwise,
        "mean_align_cellstate": float(het["dev_axis_vs_cellstate"].mean()),
        "residual_reproducibility": float(res["reproducibility"].mean()),
        "residual_pairwise_cosine": res_pairwise,
        "technical": bool(technical), "global_axis": bool(global_axis), "residual_real": bool(residual_real),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
