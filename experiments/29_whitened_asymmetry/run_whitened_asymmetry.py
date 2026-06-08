r"""Experiment 29: whitened interventional-asymmetry gate (RPE1; Direction A).

The one non-circular pairwise question left on RPE1: after accounting for the two
reproducible per-gene axes (net_out cascade position, response magnitude), is there any
reproducible orientation asymmetry left in the response matrix, and does WHITENING the
dominant mode (downweighting it, not subtracting it) recover more of it than the raw
asymmetry. This is the honest resolution of the pairwise-difference intuition, scoped to
the object that can actually carry direction: A = |M| - |M|^T on the square response block.

Gate 0 (this script, cheap, no external data): split-half reproducibility of the RESIDUAL
asymmetry (raw A minus its net_out and magnitude fit) across independent cell halves, swept
over whitening strength alpha in [0, 1]. If the residual is not reproducible at any alpha,
stop: there is no stable pairwise signal beyond the known per-gene axes, and no external
anchor is worth loading. Expected outcome on RPE1 (per exp 28): a clean negative, because
RPE1 sits below the SNR floor; whitening fixes dominance, not SNR.

Gate 1 (NOT run here, the next step, gated on Gate 0): validate the surviving asymmetry
against external anchors (DepMap gene-effect for severity, CORUM/STRING co-membership for
the relational part), held out by gene. That is where reproducibility becomes correctness.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/29_whitened_asymmetry/run_whitened_asymmetry.py
  # --synthetic forces the offline DAG fixture (positive control); --quick caps the gene set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_grn_inference.analysis import (
    fractional_whiten,
    net_out,
    pairwise_reproducibility,
    residualize_asymmetry,
    response_asymmetry,
    response_magnitude,
)
from stable_grn_inference.data import (
    load_interventional_frames,
    load_replogle_raw_h5ad,
    make_synthetic_interventional,
    perturbation_response_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "whitened_asymmetry"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def load_dataset(args):
    """Real RPE1 if present and not forced synthetic, else the offline DAG fixture."""
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if args.synthetic or raw_path is None:
        expr, labels, true_edges = make_synthetic_interventional(
            n_genes=40, n_cells_per_condition=60, edge_density=0.18, seed=args.random_seed
        )
        ds = load_interventional_frames("synthetic_rpe1", expr, labels, reference_edges=true_edges)
        return ds, "synthetic DAG fixture (offline positive control)"
    max_perts = 200 if args.quick else None
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    return ds, raw_path.name


def whitening_sweep(M1, M2):
    """For each whitening strength, reproducibility of the asymmetry and of the residual
    asymmetry (after removing the per-gene net_out and magnitude fit) across cell halves."""
    rows = []
    for alpha in ALPHAS:
        W1, W2 = fractional_whiten(M1, alpha), fractional_whiten(M2, alpha)
        A1, A2 = response_asymmetry(W1), response_asymmetry(W2)
        R1, _ = residualize_asymmetry(A1, [net_out(W1), response_magnitude(W1)])
        R2, _ = residualize_asymmetry(A2, [net_out(W2), response_magnitude(W2)])
        rows.append({
            "alpha": alpha,
            "repro_asymmetry": pairwise_reproducibility(A1, A2),
            "repro_residual": pairwise_reproducibility(R1, R2),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="force the offline DAG fixture")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ds, source = load_dataset(args)
    P = list(ds.perturbed_genes)
    Dfull, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
    M1 = Da.loc[P, P].to_numpy(float)
    M2 = Db.loc[P, P].to_numpy(float)
    n = len(P)

    sweep = whitening_sweep(M1, M2)
    # standing per-gene bars (the known reproducible axes)
    repro_netout = float(spearmanr(net_out(M1), net_out(M2)).statistic)
    repro_magnitude = float(spearmanr(response_magnitude(M1), response_magnitude(M2)).statistic)

    raw = sweep[sweep["alpha"] == 0.0].iloc[0]
    best = sweep.loc[sweep["repro_residual"].idxmax()]
    raw_resid = float(raw["repro_residual"])
    best_resid = float(best["repro_residual"])
    best_alpha = float(best["alpha"])
    gate0_pass = np.isfinite(best_resid) and best_resid > 0.10
    whitening_helps = best_alpha > 0.0 and best_resid > raw_resid + 0.05

    lines = ["# Experiment 29: whitened interventional-asymmetry gate (RPE1)\n"]
    lines.append(f"- source: {source}; genes {n}\n")
    lines.append("## Standing per-gene axes (the known reproducible structure)\n")
    lines.append(f"- net_out (cascade position) split-half reproducibility: Spearman {fmt(repro_netout)}")
    lines.append(f"- magnitude (severity) split-half reproducibility: Spearman {fmt(repro_magnitude)}")
    lines.append(f"- raw asymmetry |M|-|M|^T split-half reproducibility: Spearman {fmt(float(raw['repro_asymmetry']))}\n")

    lines.append("## Gate 0: whitening sweep (reproducibility of the residual asymmetry)\n")
    lines.append("Residual = asymmetry after removing the net_out and magnitude per-gene fit. "
                 "This is the pairwise signal NOT recoverable from the two known axes.\n")
    lines.append("| alpha (whitening) | repro asymmetry | repro residual asymmetry |")
    lines.append("| --- | --- | --- |")
    for _, r in sweep.iterrows():
        lines.append(f"| {fmt(r['alpha'], 2)} | {fmt(r['repro_asymmetry'])} | {fmt(r['repro_residual'])} |")

    lines.append("\n## Verdict\n")
    lines.append(f"- residual asymmetry reproducibility: raw {fmt(raw_resid)}, "
                 f"best {fmt(best_resid)} at alpha {fmt(best_alpha, 2)}.")
    lines.append(f"- whitening {'helps' if whitening_helps else 'does not help'} "
                 f"(best alpha {fmt(best_alpha, 2)} vs raw alpha 0.00).")
    if gate0_pass:
        lines.append("- GATE 0 PASS: a reproducible residual asymmetry exists beyond net_out and "
                     "magnitude. Proceed to Gate 1 (external anchors: DepMap, CORUM/STRING), held "
                     "out by gene. Gate 1 is the next experiment and is not run here.")
    else:
        lines.append("- GATE 0 FAIL: no reproducible pairwise asymmetry survives beyond the per-gene "
                     "net_out and magnitude axes. The pairwise/whitening line stops here; the "
                     "reproducible structure is the per-gene axes (exp 26), not a richer pairwise "
                     "object. This is the expected outcome on RPE1 per the exp 28 SNR floor.")
    lines.append("\n- reproducibility is consistency across cell halves, not correctness; "
                 "correctness needs Gate 1's external anchors. No external ground truth is used here.")

    sweep.to_csv(TABLES_DIR / f"{PREFIX}_sweep.csv", index=False)
    pd.DataFrame([{
        "source": source, "n_genes": n,
        "repro_netout": repro_netout, "repro_magnitude": repro_magnitude,
        "repro_asymmetry_raw": float(raw["repro_asymmetry"]),
        "repro_residual_raw": raw_resid, "repro_residual_best": best_resid,
        "best_alpha": best_alpha, "gate0_pass": bool(gate0_pass),
        "whitening_helps": bool(whitening_helps),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
