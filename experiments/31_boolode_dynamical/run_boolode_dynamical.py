r"""Experiment 31: dynamical recovery on BoolODE single-cell time-series (Direction B).

Experiment 30 showed a dynamic operator recovers directed structure that a static snapshot
cannot, on synthetic dynamics and DREAM4. This experiment runs the same test on BoolODE
single-cell data (BEELINE): real-ish single-cell expression with a pseudotime axis and an
EXACT generating network, with a built-in cell-count sweep (100, 200, 2000, 5000) that is the
sample-size / SNR axis of experiment 28.

For each network type (linear, long-linear, cycle, bifurcating, bifurcating-converging,
trifurcating) and cell count, cells are ordered along pseudotime into snapshot pairs, a dynamic
operator is fit, and its directed edges are graded against the exact ground truth, against the
static correlation baseline and the chance line. The questions: does directed recovery rise
with cell count (more cells = denser pseudotime sampling = higher SNR), and does the dynamic
operator beat the static, symmetric correlation at DIRECTED recovery?

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/31_boolode_dynamical/run_boolode_dynamical.py
  # --quick: fewer replicates and cell counts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import load_beeline_dataset
from stable_grn_inference.dynamics import (
    dmd_edges,
    dmd_operator,
    edges_to_operator,
    pseudotime_ordered_pairs,
    skeleton_recovery_aupr,
    specific_recovery_aupr,
    static_correlation_edges,
)
from stable_grn_inference.dynamics.separability import normalized_recovery

ROOT = Path(__file__).resolve().parents[2]
SYN_ROOT = ROOT / "data" / "raw" / "BEELINE-data" / "inputs" / "Synthetic"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "boolode_dynamical"
NET_TYPES = ("dyn-LI", "dyn-LL", "dyn-CY", "dyn-BF", "dyn-BFC", "dyn-TF")
CELL_COUNTS = (100, 200, 2000, 5000)


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or len(frame) == 0:
        return "_No rows._"
    cols = [str(c) for c in frame.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in frame.to_numpy():
        cells = [fmt(v) if isinstance(v, (float, np.floating)) else ("" if pd.isna(v) else str(v)) for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def score_one(base_dir: Path, name: str, ridge: float):
    ds = load_beeline_dataset(base_dir, name, reference="boolode", log1p=False)
    if ds.pseudotime is None or ds.reference_edges.empty:
        return None
    genes = ds.genes
    n = len(genes)
    X1, X2 = pseudotime_ordered_pairs(ds.expression, ds.pseudotime)
    if X1.shape[0] < n + 2:
        return None
    truth = edges_to_operator(ds.reference_edges, genes)
    chance = float((truth != 0).sum()) / (n * (n - 1)) if n > 1 else float("nan")
    dmd_score = dmd_edges(dmd_operator(X1, X2, ridge=ridge))
    static_score = static_correlation_edges(ds.expression.to_numpy(float))
    return {
        "chance": chance,
        "dmd_directed": specific_recovery_aupr(dmd_score, truth),
        "static_directed": specific_recovery_aupr(static_score, truth),
        "dmd_skeleton": skeleton_recovery_aupr(dmd_score, truth),
        "static_skeleton": skeleton_recovery_aupr(static_score, truth),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-replicates", type=int, default=10)
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()
    if not SYN_ROOT.exists():
        raise SystemExit(f"No BoolODE data at {SYN_ROOT}.")
    counts = (200, 5000) if args.quick else CELL_COUNTS
    max_rep = 3 if args.quick else args.max_replicates

    rows = []
    for count in counts:
        for net in NET_TYPES:
            base = SYN_ROOT / net / f"{net}-{count}"
            if not base.exists():
                continue
            reps = sorted(p.name for p in base.glob(f"{net}-{count}-*") if p.is_dir())[:max_rep]
            for name in reps:
                try:
                    res = score_one(base, name, args.ridge)
                except Exception:
                    res = None
                if res is None:
                    continue
                res.update({"cell_count": count, "net_type": net, "replicate": name})
                rows.append(res)

    if not rows:
        raise SystemExit("No BoolODE datasets scored.")
    df = pd.DataFrame(rows)
    for col in ("dmd_directed", "static_directed", "dmd_skeleton", "static_skeleton"):
        df[col + "_norm"] = [normalized_recovery(a, c) for a, c in zip(df[col], df["chance"])]
    df.to_csv(TABLES_DIR / f"{PREFIX}_all.csv", index=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 31: dynamical recovery on BoolODE single-cell time-series\n",
             "Directed recovery from a pseudotime axis on BoolODE single-cell data (exact ground "
             "truth), with the BEELINE cell-count sweep as the sample-size / SNR axis.\n",
             f"- network types {list(NET_TYPES)}; cell counts {list(counts)}; up to {max_rep} "
             f"replicates each; {len(df)} datasets scored.\n"]

    # cell-count sweep (the SNR / sample-size axis), averaged over types and replicates
    by_count = df.groupby("cell_count").agg(
        dmd_directed=("dmd_directed_norm", "mean"),
        static_directed=("static_directed_norm", "mean"),
        dmd_skeleton=("dmd_skeleton", "mean"),
        static_skeleton=("static_skeleton", "mean"),
        n=("dmd_directed", "size"),
    ).reset_index()
    lines.append("## Cell-count sweep (directed normalized recovery, mean over types/replicates)\n")
    lines.append(to_markdown_table(by_count))

    # per network type at the largest available count
    top = max(counts)
    by_type = df[df["cell_count"] == top].groupby("net_type").agg(
        dmd_directed=("dmd_directed_norm", "mean"),
        static_directed=("static_directed_norm", "mean"),
        n=("dmd_directed", "size"),
    ).reset_index()
    lines.append(f"\n## Per network type at {top} cells (directed normalized recovery)\n")
    lines.append(to_markdown_table(by_type))

    mean_dmd = float(df["dmd_directed_norm"].mean())
    mean_static = float(df["static_directed_norm"].mean())
    lo_dmd = float(by_count.iloc[0]["dmd_directed"]); hi_dmd = float(by_count.iloc[-1]["dmd_directed"])
    lines.append("\n## Findings\n")
    lines.append(f"- directed recovery (mean over all datasets): dynamic operator {fmt(mean_dmd)} vs "
                 f"static correlation {fmt(mean_static)}.")
    lines.append(f"- sample-size axis: directed recovery moves from {fmt(lo_dmd)} at {counts[0]} cells "
                 f"to {fmt(hi_dmd)} at {counts[-1]} cells (more cells = denser pseudotime sampling).")
    beats = mean_dmd > mean_static + 0.02
    rises = hi_dmd > lo_dmd
    lines.append(f"- the dynamic operator {'beats' if beats else 'does not beat'} the static "
                 f"correlation at directed recovery on BoolODE single-cell data.")
    lines.append(f"- directed recovery {'rises' if rises else 'does not rise'} with cell count, "
                 f"consistent with the exp 28 sample-size / SNR axis.")
    lines.append("- truth here is the exact BoolODE generating network; reproducibility is not "
                 "needed because correctness is graded directly.")

    pd.DataFrame([{
        "n_datasets": len(df), "mean_dmd_directed_norm": mean_dmd,
        "mean_static_directed_norm": mean_static,
        "dmd_directed_low_count": lo_dmd, "dmd_directed_high_count": hi_dmd,
        "dmd_beats_static": bool(beats), "recovery_rises_with_count": bool(rises),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
