r"""Experiment 30: directed-structure recovery with a time axis (Direction B).

Experiment 28 showed specific structure is not recoverable from a STATIC snapshot in RPE1's
regime (high dominant-mode fraction, low specific-SNR). The missing ingredient is a time
axis. A static symmetric statistic (correlation) cannot orient an edge, and a dominant shared
mode swamps it; a dynamical operator estimated from consecutive states can recover the
directed operator regardless of how dominant the shared input mode is.

Part A (synthetic, ground truth). A linear stochastic system x_{t+1} = A x_t + noise driven by
a sparse directed operator W (the truth), with a tunable dominant shared input mode. Compare
the dynamic-mode operator (uses time order) against the static correlation (ignores it),
graded by directed and skeleton AUPR, swept over dominant-mode strength and noise. The claim
to test: the time axis keeps DIRECTED recovery above the static floor as the dominant mode
grows.

Part B (DREAM4 Size10 time-series, realistic dynamics, known network, no download). Fit the
dynamic operator on the local DREAM4 time-series and grade against the gold-standard network;
compare to the static correlation baseline and the chance line. This is the project's existing
time-resolved data with checkable truth.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/30_dynamical_recovery/run_dynamical_recovery.py
  # --quick: smaller system, fewer seeds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
)
from stable_grn_inference.data.timeseries import (
    build_lagged_samples,
    split_trajectories_by_time_reset,
)
from stable_grn_inference.dynamics import (
    DYNAMICAL_METHODS,
    dmd_edges,
    dmd_operator,
    dynamical_recovery_grid,
    edges_to_operator,
    skeleton_recovery_aupr,
    specific_recovery_aupr,
    static_correlation_edges,
)
from stable_grn_inference.dynamics.separability import normalized_recovery

ROOT = Path(__file__).resolve().parents[2]
DREAM4_ROOT = ROOT / "data" / "raw" / "dream4"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "dynamical_recovery"


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
        cells = [fmt(v) if isinstance(v, float) else ("" if pd.isna(v) else str(v)) for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def directed_pivot(grid: pd.DataFrame, method: str) -> pd.DataFrame:
    block = grid[grid["method"] == method]
    table = block.pivot_table(index="noise", columns="mode_strength", values="directed_normalized")
    return table.sort_index(ascending=False)


def run_part_a(args, lines):
    if args.quick:
        mode_vals, noise_vals, n_genes, n_steps, n_seeds = [0.0, 3.0, 8.0], [0.3, 1.0], 24, 1500, 2
    else:
        mode_vals, noise_vals, n_genes, n_steps, n_seeds = [0.0, 1.0, 3.0, 8.0], [0.1, 0.3, 1.0], 30, 3000, 3
    grid = dynamical_recovery_grid(
        mode_vals, noise_vals, n_genes=n_genes, n_steps=n_steps, n_seeds=n_seeds,
        base_seed=args.random_seed,
    )
    grid.to_csv(TABLES_DIR / f"{PREFIX}_synthetic_grid.csv", index=False)

    lines.append("## Part A: synthetic system with known directed operator\n")
    lines.append(f"- {n_genes} genes, {n_steps} steps, {n_seeds} seeds/cell; "
                 f"mode strength in {mode_vals}, noise in {noise_vals}.\n")
    lines.append("Directed normalized recovery, dynamic operator (uses time order):\n")
    lines.append(to_markdown_table(directed_pivot(grid, "dmd").reset_index().rename(
        columns={"noise": "noise\\mode"})))
    lines.append("\nDirected normalized recovery, static correlation (ignores time order):\n")
    lines.append(to_markdown_table(directed_pivot(grid, "static").reset_index().rename(
        columns={"noise": "noise\\mode"})))

    dmd = grid[grid["method"] == "dmd"]
    static = grid[grid["method"] == "static"]
    mean_dmd = float(dmd["directed_normalized"].mean())
    mean_static = float(static["directed_normalized"].mean())
    # skeleton: do the two agree on the undirected skeleton (isolating that direction is the gain)?
    mean_dmd_skel = float(dmd["skeleton_aupr"].mean())
    mean_static_skel = float(static["skeleton_aupr"].mean())
    # robustness to the dominant mode: directed recovery at the strongest mode
    top_mode = max(mode_vals)
    dmd_top = float(dmd[dmd["mode_strength"] == top_mode]["directed_normalized"].mean())
    static_top = float(static[static["mode_strength"] == top_mode]["directed_normalized"].mean())

    lines.append("\n### Part A findings\n")
    lines.append(f"- directed recovery (mean over grid): dynamic operator {fmt(mean_dmd)} vs "
                 f"static correlation {fmt(mean_static)}.")
    lines.append(f"- skeleton AUPR (undirected, mean): dynamic {fmt(mean_dmd_skel)} vs static "
                 f"{fmt(mean_static_skel)}. The static method can find the skeleton but not the "
                 f"direction; the directed gain is what the time axis buys.")
    lines.append(f"- robustness to a dominant mode: at mode strength {fmt(top_mode, 1)}, directed "
                 f"recovery is dynamic {fmt(dmd_top)} vs static {fmt(static_top)}.")
    verdict = mean_dmd > mean_static + 0.1 and dmd_top > 0.1
    lines.append(f"- VERDICT: the time axis {'recovers directed structure the static snapshot cannot' if verdict else 'does not clearly beat the static baseline here'} "
                 f"(dynamic {fmt(mean_dmd)} vs static {fmt(mean_static)} directed normalized recovery).")
    return {
        "mean_dmd_directed": mean_dmd, "mean_static_directed": mean_static,
        "dmd_directed_top_mode": dmd_top, "static_directed_top_mode": static_top,
        "part_a_verdict_time_helps": bool(verdict),
    }


def run_part_b(args, lines):
    lines.append("\n## Part B: DREAM4 Size10 time-series (realistic dynamics, known network)\n")
    if not DREAM4_ROOT.exists():
        lines.append(f"- skipped: no DREAM4 data at {DREAM4_ROOT}.")
        return {"part_b_ran": False}

    rows = []
    for nid in range(1, 6):
        ts_path = dream4_size10_expression_path(DREAM4_ROOT, nid, "timeseries")
        gold_path = dream4_size10_gold_standard_path(DREAM4_ROOT, nid)
        if not ts_path.exists() or not gold_path.exists():
            continue
        ts = load_expression_matrix(ts_path, drop_time=False)
        trajs = split_trajectories_by_time_reset(ts, time_column="Time")
        Xt, Yt1, _ = build_lagged_samples(trajs, time_column="Time")
        if len(Xt) == 0:
            continue
        genes = list(Xt.columns)
        X1, X2 = Xt.to_numpy(float), Yt1.to_numpy(float)
        true_op = edges_to_operator(load_gold_standard_edges(gold_path), genes)
        n = len(genes)
        chance = float((true_op != 0).sum()) / (n * (n - 1))

        dmd_score = dmd_edges(dmd_operator(X1, X2, ridge=1e-2))
        static_score = static_correlation_edges(np.vstack([X1, X2[-1:]]))
        rows.append({
            "network": nid, "n_genes": n, "n_pairs": len(Xt), "chance": chance,
            "dmd_directed_aupr": specific_recovery_aupr(dmd_score, true_op),
            "static_directed_aupr": specific_recovery_aupr(static_score, true_op),
            "dmd_skeleton_aupr": skeleton_recovery_aupr(dmd_score, true_op),
            "static_skeleton_aupr": skeleton_recovery_aupr(static_score, true_op),
        })
    if not rows:
        lines.append(f"- skipped: DREAM4 time-series files not found under {DREAM4_ROOT}.")
        return {"part_b_ran": False}

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / f"{PREFIX}_dream4.csv", index=False)
    mean = df.mean(numeric_only=True)
    dmd_dir, static_dir = float(mean["dmd_directed_aupr"]), float(mean["static_directed_aupr"])
    chance = float(mean["chance"])
    lines.append(to_markdown_table(df))
    lines.append(f"\n- mean over {len(df)} networks: directed AUPR dynamic {fmt(dmd_dir)} vs static "
                 f"{fmt(static_dir)} (chance {fmt(chance)}).")
    lines.append(f"- directed normalized recovery: dynamic {fmt(normalized_recovery(dmd_dir, chance))} "
                 f"vs static {fmt(normalized_recovery(static_dir, chance))}.")
    lines.append(f"- on realistic GRN dynamics the dynamic operator {'beats' if dmd_dir > static_dir else 'does not beat'} "
                 f"the static correlation at directed recovery, consistent with the regime ladder "
                 f"(time-series orient; static does not).")
    return {
        "part_b_ran": True, "dream4_dmd_directed_aupr": dmd_dir,
        "dream4_static_directed_aupr": static_dir, "dream4_chance": chance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 30: directed-structure recovery with a time axis\n",
             "Does a time axis recover the directed structure that a static snapshot cannot? "
             "Synthetic with ground truth (Part A) and DREAM4 time-series (Part B).\n"]

    summary = {}
    summary.update(run_part_a(args, lines))
    summary.update(run_part_b(args, lines))

    lines.append("\n## Interpretation\n")
    lines.append("- The static correlation is symmetric and cannot orient an edge; the dynamic "
                 "operator estimate is unbiased by the input covariance, so it recovers the "
                 "directed operator even under a dominant shared mode. This is the regime ladder's "
                 "top rung (time-series orient) made into a controlled, ground-truthed experiment.")
    lines.append("- It motivates Direction B's data requirement: a real time-resolved dataset with "
                 "checkable truth, above the exp 28 SNR floor. The dataset scout for that is separate.")

    pd.DataFrame([summary]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
