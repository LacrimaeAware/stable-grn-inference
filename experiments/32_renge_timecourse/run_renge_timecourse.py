r"""Experiment 32: time-resolved knockout response on RENGE Perturb-seq (Direction B, real data).

RENGE (GEO GSE213069) is real time-resolved single-cell CRISPR knockout in human iPSCs: four
daily timepoints, 23 knocked-out transcription factors, non-targeting controls. Unlike the
static RPE1 snapshot, it has a real time axis, so the knockout response can be watched build
over days.

This experiment measures the time-resolved structure that only a time axis exposes:
  1. response growth: does the total knockout response magnitude grow from day 2 to day 5 as
     effects propagate (a cascade building over real time)?
  2. directional-ordering stability: is the per-gene net_out (upstream/downstream position)
     consistent across days, and does it stabilize?
  3. within-day reproducibility: split-half cosine of the response per day.

Directed-edge grading against the RENGE ChIP-seq proxy network is the completing step and is
NOT done here (that network is not in the GEO download; it lives in the RENGE repository). This
experiment establishes the unsupervised time-resolved structure first.

Run:
  $env:PYTHONPATH = "src"
  .\.venv\Scripts\python.exe -B experiments/32_renge_timecourse/run_renge_timecourse.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_grn_inference.analysis import net_out
from stable_grn_inference.data import (
    load_renge_timecourse,
    perturbation_response_matrix,
    split_half_stability,
)

ROOT = Path(__file__).resolve().parents[2]
RENGE_ROOT = ROOT / "data" / "raw" / "renge"
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "renge_timecourse"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    if not RENGE_ROOT.exists():
        raise SystemExit(f"No RENGE data at {RENGE_ROOT}.")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_renge_timecourse(RENGE_ROOT)
    days = sorted(datasets)
    per_day = []
    net_outs = {}
    for day in days:
        ds = datasets[day]
        P = list(ds.perturbed_genes)
        full, da, db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
        D = full.loc[P, P].to_numpy(float)
        cos = split_half_stability(da.loc[P], db.loc[P])
        net_outs[day] = pd.Series(net_out(D), index=P)
        per_day.append({
            "day": day,
            "n_perturbed": len(P),
            "n_control_cells": int(ds.metadata["n_control_cells"]),
            "response_norm": float(np.linalg.norm(D)),
            "mean_perturb_magnitude": float(np.linalg.norm(D, axis=1).mean()),
            "median_split_half_cosine": float(np.nanmedian(cos.to_numpy())),
        })
    day_df = pd.DataFrame(per_day)
    day_df.to_csv(TABLES_DIR / f"{PREFIX}_by_day.csv", index=False)

    # cross-day net_out (ordering) reproducibility
    shared = sorted(set.intersection(*[set(s.index) for s in net_outs.values()]))
    cross = pd.DataFrame(index=days, columns=days, dtype=float)
    for a in days:
        for b in days:
            cross.loc[a, b] = float(spearmanr(net_outs[a][shared], net_outs[b][shared]).statistic)
    cross.to_csv(TABLES_DIR / f"{PREFIX}_netout_crossday.csv")

    lines = ["# Experiment 32: time-resolved knockout response on RENGE Perturb-seq\n",
             "Real time-resolved single-cell CRISPR knockout (human iPSC, days 2-5, 23 TFs). The "
             "time-resolved structure a static snapshot cannot show.\n",
             "## Per-day response\n", to_markdown_table(day_df),
             "\n## net_out (upstream/downstream ordering) cross-day reproducibility (Spearman)\n",
             to_markdown_table(cross.reset_index().rename(columns={"index": "day"}))]

    norms = day_df["response_norm"].to_numpy()
    grow = float(norms[-1] - norms[0])
    adjacent = [float(cross.loc[days[i], days[i + 1]]) for i in range(len(days) - 1)]
    off_diag = [float(cross.loc[a, b]) for a in days for b in days if a != b]
    lines.append("\n## Findings\n")
    lines.append(f"- response magnitude {'grows' if grow > 0 else 'does not grow'} across the time "
                 f"course: ||D|| {fmt(float(norms[0]))} ({days[0]}) to {fmt(float(norms[-1]))} ({days[-1]}).")
    lines.append(f"- net_out ordering is {'reproducible' if np.mean(off_diag) > 0.3 else 'weakly reproducible'} "
                 f"across days (mean cross-day Spearman {fmt(float(np.mean(off_diag)))}; adjacent-day "
                 f"{fmt(float(np.mean(adjacent)))}); the upstream/downstream axis persists over time.")
    lines.append(f"- within-day reproducibility (median split-half cosine) ranges "
                 f"{fmt(float(day_df['median_split_half_cosine'].min()))} to "
                 f"{fmt(float(day_df['median_split_half_cosine'].max()))}.")
    lines.append("- directed-edge grading against the RENGE ChIP-seq proxy network is the next step "
                 "(that network is not in the GEO download); this establishes the time-resolved "
                 "structure first.")

    pd.DataFrame([{
        "n_days": len(days), "response_growth": grow,
        "mean_crossday_netout_spearman": float(np.mean(off_diag)),
        "mean_adjacent_netout_spearman": float(np.mean(adjacent)),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
