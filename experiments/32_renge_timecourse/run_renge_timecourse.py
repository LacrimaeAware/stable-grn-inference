r"""Experiment 32: time-resolved knockout response on RENGE Perturb-seq (Direction B, real data).

RENGE (GEO GSE213069) is real time-resolved single-cell CRISPR knockout in human iPSCs: four
daily timepoints, 23 knocked-out transcription factors, non-targeting controls. Unlike the
static RPE1 snapshot, it has a real time axis, so the knockout response can be watched build
over days, and recovery can be graded against an external network.

Parts:
  1. time-resolved structure: response growth over days, directional-ordering (net_out)
     stability across days, within-day reproducibility.
  2. graded recovery: the per-day interventional response is graded against the STRING
     functional network (a downloadable, literature-validated proxy; section 8 of
     next_direction.md), against an observational control-cell correlation baseline and the
     chance line, to test whether recovery strengthens over the time course. STRING is
     undirected, so this grades the interaction skeleton; directed grading against a TF-target
     or ChIP network is the further step.

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
    load_string_network,
    perturbation_response_matrix,
    skeleton_truth_matrix,
    split_half_stability,
)
from stable_grn_inference.dynamics import skeleton_recovery_aupr

ROOT = Path(__file__).resolve().parents[2]
RENGE_ROOT = ROOT / "data" / "raw" / "renge"
STRING_PATH = ROOT / "data" / "raw" / "string" / "renge_tfs_string.tsv"
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
    per_day, net_outs, response_mats, control_corr = [], {}, {}, {}
    genes = None
    for day in days:
        ds = datasets[day]
        P = list(ds.perturbed_genes)
        genes = P
        full, da, db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
        D = full.loc[P, P].to_numpy(float)
        response_mats[day] = D
        ctrl = ds.expression.loc[ds.is_control.to_numpy(), P].to_numpy(float)
        cc = np.corrcoef(ctrl, rowvar=False) if ctrl.shape[0] > 2 else np.zeros((len(P), len(P)))
        control_corr[day] = np.nan_to_num(np.atleast_2d(cc))
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
             "Real time-resolved single-cell CRISPR knockout (human iPSC, days 2-5, 23 TFs).\n",
             "## Per-day response\n", to_markdown_table(day_df),
             "\n## net_out (upstream/downstream ordering) cross-day reproducibility (Spearman)\n",
             to_markdown_table(cross.reset_index().rename(columns={"index": "day"}))]

    norms = day_df["response_norm"].to_numpy()
    grow = float(norms[-1] - norms[0])
    adjacent = [float(cross.loc[days[i], days[i + 1]]) for i in range(len(days) - 1)]
    off_diag = [float(cross.loc[a, b]) for a in days for b in days if a != b]

    # graded recovery against the STRING functional network (external proxy)
    string_summary = {}
    if STRING_PATH.exists() and genes is not None:
        T = skeleton_truth_matrix(load_string_network(STRING_PATH), genes, min_score=0.4)
        n = len(genes)
        off = ~np.eye(n, dtype=bool)
        edge_rate = float(T[off].mean())
        srows = [{
            "day": day,
            "interventional_aupr": skeleton_recovery_aupr(np.abs(response_mats[day]), T),
            "observational_aupr": skeleton_recovery_aupr(np.abs(control_corr[day]), T),
            "chance": edge_rate,
        } for day in days]
        sdf = pd.DataFrame(srows)
        sdf.to_csv(TABLES_DIR / f"{PREFIX}_string_grading.csv", index=False)
        iv = sdf["interventional_aupr"].to_numpy(float)
        lines.append("\n## Graded recovery vs STRING functional network (external proxy)\n")
        lines.append(f"- {int(T[off].sum() / 2)} STRING edges among {n} TFs (min score 0.4); "
                     f"skeleton chance {fmt(edge_rate)}. STRING is undirected, so this grades the "
                     f"interaction skeleton.\n")
        lines.append(to_markdown_table(sdf))
        lines.append(f"\n- interventional recovery vs STRING moves from {fmt(float(iv[0]))} "
                     f"({days[0]}) to {fmt(float(iv[-1]))} ({days[-1]}); observational control-cell "
                     f"correlation baseline {fmt(float(sdf['observational_aupr'].mean()))}; "
                     f"chance {fmt(edge_rate)}.")
        string_summary = {
            "string_interventional_first": float(iv[0]),
            "string_interventional_last": float(iv[-1]),
            "string_observational_mean": float(sdf["observational_aupr"].mean()),
            "string_chance": edge_rate,
        }
    else:
        lines.append(f"\n## Graded recovery vs STRING\n- skipped: no STRING network at {STRING_PATH}.")

    lines.append("\n## Findings\n")
    lines.append(f"- response magnitude {'grows' if grow > 0 else 'does not grow'} across the time "
                 f"course: ||D|| {fmt(float(norms[0]))} ({days[0]}) to {fmt(float(norms[-1]))} ({days[-1]}).")
    lines.append(f"- net_out ordering is {'reproducible' if np.mean(off_diag) > 0.3 else 'weakly reproducible'} "
                 f"across days (mean cross-day Spearman {fmt(float(np.mean(off_diag)))}; adjacent-day "
                 f"{fmt(float(np.mean(adjacent)))}); the upstream/downstream axis persists over time.")
    if string_summary:
        better = string_summary["string_interventional_last"] > string_summary["string_observational_mean"]
        rises = string_summary["string_interventional_last"] > string_summary["string_interventional_first"]
        lines.append(f"- graded vs STRING: the interventional response recovers STRING links "
                     f"{'above' if string_summary['string_interventional_last'] > string_summary['string_chance'] else 'at'} "
                     f"chance and {'beats' if better else 'does not beat'} the observational baseline; "
                     f"recovery {'rises' if rises else 'does not rise'} over the time course.")
    lines.append("- directed grading against a TF-target or ChIP network is the further step (STRING "
                 "is undirected); the time-resolved structure and skeleton recovery are graded here.")

    summary = {
        "n_days": len(days), "response_growth": grow,
        "mean_crossday_netout_spearman": float(np.mean(off_diag)),
        "mean_adjacent_netout_spearman": float(np.mean(adjacent)),
    }
    summary.update(string_summary)
    pd.DataFrame([summary]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
