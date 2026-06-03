r"""Experiment 24 - Does perturbation response have TRANSFERABLE structure? (RPE1)

My own follow-up (not the OpenAI prompt). Exp 20-23 mostly produced negatives about
recovering edges. This asks a different, more optimistic question that we had not tested:

  When we hold out one perturbation g, can we predict its true effect (its half-B response
  Delta_g^B) BETTER by using the shared structure of all the OTHER perturbations than by
  using g's own noisy estimate alone -- and crucially, better on the GENE-SPECIFIC part,
  not just the cell-cycle average?

If the shared low-rank response subspace DENOISES an individual perturbation, that is real,
useful, transferable structure (the response geometry "knows" something). If only the mean
cell-cycle program transfers, then the geometry is mostly the global program and little else.

Method (leave-one-perturbation-out, two independent cell halves A and B):
  target   = Delta_g^B                              (g's effect, estimated on half B)
  - self_only : predict with Delta_g^A              (g's own noisy half-A estimate)
  - mean_prog : predict with mean of OTHERS' Delta^A (the cell-cycle program)
  - lowrank_k : project Delta_g^A onto the top-k gene-space subspace learned from the
                OTHER perturbations' half-A responses (shared-structure denoiser)
Scored by cosine(pred, target). Reported BOTH raw and RESIDUAL (mean-program removed) so
we can see whether gene-specific structure (not just the cell cycle) transfers.

PRE-REGISTERED PREDICTION:
  - lowrank denoising beats self_only on RAW cosine: likely (~70%) - the shared subspace
    captures the reproducible part.
  - on the RESIDUAL (gene-specific, mean removed): genuinely unsure (~50/50). THIS is the
    crux. If yes -> a real positive direction. If no -> the only transferable thing is the
    cell cycle, and that is an honest, clarifying negative.

No new data, no wavelets, no RL, no neural nets. --quick samples fewer held-out genes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import load_replogle_raw_h5ad, perturbation_response_matrix

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
PREFIX = "causalbench_response_generalization"


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def residual_cosine(a, b, program_unit):
    """Cosine after removing the shared program direction from both vectors."""
    ar = a - np.dot(a, program_unit) * program_unit
    br = b - np.dot(b, program_unit) * program_unit
    return cosine(ar, br)


def low_rank_project(vec, basis_rows):
    """Project vec onto the span of the orthonormal rows in basis_rows (k x n)."""
    coeff = basis_rows @ vec
    return basis_rows.T @ coeff


def evaluate(D1, D2, perturbed, ks, *, sample, seed):
    """Leave-one-perturbation-out prediction of half-B response from half-A + others."""
    rng = np.random.default_rng(seed)
    n = len(perturbed)
    idx = np.arange(n)
    if sample is not None and sample < n:
        idx = rng.choice(n, sample, replace=False)
    methods = ["self_only", "mean_prog"] + [f"lowrank_{k}" for k in ks]
    raw = {m: [] for m in methods}
    res = {m: [] for m in methods}

    for g in idx:
        target = D2[g]
        self_pred = D1[g]
        others = np.delete(D1, g, axis=0)            # (n-1) x genes, half-A
        mean_p = others.mean(axis=0)
        program_unit = _unit(mean_p)
        # shared gene-space subspace from OTHERS (right singular vectors)
        # economy SVD; Vt rows are orthonormal gene-space directions
        Vt = np.linalg.svd(others, full_matrices=False)[2]

        preds = {"self_only": self_pred, "mean_prog": mean_p}
        for k in ks:
            preds[f"lowrank_{k}"] = low_rank_project(self_pred, Vt[:k])
        for m, p in preds.items():
            raw[m].append(cosine(p, target))
            res[m].append(residual_cosine(p, target, program_unit))

    rows = []
    for m in methods:
        rows.append({"method": m,
                     "raw_cosine": float(np.nanmean(raw[m])),
                     "residual_cosine": float(np.nanmean(res[m])),
                     "n_eval": len(raw[m])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    max_perts = 200 if args.quick else None
    sample = 60 if args.quick else 300
    ks = (1, 3, 5, 10, 20, 50)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if raw_path is None:
        raise SystemExit(f"No raw RPE1 h5ad in {CB_DIR}.")
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    perturbed = list(ds.perturbed_genes)
    _, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
    P = [g for g in perturbed if g in Da.columns]
    D1 = Da.loc[P, P].to_numpy(float)
    D2 = Db.loc[P, P].to_numpy(float)

    table = evaluate(D1, D2, P, ks, sample=sample, seed=args.random_seed)
    table.to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)

    self_raw = float(table.loc[table.method == "self_only", "raw_cosine"].iloc[0])
    self_res = float(table.loc[table.method == "self_only", "residual_cosine"].iloc[0])
    mean_raw = float(table.loc[table.method == "mean_prog", "raw_cosine"].iloc[0])
    lr = table[table.method.str.startswith("lowrank_")]
    best_raw = lr.loc[lr["raw_cosine"].idxmax()]
    best_res = lr.loc[lr["residual_cosine"].idxmax()]

    lines = ["# Experiment 24 - Transferable structure in perturbation response\n"]
    lines.append("_Pre-registered: low-rank denoising likely beats self_only on RAW cosine; "
                 "the crux is the RESIDUAL (gene-specific) - genuinely 50/50._\n")
    lines.append(f"- response block {D1.shape[0]} x {D1.shape[1]}; held-out perturbations evaluated: {int(table['n_eval'].iloc[0])}\n")
    lines.append("| method | raw cosine | residual cosine (mean-program removed) |")
    lines.append("| --- | --- | --- |")
    for _, r in table.iterrows():
        lines.append(f"| {r['method']} | {fmt(r['raw_cosine'])} | {fmt(r['residual_cosine'])} |")

    raw_gain = best_raw["raw_cosine"] - self_raw
    res_gain = best_res["residual_cosine"] - self_res
    lines.append("\n## Verdict (no hype)\n")
    lines.append(f"- self-only baseline: raw {fmt(self_raw)}, residual {fmt(self_res)}")
    lines.append(f"- mean-program (cell-cycle) baseline: raw {fmt(mean_raw)} (high = response is cell-cycle-dominated)")
    lines.append(f"- best low-rank denoiser RAW: {best_raw['method']} {fmt(best_raw['raw_cosine'])} "
                 f"(gain over self-only {fmt(raw_gain)})")
    lines.append(f"- best low-rank denoiser RESIDUAL: {best_res['method']} {fmt(best_res['residual_cosine'])} "
                 f"(gain over self-only {fmt(res_gain)})")

    raw_helps = raw_gain > 0.02
    res_helps = res_gain > 0.02
    if res_helps:
        verdict = ("PROMISING: shared structure denoises the GENE-SPECIFIC response (not just the "
                   "cell cycle) -> there is transferable structure beyond the global program.")
    elif raw_helps:
        verdict = ("MIXED: shared low-rank structure denoises the response, but the gain is mostly the "
                   "cell-cycle program; little gene-specific transferable structure.")
    else:
        verdict = ("NEGATIVE: shared structure does not denoise a held-out perturbation beyond its own "
                   "noisy estimate; response geometry is not transferably predictive here.")
    lines.append(f"\n**VERDICT: {verdict}**\n")
    lines.append("Reading: residual cosine is the honest metric. raw cosine is inflated by the shared "
                 "cell-cycle direction that every perturbation partly engages.\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(table))
        ax.plot(x, table["raw_cosine"], "o-", label="raw cosine")
        ax.plot(x, table["residual_cosine"], "s-", label="residual cosine (gene-specific)")
        ax.set_xticks(list(x)); ax.set_xticklabels(table["method"], rotation=45, ha="right", fontsize=8)
        ax.axhline(self_raw, color="C0", ls=":", lw=1); ax.axhline(self_res, color="C1", ls=":", lw=1)
        ax.set_ylabel("cosine(pred, held-out response)"); ax.legend(); ax.set_title("Leave-one-perturbation-out prediction")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        lines.append(f"(figure skipped: {e})")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
