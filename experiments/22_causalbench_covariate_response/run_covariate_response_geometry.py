r"""Experiment 22 - Covariate-aware direct-effect response geometry (RPE1).

Experiment 21 found the RPE1 perturbation response is dominated by a low-rank GLOBAL mode
(top SVD direction ~53% of variance), and that blindly removing top SVD modes hurt (it
deleted real directional signal and made residuals MORE diffuse). So the global mode is not
just technical noise.

This experiment takes the covariate-aware route instead of blind SVD removal:

1. Characterize the global mode: is it TECHNICAL (tracks UMI / library size / #cells /
   knockdown strength) or BIOLOGICAL (a coherent gene program)?
2. Build cleaner response targets WITHOUT deleting SVD modes:
   - raw Delta;
   - shared-PROGRAM residual (remove the average response profile, but keep it as an object);
   - COVARIATE residual (regress out self-knockdown strength + log #cells per perturbation).
3. Ask whether the cleaned targets are sharper: less diffuse, more split-half stable, more
   reproducibly ORIENTED across cell halves, and better aligned with inferred graph scores.
4. Keep the global program as its own object and inspect its top genes.

No gene ordering/graph is assumed (no wavelets). Core question: can we separate stable,
directional, gene-specific effects from the broad global response without deleting real
signal, and do inferred graphs explain the cleaned response better than co-expression?

Run:
    $env:PYTHONPATH = "src"
    .\.venv\Scripts\python.exe -B experiments/22_causalbench_covariate_response/run_covariate_response_geometry.py
    # --quick caps perturbations and skips GENIE3
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_grn_inference.data import (
    load_replogle_raw_h5ad,
    perturbation_response_matrix,
    residualize_against_covariates,
    response_sparsity,
    shared_response_program,
    split_half_stability,
)

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
PREFIX = "rpe1_covariate_response"

_EXP17 = ROOT / "experiments" / "17_dream4_stability_orientation_diagnostics" / "run_stability_orientation_diagnostics.py"
_spec = importlib.util.spec_from_file_location("exp17_diag", _EXP17)
exp17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp17)


def fmt(v, d=4):
    return exp17.fmt(v, d)


def program_residual(response, program_unit):
    """Remove a fixed program direction from each row (keeps a consistent basis across
    full/half matrices)."""
    M = response.to_numpy(dtype=float)
    pn = program_unit.reindex(response.columns).to_numpy(dtype=float)
    a = M @ pn
    R = M - np.outer(a, pn)
    return pd.DataFrame(R, index=response.index, columns=response.columns)


def cross_split_orientation(Da, Db, perturbed, *, margin=0.1):
    """Ground-truth-free directionality: fraction of perturbed pairs (with a clear
    relative asymmetry in BOTH halves) whose implied direction AGREES across halves.
    Margin is on the relative asymmetry |mAB-mBA|/(mAB+mBA), so it is scale-free and
    comparable across raw vs residual targets."""
    A = np.abs(Da.loc[perturbed, perturbed].to_numpy(dtype=float))
    B = np.abs(Db.loc[perturbed, perturbed].to_numpy(dtype=float))
    n = len(perturbed)
    agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            aab, aba = A[i, j], A[j, i]
            bab, bba = B[i, j], B[j, i]
            sa, sb = aab + aba, bab + bba
            if sa <= 0 or sb <= 0:
                continue
            if abs(aab - aba) / sa <= margin or abs(bab - bba) / sb <= margin:
                continue
            tot += 1
            agree += int((aab >= aba) == (bab >= bba))
    return {"agree_rate": agree / tot if tot else float("nan"), "n_pairs": tot}


def graph_scores(dataset, perturbed, *, alpha, n_obs_cap, seed, do_genie3):
    """Inferred edge-score matrices from CONTROL cells only: correlation, sparse LASSO,
    and (optionally) a light GENIE3 random forest. Returns dict name -> {(A,B): score}."""
    rng = np.random.default_rng(seed)
    ctrl = dataset.expression.loc[dataset.is_control.values]
    if len(ctrl) > n_obs_cap:
        ctrl = ctrl.iloc[rng.choice(len(ctrl), n_obs_cap, replace=False)]
    out = {}
    corr = ctrl.corr().abs()
    out["correlation"] = {(a, b): float(corr.loc[a, b]) for a in perturbed for b in perturbed if a != b}
    sp = exp17.fit_targetwise(ctrl, ctrl, alpha=alpha, include_self=False)
    out["sparse"] = {(s, t): float(v) for s, t, v in zip(sp["source"], sp["target"], sp["score"])}
    if do_genie3:
        from sklearn.ensemble import RandomForestRegressor
        genes = list(ctrl.columns)
        X = ctrl.to_numpy(dtype=float)
        gi = {g: i for i, g in enumerate(genes)}
        gmap = {}
        for t in perturbed:
            others = [g for g in genes if g != t]
            rf = RandomForestRegressor(n_estimators=15, max_features="sqrt", n_jobs=-1, random_state=0)
            rf.fit(X[:, [gi[g] for g in others]], X[:, gi[t]])
            for g, imp in zip(others, rf.feature_importances_):
                gmap[(g, t)] = float(imp)
        out["genie3"] = gmap
    return out


def alignment(score_map, target, perturbed):
    """Spearman between |inferred score| and |response| over perturbed->perturbed edges."""
    M = target.loc[perturbed, perturbed]
    xs, ys = [], []
    for a in perturbed:
        for b in perturbed:
            if a == b:
                continue
            xs.append(score_map.get((a, b), 0.0))
            ys.append(abs(float(M.loc[a, b])))
    if len(ys) <= 10:
        return float("nan")
    return float(spearmanr(xs, ys).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    max_perts = 200 if args.quick else None
    n_obs_cap = 1500 if args.quick else 4000
    do_genie3 = not args.quick

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if raw_path is None:
        raise SystemExit(f"No raw RPE1 h5ad in {CB_DIR}.")
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    perturbed = list(ds.perturbed_genes)

    D, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)

    # per-perturbation covariates + per-gene covariates
    self_kd = pd.Series({g: abs(float(D.loc[g, g])) for g in perturbed})
    n_cells = pd.Series({g: int((ds.perturbation_labels == g).sum()) for g in perturbed})
    log_cells = np.log(n_cells.astype(float))
    ctrl = ds.expression.loc[ds.is_control.values]
    gene_mean = ctrl.mean()
    gene_var = ctrl.var()

    lines = ["# Experiment 22 - RPE1 covariate-aware direct-effect response geometry\n"]
    lines.append("## 1. Setup\n")
    lines.append(f"- response matrix {D.shape[0]} x {D.shape[1]}; control cells {ds.metadata['n_control_cells']}; "
                 f"GENIE3 {'on' if do_genie3 else 'off'}\n")

    # --- 2. is the global mode technical or biological? ---
    U, s, Vt = np.linalg.svd(D.to_numpy(dtype=float), full_matrices=False)
    u0 = pd.Series(U[:, 0], index=perturbed)        # perturbation-side mode 1
    v0 = pd.Series(Vt[0, :], index=D.columns)        # gene-side mode 1
    rho_u_selfkd = spearmanr(np.abs(u0.values), self_kd.reindex(perturbed).values).statistic
    rho_u_cells = spearmanr(np.abs(u0.values), n_cells.reindex(perturbed).values).statistic
    rho_v_mean = spearmanr(np.abs(v0.values), gene_mean.reindex(D.columns).values).statistic
    rho_v_var = spearmanr(np.abs(v0.values), gene_var.reindex(D.columns).values).statistic
    lines.append("## 2. Is the global mode technical or biological?\n")
    lines.append(f"- top-1 mode variance share = {fmt(s[0]**2/ (s**2).sum())}")
    lines.append(f"- perturbation side |mode1| vs self-knockdown strength: Spearman {fmt(rho_u_selfkd)}; "
                 f"vs #cells: {fmt(rho_u_cells)}")
    lines.append(f"- gene side |mode1| vs gene mean-expression: Spearman {fmt(rho_v_mean)}; "
                 f"vs gene variance: {fmt(rho_v_var)}")
    lines.append("- reading: high |mode1|-vs-mean/variance => abundance/technical axis; "
                 "high |mode1|-vs-self-knockdown => biological response strength.\n")

    # --- 3. cleaned targets (no SVD deletion) ---
    prog = shared_response_program(D)
    program_unit = prog["program"]
    cov_df = pd.DataFrame({"self_kd": self_kd, "log_cells": log_cells}).reindex(perturbed)
    targets = {
        "raw": (D, Da, Db),
        "program_residual": (
            prog["residual"],
            program_residual(Da, program_unit),
            program_residual(Db, program_unit),
        ),
        "covariate_residual": (
            residualize_against_covariates(D, cov_df),
            residualize_against_covariates(Da, cov_df),
            residualize_against_covariates(Db, cov_df),
        ),
    }
    lines.append("## 3. Cleaned response targets (covariate-aware, no SVD deletion)\n")
    lines.append(f"- shared program explains {fmt(prog['program_var_explained'])} of total response variance")
    lines.append("\n| target | median eff. responders | median split-half cosine | cross-split orient. agreement | n pairs |")
    lines.append("| --- | --- | --- | --- | --- |")
    target_rows = {}
    for name, (full, ha, hb) in targets.items():
        diffuse = float(response_sparsity(full).median())
        stab = float(split_half_stability(ha, hb).median())
        orient = cross_split_orientation(ha, hb, perturbed)
        target_rows[name] = {"diffuse": diffuse, "stab": stab, **orient}
        lines.append(f"| {name} | {fmt(diffuse)} | {fmt(stab)} | {fmt(orient['agree_rate'])} | {orient['n_pairs']} |")
    lines.append("\n- orientation agreement vs 0.5 chance; >raw means cleaning improved reproducible direction.\n")

    # --- 4. graph explanation: do inferred graphs explain the response, raw vs cleaned? ---
    theory_alpha = 1.1 * np.sqrt(2.0 * np.log(max(len(perturbed), 2)) / n_obs_cap)
    scores = graph_scores(ds, perturbed, alpha=theory_alpha, n_obs_cap=n_obs_cap,
                          seed=args.random_seed, do_genie3=do_genie3)
    lines.append("## 4. Graph explanation: Spearman(|inferred score|, |response|)\n")
    header = "| structure | " + " | ".join(targets.keys()) + " |"
    lines.append(header)
    lines.append("| " + " --- |" * (len(targets) + 1))
    align_table = {}
    for sname, smap in scores.items():
        cells = [fmt(alignment(smap, targets[t][0], perturbed)) for t in targets]
        align_table[sname] = cells
        lines.append(f"| {sname} | " + " | ".join(cells) + " |")
    lines.append("\n- best structure / target tells us what, if anything, predicts the response.\n")

    # --- 5. keep the global program as an object ---
    top_prog = program_unit.abs().sort_values(ascending=False).head(15)
    lines.append("## 5. The shared global program (kept, not discarded)\n")
    lines.append(f"- top-15 program genes by |loading|: {', '.join(top_prog.index.tolist())}")
    lines.append("- (eyeball whether these are a coherent biological program, e.g. ribosomal/"
                 "proteostasis/cell-cycle, vs a generic abundance axis.)\n")

    # --- headline ---
    best_orient = max(target_rows, key=lambda k: (target_rows[k]["agree_rate"]
                                                  if np.isfinite(target_rows[k]["agree_rate"]) else -1))
    lines.append("## 6. Headline\n")
    lines.append(f"1. Global mode looks {'technical/abundance-linked' if abs(rho_v_mean) > abs(rho_u_selfkd) else 'response-strength/biological'} "
                 f"(|mode1|-vs-gene-mean {fmt(rho_v_mean)}, vs self-knockdown {fmt(rho_u_selfkd)}).")
    lines.append(f"2. Most reproducibly-oriented target: **{best_orient}** "
                 f"(agreement {fmt(target_rows[best_orient]['agree_rate'])} vs raw {fmt(target_rows['raw']['agree_rate'])}).")
    lines.append("3. Graph explanation: see table - whether any inferred structure beats correlation, "
                 "and whether cleaning the response improves alignment.")
    lines.append("4. The shared program is retained as its own object for interpretation.\n")

    # outputs
    pd.DataFrame(target_rows).T.to_csv(TABLES_DIR / f"{PREFIX}_targets.csv")
    pd.DataFrame(align_table, index=list(targets.keys())).to_csv(TABLES_DIR / f"{PREFIX}_alignment.csv")
    pd.DataFrame({
        "perturbation": perturbed, "self_knockdown": self_kd.reindex(perturbed).values,
        "n_cells": n_cells.reindex(perturbed).values, "mode1_loading": u0.reindex(perturbed).values,
    }).to_csv(TABLES_DIR / f"{PREFIX}_covariates.csv", index=False)
    top_prog.to_csv(TABLES_DIR / f"{PREFIX}_program_top_genes.csv")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
