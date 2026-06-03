r"""Experiment 21 - Perturbation-response geometry and direct-effect filtering (RPE1).

Reframes Track A through Track B's lens: an intervention produces a DISPLACEMENT vector
(gene perturbation -> expression-response delta), and we study the GEOMETRY of those
displacements. This directly attacks experiment 20's open wound: the interventional
"reference" was extremely dense (0.82) because perturbing a gene shifts much of the
transcriptome (broad/global response). Here we (a) measure how low-rank/global that
response is, (b) subtract the broad component to get a sharper DIRECT-effect target, and
(c) re-ask orientation and observational-transfer questions against the sharpened target -
including a ground-truth-free orientation check: is the asymmetry-implied direction
REPRODUCIBLE across independent cell halves?

No gene ordering or graph is assumed, so no wavelets/scattering are forced on genes.

Run:
    $env:PYTHONPATH = "src"
    .\.venv\Scripts\python.exe -B experiments/21_causalbench_response_geometry/run_response_geometry.py
    # --quick caps perturbations for a fast pass
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_grn_inference.data import (
    direct_effect_filter,
    load_replogle_raw_h5ad,
    perturbation_response_matrix,
    response_low_rank,
    response_sparsity,
    split_half_stability,
)

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
PREFIX = "rpe1_response_geometry"

_EXP17 = ROOT / "experiments" / "17_dream4_stability_orientation_diagnostics" / "run_stability_orientation_diagnostics.py"
_spec = importlib.util.spec_from_file_location("exp17_diag", _EXP17)
exp17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp17)


def fmt(v, d=4):
    return exp17.fmt(v, d)


def control_null_threshold(dataset, *, n_splits=10, seed=0, pct=95.0):
    """Pooled control-vs-control mean-shift null: split control cells, |meanA - meanB|
    per gene; threshold = pooled percentile. Sets 'a response counts as real' bar."""
    X = dataset.expression.to_numpy(dtype=float)
    ctrl = np.where(dataset.is_control.to_numpy())[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_splits):
        sel = rng.permutation(ctrl)
        half = sel.size // 2
        diff = np.abs(X[sel[:half]].mean(0) - X[sel[half:2 * half]].mean(0))
        vals.append(diff)
    return float(np.percentile(np.concatenate(vals), pct))


def reference_density(D, threshold):
    """Fraction of off-diagonal entries whose |response| exceeds the null threshold."""
    M = D.to_numpy(dtype=float).copy()
    np.fill_diagonal(M, 0.0)
    n = M.shape[0]
    return float((np.abs(M) > threshold).sum()) / (n * (n - 1))


def implied_directions(D, perturbed, *, threshold):
    """For each unordered perturbed pair, implied direction = larger |response|; decidable
    if max|.|>threshold and the asymmetry exceeds threshold. Returns dict pair->source."""
    M = {(a, b): float(D.loc[a, b]) for a in perturbed for b in perturbed if a != b}
    out, decided = {}, {}
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            a, b = perturbed[i], perturbed[j]
            mab, mba = abs(M[(a, b)]), abs(M[(b, a)])
            src = a if mab >= mba else b
            out[(a, b)] = src
            decided[(a, b)] = max(mab, mba) > threshold and abs(mab - mba) > threshold
    return out, decided


def cross_split_orientation(Da, Db, perturbed, *, threshold):
    """Ground-truth-free directionality: fraction of pairs (decidable in BOTH halves)
    whose implied direction AGREES across the two independent cell halves. 0.5 = chance."""
    da, dda = implied_directions(Da, perturbed, threshold=threshold)
    db, ddb = implied_directions(Db, perturbed, threshold=threshold)
    agree = tot = 0
    for k in da:
        if dda[k] and ddb[k]:
            tot += 1
            agree += int(da[k] == db[k])
    return {"agree_rate": agree / tot if tot else float("nan"), "n_pairs": tot}


def observational_alignment(dataset, D, perturbed, *, alpha, n_obs_cap, seed, restrict=None):
    """Spearman between observational scores (control-cell correlation & sparse) and the
    interventional |response| over perturbed->perturbed edges. ``restrict`` optionally
    limits to a subset of source genes (e.g. split-half-stable perturbations)."""
    rng = np.random.default_rng(seed)
    ctrl = dataset.expression.loc[dataset.is_control.values]
    if len(ctrl) > n_obs_cap:
        ctrl = ctrl.iloc[rng.choice(len(ctrl), n_obs_cap, replace=False)]
    corr = ctrl.corr().abs()
    sparse = exp17.fit_targetwise(ctrl, ctrl, alpha=alpha, include_self=False)
    smap = {(s, t): v for s, t, v in zip(sparse["source"], sparse["target"], sparse["score"])}

    src_set = set(perturbed if restrict is None else restrict)
    corr_x, sparse_x, resp_y = [], [], []
    for a in perturbed:
        if a not in src_set:
            continue
        for b in perturbed:
            if a == b:
                continue
            corr_x.append(corr.loc[a, b])
            sparse_x.append(smap.get((a, b), 0.0))
            resp_y.append(abs(float(D.loc[a, b])))
    out = {}
    if len(resp_y) > 10:
        out["correlation_spearman"] = float(spearmanr(corr_x, resp_y).statistic)
        out["sparse_spearman"] = float(spearmanr(sparse_x, resp_y).statistic)
    out["n_edges"] = len(resp_y)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-modes", type=int, default=10, help="global modes removed for direct-effect filtering")
    args = ap.parse_args()
    max_perts = 200 if args.quick else None
    n_obs_cap = 2000 if args.quick else 4000
    n_modes = args.n_modes

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if raw_path is None:
        raise SystemExit(f"No raw RPE1 h5ad in {CB_DIR}.")
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    perturbed = list(ds.perturbed_genes)

    D, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)

    lines = ["# Experiment 21 - RPE1 perturbation-response geometry\n"]
    lines.append("## 1. Setup\n")
    lines.append(f"- response matrix: {D.shape[0]} perturbations x {D.shape[1]} genes; "
                 f"control cells: {ds.metadata['n_control_cells']}; n_modes removed: {n_modes}\n")

    # --- QC: self-knockdown should lower the perturbed gene ---
    self_resp = np.array([D.loc[g, g] for g in perturbed])
    lines.append("## 2. QC: self-knockdown response\n")
    lines.append(f"- median self-response D[g,g] = {fmt(np.median(self_resp))} "
                 f"(should be NEGATIVE: CRISPRi lowers the gene); fraction negative = "
                 f"{fmt(np.mean(self_resp < 0))}\n")

    # --- response geometry ---
    lr = response_low_rank(D, var_cutoff=0.9)
    spars = response_sparsity(D)
    stab = split_half_stability(Da, Db)
    cells_per = pd.Series({g: int((ds.perturbation_labels == g).sum()) for g in perturbed})
    stab_cell_rho = spearmanr(stab.values, cells_per.reindex(stab.index).values).statistic
    lines.append("## 3. Response geometry\n")
    lines.append(f"- LOW-RANK: rank@90%var = {lr['rank_at_cutoff']} / {len(perturbed)}; "
                 f"top-1 mode variance = {fmt(lr['top1_var'])}; "
                 f"top-{n_modes} cum var = {fmt(lr['cum_var_explained'][min(n_modes-1, len(lr['cum_var_explained'])-1)])}; "
                 f"spectral entropy = {fmt(lr['spectral_entropy'])}")
    lines.append(f"- DIFFUSENESS: median effective #responders/perturbation = {fmt(spars.median())} "
                 f"of {D.shape[1]} genes")
    lines.append(f"- SPLIT-HALF STABILITY: median cosine = {fmt(stab.median())}; "
                 f"fraction > 0.5 = {fmt(np.mean(stab.dropna() > 0.5))}; "
                 f"Spearman(stability, #cells) = {fmt(stab_cell_rho)}\n")

    # --- direct-effect filtering ---
    threshold = control_null_threshold(ds, seed=args.random_seed)
    D_direct, broad = direct_effect_filter(D, n_modes=n_modes)
    Da_d, _ = direct_effect_filter(Da, n_modes=n_modes)
    Db_d, _ = direct_effect_filter(Db, n_modes=n_modes)
    dens_raw = reference_density(D, threshold)
    dens_dir = reference_density(D_direct, threshold)
    spars_dir = response_sparsity(D_direct)
    lines.append("## 4. Direct-effect filtering (remove top global modes)\n")
    lines.append(f"- control-null threshold (95th pct |ctrl-split|) = {fmt(threshold)}")
    lines.append(f"- reference density: raw = {fmt(dens_raw)} -> direct = {fmt(dens_dir)} "
                 f"({'sharper/sparser' if dens_dir < dens_raw else 'not sparser'})")
    lines.append(f"- median effective #responders: raw = {fmt(spars.median())} -> "
                 f"direct = {fmt(spars_dir.median())}\n")

    # --- orientation: cross-split reproducibility (ground-truth-free) ---
    rep_raw = cross_split_orientation(Da, Db, perturbed, threshold=threshold)
    rep_dir = cross_split_orientation(Da_d, Db_d, perturbed, threshold=threshold)
    lines.append("## 5. Orientation reproducibility across cell halves (no ground truth needed)\n")
    lines.append("| target | cross-split direction agreement | n decidable-in-both | chance |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| raw response | {fmt(rep_raw['agree_rate'])} | {rep_raw['n_pairs']} | 0.5 |")
    lines.append(f"| direct-effect | {fmt(rep_dir['agree_rate'])} | {rep_dir['n_pairs']} | 0.5 |")
    lines.append("\n- Agreement > 0.5 means the interventional direction is REPRODUCIBLE "
                 "(verifiable directionality), unlike observational orientation (undecidable).\n")

    # --- observational alignment: raw vs direct vs stable-only ---
    theory_alpha = 1.1 * np.sqrt(2.0 * np.log(max(len(perturbed), 2)) / n_obs_cap)
    stable_src = list(stab[stab > stab.median()].index)
    align_raw = observational_alignment(ds, D, perturbed, alpha=theory_alpha, n_obs_cap=n_obs_cap, seed=args.random_seed)
    align_dir = observational_alignment(ds, D_direct, perturbed, alpha=theory_alpha, n_obs_cap=n_obs_cap, seed=args.random_seed)
    align_stab = observational_alignment(ds, D_direct, perturbed, alpha=theory_alpha, n_obs_cap=n_obs_cap,
                                         seed=args.random_seed, restrict=stable_src)
    lines.append("## 6. Observational->interventional alignment (Spearman |score| vs |response|)\n")
    lines.append("| target | correlation rho | sparse rho | n edges |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| raw response | {fmt(align_raw.get('correlation_spearman'))} | "
                 f"{fmt(align_raw.get('sparse_spearman'))} | {align_raw['n_edges']} |")
    lines.append(f"| direct-effect | {fmt(align_dir.get('correlation_spearman'))} | "
                 f"{fmt(align_dir.get('sparse_spearman'))} | {align_dir['n_edges']} |")
    lines.append(f"| direct + split-stable sources | {fmt(align_stab.get('correlation_spearman'))} | "
                 f"{fmt(align_stab.get('sparse_spearman'))} | {align_stab['n_edges']} |\n")

    # --- headline ---
    lines.append("## 7. Headline\n")
    lines.append(f"1. RPE1 perturbation response is {'LOW-RANK/global-mode dominated' if lr['top1_var']>0.2 else 'fairly diffuse'} "
                 f"(top-1 mode {fmt(lr['top1_var'])} of variance; rank@90% = {lr['rank_at_cutoff']}).")
    lines.append(f"2. Direct-effect filtering {'sharpens' if dens_dir<dens_raw else 'does not sharpen'} the target "
                 f"(density {fmt(dens_raw)} -> {fmt(dens_dir)}).")
    lines.append(f"3. Interventional orientation is reproducible across cell halves: raw {fmt(rep_raw['agree_rate'])}, "
                 f"direct {fmt(rep_dir['agree_rate'])} (vs 0.5 chance) - VERIFIABLE directionality.")
    lines.append(f"4. Observational alignment with the response is weak and "
                 f"{'improves' if (align_dir.get('correlation_spearman') or 0) > (align_raw.get('correlation_spearman') or 0) else 'does not improve'} "
                 f"after direct-effect filtering.\n")

    # --- outputs ---
    pd.DataFrame({
        "perturbation": perturbed,
        "self_response": self_resp,
        "effective_responders_raw": spars.reindex(perturbed).values,
        "effective_responders_direct": spars_dir.reindex(perturbed).values,
        "split_half_cosine": stab.reindex(perturbed).values,
        "n_cells": cells_per.reindex(perturbed).values,
    }).to_csv(TABLES_DIR / f"{PREFIX}_per_perturbation.csv", index=False)
    pd.DataFrame({
        "k": np.arange(1, len(lr["singular_values"]) + 1),
        "singular_value": lr["singular_values"],
        "var_explained": lr["var_explained"],
        "cum_var_explained": lr["cum_var_explained"],
    }).to_csv(TABLES_DIR / f"{PREFIX}_spectrum.csv", index=False)
    pd.DataFrame([{
        "n_perturbations": len(perturbed), "rank90": lr["rank_at_cutoff"],
        "top1_var": lr["top1_var"], "spectral_entropy": lr["spectral_entropy"],
        "median_effective_responders_raw": float(spars.median()),
        "median_effective_responders_direct": float(spars_dir.median()),
        "median_split_half_cosine": float(stab.median()),
        "density_raw": dens_raw, "density_direct": dens_dir,
        "cross_split_orient_raw": rep_raw["agree_rate"], "cross_split_orient_direct": rep_dir["agree_rate"],
        "align_corr_raw": align_raw.get("correlation_spearman"), "align_corr_direct": align_dir.get("correlation_spearman"),
        "align_corr_direct_stable": align_stab.get("correlation_spearman"),
    }]).to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(np.arange(1, len(lr["singular_values"]) + 1), lr["cum_var_explained"], marker=".")
        ax[0].axhline(0.9, color="k", ls="--"); ax[0].set_title("Response spectrum (cum var)")
        ax[0].set_xlabel("component"); ax[0].set_xlim(0, min(60, len(lr["singular_values"])))
        ax[1].hist(stab.dropna(), bins=40, color="#37b"); ax[1].set_title("Split-half stability (cosine)")
        ax[2].hist(spars.dropna(), bins=40, alpha=0.6, label="raw")
        ax[2].hist(spars_dir.dropna(), bins=40, alpha=0.6, label="direct")
        ax[2].set_title("Effective #responders"); ax[2].legend()
        fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        lines.append(f"\n(figure skipped: {e})")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
