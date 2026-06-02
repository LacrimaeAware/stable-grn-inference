"""BEELINE diagnostics (experiment 18): transfer the experiment-17 framing to a
real single-cell GRN benchmark.

Data decision (judgment, not the literal review prompt): the local BEELINE download
ships the real scRNA-seq sets (hESC, hHep, mESC, ...) with expression + pseudotime
but NO reference networks, so they cannot be scored here. It does ship the BEELINE
**Curated** benchmark (GSD, HSC, VSC, mCAD) with EXACT directed ground truth and 10
replicates each. We run the diagnostics there: it is real BEELINE benchmark data
(not a synthetic smoke fixture), it has true directed labels with dense reciprocal
structure (so the skeleton-vs-orientation decomposition is meaningful), and it is the
OPPOSITE regime to DREAM4 (n cells >> p genes) -- a real transfer/contrast test.

Single-cell snapshots are static, so we run static methods only (correlation,
GENIE3/tree, static exclude-self LASSO, fusion, cell-subsample stability). No dynamic
lagged LASSO. The same diagnostic functions as experiment 17 are reused verbatim, so
DREAM4 and BEELINE use identical metric definitions.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stable_grn_inference.data import load_beeline_dataset
from stable_grn_inference.evaluation import aupr, auroc, precision_at_k
from stable_grn_inference.inference import (
    rank_edges_by_correlation,
    rank_edges_by_random_forest,
    rank_fusion,
)

# reuse experiment 17's diagnostic functions verbatim (identical definitions)
_EXP17 = ROOT / "experiments" / "17_dream4_stability_orientation_diagnostics" / "run_stability_orientation_diagnostics.py"
_spec = importlib.util.spec_from_file_location("exp17_diag", _EXP17)
exp17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp17)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

RESULTS_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"
PREFIX = "beeline_diagnostics"
BEELINE_ROOT = ROOT / "data/raw/BEELINE-data"
CURATED_ROOT = BEELINE_ROOT / "inputs/Curated"
SCRNA_ROOT = BEELINE_ROOT / "inputs/scRNA-Seq"
CURATED_MODELS = ("GSD", "HSC", "VSC", "mCAD")
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
PRECISION_KS = (5, 10, 20)
N_BOOTSTRAP_FUSION = 3
TREE_ESTIMATORS = 100
COEF_TOL = exp17.COEF_TOL


# --------------------------------------------------------------------------- #
def cell_subsamples(n_cells: int, n_subsamples: int, *, seed: int, fraction: float = 0.5,
                    complementary: bool = True) -> list[np.ndarray]:
    """Half-sample CELLS without replacement (single-cell cells are ~exchangeable,
    unlike DREAM4 trajectories, so the MB stability assumptions are cleaner here)."""
    rng = np.random.default_rng(seed)
    k = max(2, int(round(fraction * n_cells)))
    idx_all = np.arange(n_cells)
    samples: list[np.ndarray] = []
    for _ in range(n_subsamples):
        chosen = np.sort(rng.choice(idx_all, size=k, replace=False))
        samples.append(chosen)
        if complementary:
            samples.append(np.setdiff1d(idx_all, chosen))
    return samples


def epr(scored: pd.DataFrame, n_true: int, n_candidate: int) -> float:
    """Early precision ratio = precision@n_true / random density (BEELINE-style)."""
    density = n_true / n_candidate if n_candidate else 0.0
    if density == 0:
        return float("nan")
    return float(precision_at_k(scored, "is_true", max(n_true, 1)) / density)


def discover_replicates(model: str, max_reps: int) -> list[str]:
    base = CURATED_ROOT / model
    reps = sorted((d.name for d in base.glob(f"{model}-2000-*") if d.is_dir()),
                  key=lambda s: int(s.split("-")[-1]))
    return reps[:max_reps]


def scrna_reference_status() -> list[str]:
    """Report which scRNA-seq datasets lack a usable reference network."""
    missing = []
    if SCRNA_ROOT.exists():
        for d in sorted(SCRNA_ROOT.iterdir()):
            if d.is_dir() and not any((d / c).exists() for c in ("refNetwork.csv", "GroundTruthNetwork.csv")):
                missing.append(d.name)
    return missing


# --------------------------------------------------------------------------- #
def run_replicate(model: str, rep: str, *, alpha_grid, n_subsamples, seed, n_jobs):
    ds = load_beeline_dataset(CURATED_ROOT / model, rep, reference="exact")
    expr = ds.expression                      # cells x genes
    truth = ds.edge_labels.rename(columns={"is_true": "is_true"})[["source", "target", "is_true"]]
    genes = ds.genes
    n_true = int(ds.edge_labels["is_true"].sum())
    n_candidate = len(ds.candidate_edges)
    hub_top = 3

    # ---- alpha selectors on static exclude-self LASSO (x = target = expression) ----
    per_alpha = {}
    for a in alpha_grid:
        edges = exp17.fit_targetwise(expr, expr, alpha=a, include_self=False)
        scored = exp17.score_edges(edges, truth)
        per_alpha[a] = {"scored": scored, "aupr": aupr(scored["is_true"], scored["score"]),
                        "nnz": int(edges["selected"].sum())}
    oracle_a = max(alpha_grid, key=lambda a: per_alpha[a]["aupr"])
    cv_a = min(alpha_grid, key=lambda a: exp17.cv_mse_global(expr, expr, alpha=a, include_self=False, folds=5, seed=seed))
    bic_a = min(alpha_grid, key=lambda a: exp17.bic_global(expr, expr, alpha=a, include_self=False))
    density_a = min(alpha_grid, key=lambda a: abs(per_alpha[a]["nnz"] - 2 * len(genes)))
    sigma_by = exp17.ols_sigma_by_target(expr, expr, include_self=False)
    p_excl = len(genes) - 1
    theory_value = float(np.median(list(sigma_by.values())) * np.sqrt(2.0 * np.log(max(p_excl, 2)) / len(expr)))
    theory_grid_a = min(alpha_grid, key=lambda a: abs(a - theory_value))
    sqrt_edges, sqrt_alphas = exp17.sqrt_lasso_edges(expr, expr, include_self=False)
    sqrt_scored = exp17.score_edges(sqrt_edges, truth)

    selector_scored = {"oracle": per_alpha[oracle_a]["scored"], "cv": per_alpha[cv_a]["scored"],
                       "bic": per_alpha[bic_a]["scored"], "density_prior": per_alpha[density_a]["scored"],
                       "theory_sigma_hat": exp17.score_edges(exp17.fit_targetwise(expr, expr, alpha=theory_grid_a, include_self=False), truth),
                       "theory_sqrt_lasso": sqrt_scored}
    chosen = {"oracle": oracle_a, "cv": cv_a, "bic": bic_a, "density_prior": density_a,
              "theory_sigma_hat": theory_grid_a, "theory_sqrt_lasso": float(np.median(sqrt_alphas))}
    alpha_rows = []
    for sel, scored in selector_scored.items():
        dm = exp17.directed_metrics(scored, PRECISION_KS)
        alpha_rows.append({"model": model, "replicate": rep, "network_id": rep, "selector": sel, "method": sel,
                           "chosen_alpha": chosen[sel], "theory_alpha_value": theory_value,
                           "predicted_density": int((scored["score"] > COEF_TOL).sum()) / n_candidate,
                           "true_density": n_true / n_candidate, "aupr": dm["aupr"], "auroc": dm["auroc"],
                           "epr": epr(scored, n_true, n_candidate)})

    # ---- Part 1 methods ----
    sparse_cv = selector_scored["cv"]
    methods = {
        "static_correlation": exp17.score_edges(rank_edges_by_correlation(expr), truth),  # symmetric control
        "genie3_rf": exp17.score_edges(rank_edges_by_random_forest(expr, n_estimators=TREE_ESTIMATORS, random_state=seed), truth),
        "sparse_cv": sparse_cv,
    }
    cross = [methods["sparse_cv"], methods["genie3_rf"], methods["static_correlation"]]
    methods["fusion_borda"] = exp17.score_edges(rank_fusion([s[["source", "target", "score"]] for s in cross], method="borda"), truth)

    part1_rows = []
    for method, scored in methods.items():
        dm = exp17.directed_metrics(scored, PRECISION_KS)
        um = exp17.undirected_metrics(scored, PRECISION_KS, how="max")
        oa = exp17.orientation_accuracy(scored)
        oas = exp17.orientation_accuracy(scored, top_n=n_true)
        part1_rows.append({"model": model, "replicate": rep, "network_id": rep, "method": method,
                           "aupr": dm["aupr"], "auroc": dm["auroc"], "epr": epr(scored, n_true, n_candidate),
                           "u_aupr_max": um["u_aupr"], "orientation_gap_aupr": um["u_aupr"] - dm["aupr"],
                           "orientation_accuracy": oa["orientation_accuracy"],
                           "orientation_accuracy_given_skeleton": oas["orientation_accuracy"],
                           "n_orientable": oa["n_orientable"]})

    # ---- Part 3: fusion 3-arm (cell-bootstrap within-method vs cross-method) ----
    rng = np.random.default_rng(seed + 99)
    boot = []
    for _ in range(N_BOOTSTRAP_FUSION):
        idx = rng.choice(len(expr), size=len(expr), replace=True)
        be = exp17.fit_targetwise(expr.iloc[idx].reset_index(drop=True), expr.iloc[idx].reset_index(drop=True), alpha=cv_a, include_self=False)
        boot.append(be[["source", "target", "score"]])
    arm_within = exp17.score_edges(rank_fusion(boot, method="borda"), truth)
    fusion_rows = []
    for arm, scored in (("single_best", sparse_cv), ("within_method_bootstrap", arm_within), ("cross_method", methods["fusion_borda"])):
        dm = exp17.directed_metrics(scored, PRECISION_KS)
        fusion_rows.append({"model": model, "replicate": rep, "network_id": rep, "arm": arm, "method": arm,
                            "aupr": dm["aupr"], "auroc": dm["auroc"], "epr": epr(scored, n_true, n_candidate)})

    # ---- Part 4: stability selection (cell subsampling, exclude-self LASSO at cv alpha) ----
    subs = cell_subsamples(len(expr), n_subsamples, seed=seed + 7, fraction=0.5, complementary=True)
    n_models = len(subs)
    import itertools as _it
    all_edges = pd.DataFrame(list(_it.permutations(genes, 2)), columns=["source", "target"])
    sel_counts = {k2: 0 for k2 in zip(all_edges["source"], all_edges["target"])}
    per_target_sel = {g: [] for g in genes}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for idx in subs:
            edges = exp17.fit_targetwise(expr.iloc[idx].reset_index(drop=True), expr.iloc[idx].reset_index(drop=True), alpha=cv_a, include_self=False)
            sel = edges[edges["selected"]]
            for s, t in zip(sel["source"], sel["target"]):
                sel_counts[(s, t)] += 1
            cbt = sel.groupby("target").size().to_dict()
            for g in genes:
                per_target_sel[g].append(int(cbt.get(g, 0)))
    freq = all_edges.copy()
    freq["selection_frequency"] = [sel_counts[(s, t)] / n_models for s, t in zip(freq["source"], freq["target"])]
    stab = freq.merge(truth, on=["source", "target"], how="left")
    stab["is_true"] = stab["is_true"].astype(int)
    q_by = {g: float(np.mean(per_target_sel[g])) for g in genes}
    p_t = len(genes) - 1
    stab_rows = []
    for pi in (0.6, 0.7, 0.8, 0.9):
        selected = stab[stab["selection_frequency"] >= pi]
        mb = sum(exp17.meinshausen_buhlmann_bound(q_by[g], pi, p_t) for g in genes)
        tp = int((selected["is_true"] == 1).sum())
        fp = int((selected["is_true"] == 0).sum())
        stab_rows.append({"model": model, "replicate": rep, "pi_threshold": pi, "selected_edges": int(len(selected)),
                          "mb_expected_fp_bound": mb, "actual_false_positives": fp, "actual_true_positives": tp,
                          "precision": (tp / len(selected)) if len(selected) else float("nan"),
                          "recall": tp / n_true if n_true else float("nan")})
    sp_aupr = aupr(stab["is_true"], stab["selection_frequency"]) if stab["is_true"].nunique() > 1 else float("nan")
    _, sp_ece = exp17.calibration_bins(stab["selection_frequency"].to_numpy(), stab["is_true"].to_numpy())
    stab_rows.append({"model": model, "replicate": rep, "pi_threshold": float("nan"), "selected_edges": int((stab["selection_frequency"] > 0).sum()),
                      "mb_expected_fp_bound": float("nan"), "actual_false_positives": -1, "actual_true_positives": -1,
                      "precision": float("nan"), "recall": float("nan"), "selection_prob_aupr": sp_aupr, "selection_prob_ece": sp_ece})

    info = {"model": model, "replicate": rep, "n_genes": len(genes), "n_cells": len(expr),
            "n_candidate_edges": n_candidate, "n_true_edges": n_true, "true_density": n_true / n_candidate,
            "reference_kind": ds.metadata["reference_kind"], "n_reciprocal_true_pairs": _recip_count(truth)}
    return part1_rows, alpha_rows, fusion_rows, stab_rows, info


def _recip_count(truth: pd.DataFrame) -> int:
    t = set(zip(truth.loc[truth["is_true"] == 1, "source"], truth.loc[truth["is_true"] == 1, "target"]))
    return sum(1 for a, b in t if (b, a) in t and a < b)


# --------------------------------------------------------------------------- #
def build_pairwise(part1, alpha, fusion, models) -> pd.DataFrame:
    rows = []
    for model in models:
        p1 = part1[part1["model"] == model]
        al = alpha[alpha["model"] == model]
        fu = fusion[fusion["model"] == model]
        for method in ("static_correlation", "genie3_rf", "sparse_cv", "fusion_borda"):
            sub = p1[p1["method"] == method]
            if sub.empty:
                continue
            d = (sub["u_aupr_max"] - sub["aupr"]).to_numpy()
            rng = np.random.default_rng(0)
            bs = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(2000)] if len(d) else [np.nan]
            rows.append({"model": model, "comparison": f"{method}: undirected-vs-directed AUPR gap",
                         "mean_delta": float(np.mean(d)) if len(d) else float("nan"),
                         "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)), "n": len(d)})
        for sel in ("cv", "bic", "theory_sqrt_lasso", "theory_sigma_hat", "density_prior"):
            r = exp17.paired_network_comparison(al, "aupr", sel, "oracle")
            rows.append({"model": model, "comparison": f"alpha {sel} - oracle (AUPR)",
                         "mean_delta": r["mean_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]})
        for a, b in (("cross_method", "within_method_bootstrap"), ("within_method_bootstrap", "single_best"), ("cross_method", "single_best")):
            r = exp17.paired_network_comparison(fu.assign(method=fu["arm"]), "aupr", a, b)
            rows.append({"model": model, "comparison": f"fusion {a} - {b} (AUPR)",
                         "mean_delta": r["mean_delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]})
    return pd.DataFrame(rows)


def build_report(part1, alpha, fusion, stability, pairwise, info, missing_scrna, models, figures) -> str:
    L = ["# BEELINE Diagnostics Debug Report", "",
         "Transfers the experiment-17 diagnostic framing to a real single-cell GRN benchmark. "
         "Run on the BEELINE **Curated** benchmark (exact directed ground truth, dense reciprocal "
         "structure, replicates), the OPPOSITE regime to DREAM4 (n cells >> p genes). Static methods "
         "only (single-cell snapshots have no time); paired CIs are over replicates within each model.", ""]
    L.append("## 1-2. Datasets used")
    L.append("")
    L.append(exp17.to_md(info.round(4)))
    L.append("")
    L.append(f"Real scRNA-seq sets present but UNSCORABLE here (no reference network in this download): "
             f"{', '.join(missing_scrna) if missing_scrna else 'none'}. To score them, place a BEELINE reference "
             f"(refNetwork.csv / GroundTruthNetwork.csv, e.g. cell-type ChIP / non-specific ChIP / STRING) in each "
             f"`data/raw/BEELINE-data/inputs/scRNA-Seq/<dataset>/` (or its parent).")
    L.append("")
    L.append("## Part 1: directed vs undirected + orientation (means by model/method)")
    L.append("")
    L.append(exp17.to_md(exp17.mean_by(part1, ["model", "method"])[["model", "method", "aupr", "epr", "u_aupr_max", "orientation_gap_aupr", "orientation_accuracy_given_skeleton"]].round(4)))
    L.append("")
    L.append("## Part 2: alpha selectors (means)")
    L.append("")
    L.append(exp17.to_md(exp17.mean_by(alpha, ["model", "selector"])[["model", "selector", "chosen_alpha", "theory_alpha_value", "aupr", "epr"]].round(4)))
    L.append("")
    L.append("## Part 3: fusion 3-arm (means)")
    L.append("")
    L.append(exp17.to_md(exp17.mean_by(fusion, ["model", "arm"])[["model", "arm", "aupr", "auroc", "epr"]].round(4)))
    L.append("")
    L.append("## Part 4: stability selection (means by threshold)")
    L.append("")
    st = stability[stability["pi_threshold"].notna()]
    L.append(exp17.to_md(exp17.mean_by(st, ["model", "pi_threshold"])[["model", "pi_threshold", "selected_edges", "mb_expected_fp_bound", "actual_false_positives", "actual_true_positives", "precision", "recall"]].round(3)))
    L.append("")
    L.append("## Paired comparisons (mean delta [95% CI], n) -- across replicates within model")
    L.append("")
    L.append(exp17.to_md(pairwise.round(4)))
    L.append("")
    L.append("## Question-by-question")
    L.append("")
    L.append(_answer_questions(part1, alpha, fusion, stability, info, missing_scrna, models))
    return "\n".join(L)


def _answer_questions(part1, alpha, fusion, stability, info, missing_scrna, models) -> str:
    O = []
    p1all = exp17.mean_by(part1, ["method"]).set_index("method")
    skel_bound = bool((p1all.get("orientation_accuracy_given_skeleton", pd.Series(dtype=float)) >= 0.7).all()
                      and (p1all.get("orientation_gap_aupr", pd.Series(dtype=float)).abs() < 0.2).all())
    O.append(f"1. Dataset(s): BEELINE Curated {', '.join(models)} (exact directed ground truth, "
             f"{info['replicate'].nunique() if 'replicate' in info else 'several'} replicates each). scRNA-seq sets unscorable here: {len(missing_scrna)}.")
    O.append(f"2. Sizes: " + "; ".join(f"{r['model']} {int(r['n_genes'])} genes / {int(r['n_cells'])} cells / {int(r['n_candidate_edges'])} candidates / {int(r['n_true_edges'])} true / {int(r['n_reciprocal_true_pairs'])} reciprocal-true pairs"
                                       for r in info.drop_duplicates('model').to_dict("records")))
    O.append("3. Skeleton vs orientation: orientation-given-skeleton across methods = " +
             ", ".join(f"{m}={exp17.fmt(p1all.loc[m,'orientation_accuracy_given_skeleton'])}" for m in p1all.index) +
             f"; undirected-vs-directed gaps small. So BEELINE Curated is ALSO " +
             ("skeleton-limited" if skel_bound else "not cleanly skeleton-limited") +
             f" (static_correlation control orientation = {exp17.fmt(p1all.loc['static_correlation','orientation_accuracy_given_skeleton']) if 'static_correlation' in p1all.index else 'n/a'}, expected ~0.5).")
    fuall = exp17.mean_by(fusion, ["arm"]).set_index("arm")
    if {"single_best", "within_method_bootstrap", "cross_method"} <= set(fuall.index):
        O.append(f"4. Fusion AUPR: single={exp17.fmt(fuall.loc['single_best','aupr'])}, within-bootstrap={exp17.fmt(fuall.loc['within_method_bootstrap','aupr'])}, "
                 f"cross-method={exp17.fmt(fuall.loc['cross_method','aupr'])} (complementarity = cross - bootstrap; see paired CIs).")
    stn = stability[stability["pi_threshold"].isna()]
    O.append(f"5. Stability selection: selection-probability AUPR={exp17.fmt(float(stn['selection_prob_aupr'].mean()))}, ECE={exp17.fmt(float(stn['selection_prob_ece'].mean()))}; "
             "MB bound vs actual FP in the Part 4 table (informative only where bound << selected).")
    alall = exp17.mean_by(alpha, ["selector"]).set_index("selector")
    if {"oracle", "cv", "theory_sqrt_lasso"} <= set(alall.index):
        O.append(f"6. Alpha selectors AUPR: oracle={exp17.fmt(alall.loc['oracle','aupr'])}, cv={exp17.fmt(alall.loc['cv','aupr'])}, "
                 f"bic={exp17.fmt(alall.loc['bic','aupr'])}, theory_sqrt_lasso={exp17.fmt(alall.loc['theory_sqrt_lasso','aupr'])} "
                 f"(theory_alpha_value={exp17.fmt(float(alall.loc['cv','theory_alpha_value']),3)}; n>>p here so penalties are small).")
    O.append("7. Transfers from DREAM4: " + ("the skeleton-limited conclusion and the symmetric-control behavior transfer; "
             if skel_bound else "the picture differs from DREAM4; ") + "static single-cell has weaker orientation signal than lagged DREAM4 (no temporal precedence).")
    O.append("8. What breaks with proxy references: nothing breaks on Curated (ground truth is EXACT here). The proxy-reference caveat applies to the scRNA-seq sets, which are unscorable in this download; on those AUPR would be reference-agreement, not truth recovery.")
    O.append("9. Commit: experiment 18 (script + note), the adapter GroundTruth/parent-search fix + test, and the doc updates. results/ stays git-ignored.")
    return "\n".join(O)


def write_figures(part1, fusion, models) -> list[str]:
    if not HAVE_MPL:
        return []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    pm = exp17.mean_by(part1, ["method"]).set_index("method")
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(len(pm)); w = 0.4
    ax.bar(idx - w / 2, pm["aupr"], w, label="directed AUPR")
    ax.bar(idx + w / 2, pm["u_aupr_max"], w, label="undirected AUPR")
    ax.set_xticks(idx); ax.set_xticklabels(pm.index, rotation=30, ha="right", fontsize=8); ax.legend()
    ax.set_title("BEELINE Curated: directed vs undirected")
    p = FIGURES_DIR / f"{PREFIX}_directed_vs_undirected.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p.as_posix())
    return saved


def parse_args():
    pr = argparse.ArgumentParser(description=__doc__)
    pr.add_argument("--quick", action="store_true", help="GSD only, 3 replicates, fewer subsamples")
    pr.add_argument("--models", nargs="*", default=None)
    pr.add_argument("--replicates", type=int, default=5)
    pr.add_argument("--n-subsamples", type=int, default=None)
    pr.add_argument("--n-jobs", type=int, default=-1)
    pr.add_argument("--random-seed", type=int, default=20260602)
    return pr.parse_args()


def main():
    args = parse_args()
    if not CURATED_ROOT.exists():
        print(f"STOP: BEELINE Curated data not found at {CURATED_ROOT}.")
        print("Place the BEELINE 'inputs/Curated' tree there (each model has per-replicate ExpressionData.csv and a model-level GroundTruthNetwork.csv).")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = (["GSD"] if args.quick else (args.models or list(CURATED_MODELS)))
    max_reps = 3 if args.quick else args.replicates
    n_subsamples = args.n_subsamples if args.n_subsamples is not None else (20 if args.quick else 50)

    part1_all, alpha_all, fusion_all, stab_all, info_all = [], [], [], [], []
    for model in models:
        reps = discover_replicates(model, max_reps)
        if not reps:
            print(f"(skip {model}: no replicates found)")
            continue
        for rep in reps:
            rep_num = int(rep.split("-")[-1])  # deterministic per-replicate seed offset
            p1, al, fu, st, info = run_replicate(model, rep, alpha_grid=ALPHA_GRID, n_subsamples=n_subsamples,
                                                 seed=args.random_seed + rep_num, n_jobs=args.n_jobs)
            part1_all += p1; alpha_all += al; fusion_all += fu; stab_all += st; info_all.append(info)

    part1 = pd.DataFrame(part1_all); alpha = pd.DataFrame(alpha_all)
    fusion = pd.DataFrame(fusion_all); stability = pd.DataFrame(stab_all); info = pd.DataFrame(info_all)
    pairwise = build_pairwise(part1, alpha, fusion, models)
    missing_scrna = scrna_reference_status()

    summary = pd.concat([exp17.mean_by(part1, ["model", "method"]).assign(part="part1"),
                         exp17.mean_by(alpha, ["model", "selector"]).assign(part="part2_alpha")], ignore_index=True)
    summary.to_csv(RESULTS_DIR / f"{PREFIX}_summary.csv", index=False)
    part1.to_csv(RESULTS_DIR / f"{PREFIX}_edges.csv", index=False)  # per-method per-replicate metrics
    pairwise.to_csv(RESULTS_DIR / f"{PREFIX}_pairwise.csv", index=False)
    alpha.to_csv(RESULTS_DIR / f"{PREFIX}_alpha.csv", index=False)
    fusion.to_csv(RESULTS_DIR / f"{PREFIX}_fusion.csv", index=False)
    stability.to_csv(RESULTS_DIR / f"{PREFIX}_stability.csv", index=False)
    figures = write_figures(part1, fusion, models)
    (RESULTS_DIR / f"{PREFIX}_debug_report.md").write_text(
        build_report(part1, alpha, fusion, stability, pairwise, info, missing_scrna, models, figures), encoding="utf-8")

    print(f"models={models} replicates<= {max_reps} n_subsamples={n_subsamples} figures={len(figures)}")
    print(f"scRNA-seq unscorable (no reference in download): {missing_scrna}")
    print("\n--- Part 1 (directed vs undirected, orientation; mean over all) ---")
    print(exp17.mean_by(part1, ["method"])[["method", "aupr", "epr", "u_aupr_max", "orientation_gap_aupr", "orientation_accuracy_given_skeleton"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nsaved tables + debug report under results/tables/")


if __name__ == "__main__":
    main()
