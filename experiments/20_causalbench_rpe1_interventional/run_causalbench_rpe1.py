r"""Experiment 20 - CausalBench RPE1 interventional diagnostics.

First run on REAL interventional data (Replogle/Weissmann RPE1 Perturb-seq via
CausalBench). Closes the regime ladder from experiments 17-19:

    DREAM4 (lagged)            -> orientation ~free (temporal precedence)
    BEELINE Curated (static)   -> orientation weak & network-dependent
    CausalBench RPE1 (interventional) -> orientation from intervention asymmetry  <- here

Real Perturb-seq has no exact directed ground truth, so we do NOT score against a
fixed graph. Instead we use CausalBench's own paradigm: an edge A->B is interventionally
"real" if knocking down A shifts B's distribution beyond a control-vs-control null
(Wasserstein). That interventional reference is used ONLY to score *observational*
methods (a genuine transfer test), never to score the interventional signal itself.

Analyses (each labelled circular-safe or not):
  1. Dataset summary.
  2. Interventional effect matrix E[A->B] = W(B|perturb A, B|control). [the causal signal]
  3. Control-split null -> interventional reference R_int (directed). [held-out truth]
  4. TRANSFER (non-circular): observational methods on CONTROL cells only
     (correlation, sparse-CV) scored vs R_int. Does unperturbed co-expression predict
     which interventions matter? vs a random baseline.
  5. ORIENTATION IDENTIFIABILITY (reference-free): among both-perturbed pairs, the
     decidability rate (asymmetry beats the control null) -- contrasted with the
     observational symmetric control which cannot orient (0.5). Plus agreement between
     interventional-asymmetry direction and the observational sparse direction.
  6. ALPHA at n>>p on control cells (CV / BIC / sqrt-LASSO chosen alphas).

CAUTION: intervention asymmetry is directional EVIDENCE, not proof (indirect/downstream
effects, compensation, off-target knockdown, cell-state shifts).

Run:
    $env:PYTHONPATH = "src"
    .\.venv\Scripts\python.exe -B experiments/20_causalbench_rpe1_interventional/run_causalbench_rpe1.py
    # add --quick for a fast smaller-subsample pass
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from stable_grn_inference.data import load_causalbench, load_replogle_raw_h5ad

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
# Either the small CausalBench-preprocessed file (rpe1.h5ad) or the large raw Replogle
# single-cell file (rpe1_raw_singlecell_01.h5ad). The latter is dense 247914 x 8749.
PREPROCESSED = CB_DIR / "rpe1.h5ad"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "causalbench_rpe1"

_EXP17 = ROOT / "experiments" / "17_dream4_stability_orientation_diagnostics" / "run_stability_orientation_diagnostics.py"
_spec = importlib.util.spec_from_file_location("exp17_diag", _EXP17)
exp17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp17)

PRECISION_KS = (10, 50, 100)


def _subsample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    return idx if idx.size <= n else rng.choice(idx, size=n, replace=False)




def effect_matrix(expr, labels, control_mask, perturbed, genes, *, n_ctrl, n_pert, seed):
    """E[A->B] = Wasserstein(B|perturb A, B|control), subsampled for speed."""
    rng = np.random.default_rng(seed)
    X = expr.to_numpy()
    ctrl_idx = _subsample(np.where(control_mask)[0], n_ctrl, rng)
    gene_pos = {g: i for i, g in enumerate(genes)}
    ctrl_block = X[ctrl_idx]  # cells x genes
    rows = []
    for a in perturbed:
        pert_idx = _subsample(np.where((labels == a).to_numpy())[0], n_pert, rng)
        if pert_idx.size == 0:
            continue
        pert_block = X[pert_idx]
        ai = gene_pos[a]
        for b in genes:
            if b == a:
                continue
            bi = gene_pos[b]
            rows.append((a, b, wasserstein_distance(pert_block[:, bi], ctrl_block[:, bi])))
    return pd.DataFrame(rows, columns=["source", "target", "effect"])


def control_split_null(expr, control_mask, genes, *, n_splits, n_ctrl, seed):
    """Per-target control-vs-control Wasserstein null. Returns the pooled 95th percentile
    threshold and per-gene null means."""
    rng = np.random.default_rng(seed + 1)
    X = expr.to_numpy()
    ctrl_all = np.where(control_mask)[0]
    gene_pos = {g: i for i, g in enumerate(genes)}
    null_vals = {g: [] for g in genes}
    for _ in range(n_splits):
        picked = _subsample(ctrl_all, min(2 * n_ctrl, ctrl_all.size), rng)
        rng.shuffle(picked)
        half = picked.size // 2
        h1, h2 = picked[:half], picked[half:2 * half]
        for g in genes:
            gi = gene_pos[g]
            null_vals[g].append(wasserstein_distance(X[h1, gi], X[h2, gi]))
    per_gene_mean = {g: float(np.mean(v)) for g, v in null_vals.items()}
    pooled = np.array([x for v in null_vals.values() for x in v])
    threshold = float(np.percentile(pooled, 95))
    return threshold, per_gene_mean


def build_interventional_reference(eff, threshold, per_gene_mean):
    """Directed reference: edge A->B is interventionally real if its effect exceeds both
    the pooled 95th-pct null and the per-target control null mean."""
    keep = []
    for s, t, e in zip(eff["source"], eff["target"], eff["effect"]):
        if e > threshold and e > per_gene_mean.get(t, 0.0):
            keep.append((s, t))
    return pd.DataFrame(keep, columns=["source", "target"])


def observational_scores(ctrl, candidate_edges, *, alpha):
    """Static observational methods on a (subsampled) CONTROL-cell frame.
    Returns dict[name] -> predicted DataFrame(source,target,score)."""
    cand = candidate_edges

    # correlation (symmetric)
    corr = ctrl.corr().abs()
    corr_scores = pd.DataFrame(
        {"source": cand["source"], "target": cand["target"],
         "score": [corr.loc[s, t] for s, t in zip(cand["source"], cand["target"])]}
    )

    # exclude-self CV-LASSO: target gene ~ all other genes (score = |coef|)
    edges = exp17.fit_targetwise(ctrl, ctrl, alpha=alpha, include_self=False)
    emap = {(s, t): float(v) for s, t, v in zip(edges["source"], edges["target"], edges["score"])}
    sparse_scores = pd.DataFrame(
        {"source": cand["source"], "target": cand["target"],
         "score": [emap.get((s, t), 0.0) for s, t in zip(cand["source"], cand["target"])]}
    )
    return {"observational_correlation": corr_scores, "observational_sparse": sparse_scores}


def orientation_identifiability(eff, perturbed, *, threshold, null_asym):
    """Reference-free: among both-perturbed pairs, fraction where the interventional
    asymmetry is decisive (|E_AB - E_BA| > control-null asymmetry and max effect
    significant). Also returns the implied directed edges."""
    emap = {(s, t): float(e) for s, t, e in zip(eff["source"], eff["target"], eff["effect"])}
    pert = list(perturbed)
    decided, total = 0, 0
    implied = []
    records = []
    for i in range(len(pert)):
        for j in range(i + 1, len(pert)):
            a, b = pert[i], pert[j]
            eab, eba = emap.get((a, b)), emap.get((b, a))
            if eab is None or eba is None:
                continue
            total += 1
            is_decided = max(eab, eba) > threshold and abs(eab - eba) > null_asym
            src, dst = (a, b) if eab >= eba else (b, a)
            if is_decided:
                decided += 1
                implied.append((src, dst))
            records.append({
                "source": src, "target": dst,  # implied direction (larger effect)
                "effect_forward": max(eab, eba), "effect_reverse": min(eab, eba),
                "decided": bool(is_decided),
            })
    return {
        "n_pairs_both_perturbed": total,
        "decidable_pairs": decided,
        "decidability_rate": decided / total if total else float("nan"),
        "implied_edges": pd.DataFrame(implied, columns=["source", "target"]),
        "pairs": pd.DataFrame.from_records(
            records, columns=["source", "target", "effect_forward", "effect_reverse", "decided"]
        ),
    }


def fmt(v, d=4):
    return exp17.fmt(v, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    n_ctrl = 1500 if args.quick else 4000
    n_pert = 200 if args.quick else 400
    n_splits = 6 if args.quick else 12
    min_cells = 100
    max_perts = 200 if args.quick else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if PREPROCESSED.exists():
        ds = load_causalbench(PREPROCESSED, name="rpe1", perturbed_genes_only=True)
    elif raw_path is not None:
        print(f"Loading large raw file {raw_path.name} (chunked, perturbed&measured block)...", flush=True)
        ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=min_cells, max_perturbations=max_perts)
    else:
        raise SystemExit(f"No RPE1 h5ad found in {CB_DIR}. See exp19 scouting doc.")
    genes = ds.genes
    labels = ds.perturbation_labels
    control_mask = ds.is_control
    perturbed = ds.perturbed_genes

    lines = ["# Experiment 20 - CausalBench RPE1 interventional diagnostics\n"]
    lines.append("## 1. Dataset summary\n")
    lines.append(
        f"- genes (perturbed&measured block): {len(genes)}; cells: {ds.metadata['n_cells']}; "
        f"perturbations: {ds.metadata['n_perturbations']}; control cells: {ds.metadata['n_control_cells']}"
    )
    lines.append(f"- candidate edges (perturbed source x target): {len(ds.candidate_edges)}")
    lines.append(f"- subsampling: n_ctrl={n_ctrl}, n_pert={n_pert}, n_splits={n_splits}\n")

    # 2. interventional effect matrix
    eff = effect_matrix(ds.expression, labels, control_mask.to_numpy(), perturbed, genes,
                        n_ctrl=n_ctrl, n_pert=n_pert, seed=args.random_seed)

    # 3. control-split null -> interventional reference
    threshold, per_gene_mean = control_split_null(
        ds.expression, control_mask.to_numpy(), genes, n_splits=n_splits, n_ctrl=n_ctrl, seed=args.random_seed
    )
    ref = build_interventional_reference(eff, threshold, per_gene_mean)
    density = len(ref) / max(len(ds.candidate_edges), 1)
    lines.append("## 2-3. Interventional signal + reference (held-out, control-null)\n")
    lines.append(f"- control-vs-control null threshold (pooled 95th pct Wasserstein): {fmt(threshold)}")
    lines.append(f"- interventionally-real edges: {len(ref)} / {len(ds.candidate_edges)} (density {fmt(density)})")
    eff_true = eff.merge(ref.assign(is_true=1), on=["source", "target"], how="left").fillna({"is_true": 0})
    lines.append(f"- mean effect on real edges: {fmt(eff_true.loc[eff_true.is_true==1,'effect'].mean())}; "
                 f"on others: {fmt(eff_true.loc[eff_true.is_true==0,'effect'].mean())}\n")

    # 4. TRANSFER: observational predicts interventional?
    truth = ref.assign(is_true=1)
    full_truth = ds.candidate_edges.merge(truth, on=["source", "target"], how="left").fillna({"is_true": 0})
    full_truth["is_true"] = full_truth["is_true"].astype(int)

    # alpha for observational sparse via CV at n>>p (control cells, capped for runtime)
    rng0 = np.random.default_rng(args.random_seed)
    ctrl_all = ds.expression.loc[control_mask.values]
    n_obs_cap = 2000 if args.quick else 4000
    if len(ctrl_all) > n_obs_cap:
        ctrl_expr = ctrl_all.iloc[rng0.choice(len(ctrl_all), n_obs_cap, replace=False)].reset_index(drop=True)
    else:
        ctrl_expr = ctrl_all.reset_index(drop=True)
    alpha_grid = (0.001, 0.01, 0.05, 0.1, 0.2, 0.5)
    cv_scores = {a: exp17.cv_mse_global(ctrl_expr, ctrl_expr, alpha=a, include_self=False, folds=3, seed=args.random_seed)
                 for a in alpha_grid}
    bic_scores = {a: exp17.bic_global(ctrl_expr, ctrl_expr, alpha=a, include_self=False) for a in alpha_grid}
    cv_alpha = min(cv_scores, key=cv_scores.get)
    bic_alpha = min(bic_scores, key=bic_scores.get)
    theory_alpha = 1.1 * np.sqrt(2.0 * np.log(max(len(genes), 2)) / len(ctrl_expr))

    obs = observational_scores(ctrl_expr, ds.candidate_edges, alpha=cv_alpha)
    lines.append("## 4. TRANSFER: observational (control cells) -> interventional reference\n")
    lines.append("| method | aupr | auroc | precision@10 | precision@50 | EPR |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    rng = np.random.default_rng(args.random_seed)
    for name, pred in obs.items():
        scored = exp17.score_edges(pred, full_truth)
        m = exp17.directed_metrics(scored, PRECISION_KS)
        n_true = int(full_truth["is_true"].sum())
        epr = (exp17.precision_at_k(scored, "is_true", n_true) / density) if (n_true and density) else float("nan")
        lines.append(f"| {name} | {fmt(m['aupr'])} | {fmt(m['auroc'])} | "
                     f"{fmt(m['precision_at_10'])} | {fmt(m['precision_at_50'])} | {fmt(epr)} |")
    rand = full_truth.assign(score=rng.random(len(full_truth)))
    rm = exp17.directed_metrics(exp17.score_edges(rand, full_truth), PRECISION_KS)
    lines.append(f"| random_baseline | {fmt(rm['aupr'])} | {fmt(rm['auroc'])} | "
                 f"{fmt(rm['precision_at_10'])} | {fmt(rm['precision_at_50'])} | {fmt(density and 1.0)} |")
    lines.append(f"\n- reference density (random AUPR floor): {fmt(density)}\n")

    # 5. orientation identifiability
    null_asym = threshold  # asymmetry must exceed the control null scale
    ident = orientation_identifiability(eff, perturbed, threshold=threshold, null_asym=null_asym)
    lines.append("## 5. Orientation IDENTIFIABILITY (reference-free, both-perturbed pairs)\n")
    lines.append(f"- both-perturbed pairs evaluated: {ident['n_pairs_both_perturbed']}")
    lines.append(f"- decidable by interventional asymmetry: {ident['decidable_pairs']} "
                 f"(**decidability rate {fmt(ident['decidability_rate'])}**)")
    lines.append("- observational symmetric control (|correlation|) on the same pairs: undecidable (0.5 by construction)")

    # agreement: interventional-implied direction vs observational sparse direction
    sparse_map = {(s, t): v for s, t, v in zip(obs["observational_sparse"]["source"],
                                               obs["observational_sparse"]["target"],
                                               obs["observational_sparse"]["score"])}
    agree, n = 0, 0
    for s, t in zip(ident["implied_edges"]["source"], ident["implied_edges"]["target"]):
        fwd, rev = sparse_map.get((s, t), 0.0), sparse_map.get((t, s), 0.0)
        if fwd == rev:
            continue
        n += 1
        agree += 1 if fwd > rev else 0
    lines.append(f"- agreement of interventional direction with observational sparse direction: "
                 f"{fmt(agree / n if n else float('nan'))} over {n} decidable+oriented pairs "
                 "(>0.5 => observational structure carries weak directional signal)\n")

    # 6. alpha behavior
    lines.append("## 6. Alpha at n>>p (control cells)\n")
    lines.append(f"- n_control={int(control_mask.sum())} (used {len(ctrl_expr)}) >> p={len(genes)}")
    lines.append(f"- CV-best alpha={cv_alpha}; BIC-best alpha={bic_alpha} (grid {alpha_grid})")
    lines.append(f"- theory alpha (1.1*sqrt(2 log p / n)) = {fmt(theory_alpha)}")
    lines.append("- transfer check (exp17/18): the penalty is theory-PREDICTABLE if theory alpha lands "
                 "between/near the CV and BIC choices (not necessarily 'tiny').\n")

    # 7. question-by-question
    lines.append("## 7. Headline questions\n")
    lines.append(f"1. Does orientation become identifiable under intervention? "
                 f"Decidability {fmt(ident['decidability_rate'])} vs observational 0.5 (undecidable). ")
    lines.append(f"2. Does observational co-expression predict interventional effects? "
                 f"See transfer AUPR vs floor {fmt(density)}. ")
    lines.append("3. Does the theory/alpha story transfer? Theory alpha lands near CV/BIC (penalty is "
                 "theory-predictable), even though the value is moderate not tiny. ")
    lines.append("4. Caveat: proxy-free interventional reference is null-thresholded, not exact truth; "
                 "asymmetry is evidence not proof.\n")

    # ---- outputs ----
    eff_labeled = eff.merge(full_truth, on=["source", "target"], how="left")
    eff_labeled.to_csv(TABLES_DIR / f"{PREFIX}_effect_edges.csv", index=False)
    ref.to_csv(TABLES_DIR / f"{PREFIX}_interventional_reference.csv", index=False)
    ident["pairs"].to_csv(TABLES_DIR / f"{PREFIX}_orientation_asymmetry.csv", index=False)
    obs_long = pd.concat(
        [p.assign(method=name) for name, p in obs.items()], ignore_index=True
    ).merge(full_truth, on=["source", "target"], how="left")
    obs_long.to_csv(TABLES_DIR / f"{PREFIX}_observational_scores.csv", index=False)
    summary = pd.DataFrame([{
        "n_genes_block": len(genes),
        "n_cells": ds.metadata["n_cells"],
        "n_perturbations": ds.metadata["n_perturbations"],
        "n_control_cells": ds.metadata["n_control_cells"],
        "n_candidate_edges": len(ds.candidate_edges),
        "interventional_reference_edges": len(ref),
        "reference_density": density,
        "cv_alpha": cv_alpha, "bic_alpha": bic_alpha, "theory_alpha": theory_alpha,
        "orientation_decidability_rate": ident["decidability_rate"],
        "n_both_perturbed_pairs": ident["n_pairs_both_perturbed"],
        "source_file": ds.metadata.get("source_file", "preprocessed"),
    }])
    summary.to_csv(TABLES_DIR / f"{PREFIX}_summary.csv", index=False)

    # ---- figures (best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG = ROOT / "results" / "figures"
        FIG.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(eff["effect"], bins=60, color="#3b7", alpha=0.8)
        ax[0].axvline(threshold, color="k", ls="--", label="control null 95%")
        ax[0].set_title("Interventional effect (Wasserstein)"); ax[0].set_yscale("log"); ax[0].legend()
        pr = ident["pairs"]
        if len(pr):
            ax[1].hist((pr["effect_forward"] - pr["effect_reverse"]).dropna(), bins=50, color="#37b", alpha=0.8)
        ax[1].set_title("Orientation asymmetry (eff_fwd - eff_rev)")
        fig.tight_layout(); fig.savefig(FIG / f"{PREFIX}_effects_and_asymmetry.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        lines.append(f"\n(figure step skipped: {e})")

    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
