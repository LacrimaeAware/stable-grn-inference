r"""Experiment 23 - Response inverse / deconvolution ("solve the flow field for the stick").

The observed perturbation response D is a TOTAL effect (direct + propagated + global).
In a simplified linear-propagation model, a sparse DIRECT operator W generates it as
    D = (I - W)^{-1} - I = W + W^2 + ...,  exact inverse  W = I - (I + D)^{-1}.
Question: can any simple inverse/deconvolution turn the dense total response D into a
sparser, more stable, more direct-effect-like operator W?

Expected behavior:
  - Synthetic (model true): the noiseless inverse recovers W exactly and degrades
    gradually with noise.
  - Real RPE1: matrix inversion amplifies noise, so the inverse is expected to be less
    split-half stable than the raw |D| baseline and not to reconstruct held-out response
    better. The open question is whether a sparse (Lasso) deconvolution is more stable.

Bounded, exploratory. No new data, no wavelets, no RL, no neural nets. --quick caps size.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from stable_grn_inference.data import (
    deconvolve_response,
    load_replogle_raw_h5ad,
    make_sparse_dag,
    operator_edges,
    perturbation_response_matrix,
    propagation_forward,
)

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "data" / "raw" / "causalbench"
RAW_CANDIDATES = ("rpe1_raw_singlecell_01.h5ad", "rpe1_raw_singlecell.h5ad")
TABLES_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
PREFIX = "causalbench_response_inverse"


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}"


# ----------------------------- shared metrics -----------------------------

def topk_density(W, k_per_source):
    """Fraction of off-diagonal entries kept if we keep the top-k per source row."""
    n = W.shape[0]
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    kept = 0
    for i in range(n):
        row = A[i]
        kept += min(k_per_source, int((row > 0).sum()))
    return kept / (n * (n - 1))


def participation_ratio(W):
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum()) if s2.sum() > 0 else float("nan")


def edge_score_vector(W):
    A = np.abs(W).copy()
    np.fill_diagonal(A, np.nan)
    return A[~np.isnan(A)]


def asymmetry_directions(W):
    """For each unordered pair, implied source = larger |W|. Returns dict pair->0/1 (i<j: 1 if i->j)."""
    n = W.shape[0]
    A = np.abs(W)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            out[(i, j)] = 1 if A[i, j] >= A[j, i] else 0
    return out


# ----------------------------- PART 0: synthetic -----------------------------

def synthetic_sanity(n_genes, density, noise_levels, seed):
    rows = []
    W_true = make_sparse_dag(n_genes, density, seed=seed)
    labels = [f"g{i}" for i in range(n_genes)]
    true_edge = (np.abs(W_true) > 0).astype(int)
    iu = ~np.eye(n_genes, dtype=bool)
    y_true = true_edge[iu]
    D_clean = propagation_forward(W_true)
    rng = np.random.default_rng(seed + 99)
    methods = [("ridge", dict(method="ridge", lam=0.0)),
               ("ridge_l0.1", dict(method="ridge", lam=0.1)),
               ("pinv", dict(method="pinv", rcond=1e-3))]
    for noise in noise_levels:
        D = D_clean + rng.normal(0.0, noise * np.std(D_clean), size=D_clean.shape) if noise > 0 else D_clean
        for name, kw in methods:
            W_rec = deconvolve_response(D, **kw)
            score = np.abs(W_rec)[iu]
            raw = np.abs(D)[iu]
            rows.append({
                "noise": noise, "method": name,
                "aupr": float(average_precision_score(y_true, score)),
                "auroc": float(roc_auc_score(y_true, score)),
                "aupr_raw_D": float(average_precision_score(y_true, raw)),
                "auroc_raw_D": float(roc_auc_score(y_true, raw)),
                "recover_err": float(np.abs(W_rec - W_true).max()),
            })
    return pd.DataFrame(rows), W_true, D_clean


# ----------------------------- PART 1-3: real -----------------------------

def sparse_deconvolution(D, alpha):
    """Sparse W with D ~= W (I+D): per-row Lasso of D[i,:] on (I+D)^T. Zeroes self."""
    from sklearn.linear_model import Lasso

    n = D.shape[0]
    M = np.eye(n) + D
    A = M.T  # design: y[j] = sum_k w[k] M[k,j] -> y = A w with A = M.T
    W = np.zeros((n, n))
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i in range(n):
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
            model.fit(A, D[i, :])
            W[i, :] = model.coef_
    np.fill_diagonal(W, 0.0)
    return W


def reconstruction_metrics(W_from_split1, D_split2):
    """Predict split2's total response from split1's operator and compare."""
    D_hat = propagation_forward(W_from_split1)
    iu = ~np.eye(D_split2.shape[0], dtype=bool)
    a, b = D_hat[iu], D_split2[iu]
    fro = float(np.linalg.norm(D_hat - D_split2) / np.linalg.norm(D_split2))
    rank = float(spearmanr(a, b).statistic)
    # row-wise cosine
    num = (D_hat * D_split2).sum(1)
    den = np.linalg.norm(D_hat, axis=1) * np.linalg.norm(D_split2, axis=1)
    cos = float(np.nanmean(np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)))
    return {"frob_rel": fro, "rank_corr": rank, "row_cosine": cos}


def split_half_operator_stability(W1, W2):
    s1, s2 = edge_score_vector(W1), edge_score_vector(W2)
    rank = float(spearmanr(s1, s2).statistic)
    # top-5%-per... use global top-k overlap
    k = max(1, int(0.02 * len(s1)))
    top1 = set(np.argsort(s1)[-k:]); top2 = set(np.argsort(s2)[-k:])
    overlap = len(top1 & top2) / k
    return {"edge_rank_corr": rank, "top2pct_overlap": overlap}


def direction_reproducibility(W1, W2):
    d1, d2 = asymmetry_directions(W1), asymmetry_directions(W2)
    agree = sum(int(d1[k] == d2[k]) for k in d1) / len(d1)
    return agree


def global_mode_alignment(W, D):
    uW = np.linalg.svd(W, full_matrices=False)[2][0]
    uD = np.linalg.svd(D, full_matrices=False)[2][0]
    return float(abs(np.dot(uW, uD)) / (np.linalg.norm(uW) * np.linalg.norm(uD)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    max_perts = 200 if args.quick else None
    do_sparse = not args.quick
    n_syn = 30 if args.quick else 50

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 23 - Response inverse / deconvolution\n"]
    lines.append("_Expected: the inverse recovers W on noiseless synthetic data; on real RPE1 "
                 "the inverse is not expected to beat raw |D| (inversion amplifies noise)._\n")

    # ---- PART 0: synthetic ----
    syn, W_true, D_clean = synthetic_sanity(n_syn, 0.12, [0.0, 0.05, 0.1, 0.25, 0.5], args.random_seed)
    syn.to_csv(TABLES_DIR / f"{PREFIX}_synthetic_summary.csv", index=False)
    lines.append("## Part 0: synthetic sanity (model is TRUE)\n")
    lines.append(f"- W_true: {n_syn} genes, sparse DAG; D = (I-W)^-1 - I\n")
    lines.append("| noise | method | recover_err | aupr(W) | auroc(W) | aupr(raw D) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in syn.iterrows():
        lines.append(f"| {r['noise']} | {r['method']} | {fmt(r['recover_err'])} | {fmt(r['aupr'])} | "
                     f"{fmt(r['auroc'])} | {fmt(r['aupr_raw_D'])} |")
    clean = syn[(syn.noise == 0.0) & (syn.method == "ridge")].iloc[0]
    syn_ok = clean["recover_err"] < 1e-6
    # does inverse beat raw D at recovering direct edges, at moderate noise?
    mod = syn[(syn.noise == 0.1)]
    inv_beats_raw = bool((mod["aupr"] > mod["aupr_raw_D"]).any())
    lines.append(f"\n- noiseless exact recovery: {'PASS' if syn_ok else 'FAIL'} (err {fmt(clean['recover_err'])})")
    lines.append(f"- at 10% noise, inverse beats raw |D| on edge recovery: {inv_beats_raw}\n")

    # ---- PART 1: real ----
    raw_path = next((CB_DIR / c for c in RAW_CANDIDATES if (CB_DIR / c).exists()), None)
    if raw_path is None:
        raise SystemExit(f"No raw RPE1 h5ad in {CB_DIR}.")
    print(f"Loading {raw_path.name} (chunked)...", flush=True)
    ds = load_replogle_raw_h5ad(raw_path, name="rpe1", min_cells=100, max_perturbations=max_perts)
    perturbed = list(ds.perturbed_genes)
    Dfull, Da, Db = perturbation_response_matrix(ds, split_half=True, seed=args.random_seed)
    # align to square perturbed x perturbed block
    P = [g for g in perturbed if g in Dfull.columns]
    D = Dfull.loc[P, P].to_numpy(float)
    D1 = Da.loc[P, P].to_numpy(float)
    D2 = Db.loc[P, P].to_numpy(float)
    lines.append("## Part 1-3: real RPE1 response operators\n")
    lines.append(f"- response block: {D.shape[0]} x {D.shape[1]}; control cells {ds.metadata['n_control_cells']}\n")

    # build operators (full + each split)
    operators = {"raw_|D|": (D, D1, D2)}
    for lam in ([0.1] if args.quick else [0.03, 0.1, 0.3, 1.0]):
        operators[f"ridge_inv_l{lam}"] = (
            deconvolve_response(D, method="ridge", lam=lam),
            deconvolve_response(D1, method="ridge", lam=lam),
            deconvolve_response(D2, method="ridge", lam=lam),
        )
    operators["pinv_inv"] = (
        deconvolve_response(D, method="pinv", rcond=1e-2),
        deconvolve_response(D1, method="pinv", rcond=1e-2),
        deconvolve_response(D2, method="pinv", rcond=1e-2),
    )
    if do_sparse:
        print("Fitting sparse (Lasso) deconvolution...", flush=True)
        operators["sparse_deconv"] = (
            sparse_deconvolution(D, 0.02),
            sparse_deconvolution(D1, 0.02),
            sparse_deconvolution(D2, 0.02),
        )

    # raw-D reconstruction baseline (use D1 directly to predict D2 - no operator)
    iu = ~np.eye(D.shape[0], dtype=bool)
    raw_recon_rank = float(spearmanr(D1[iu], D2[iu]).statistic)

    # evaluate
    stab_rows, recon_rows = [], []
    lines.append("| operator | partic.ratio | top10/src density | edge split-half rankcorr | dir. reproducibility | recon rankcorr (D_hat1 vs D2) | global-mode align |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, (Wf, W1, W2) in operators.items():
        pr = participation_ratio(Wf)
        dens = topk_density(Wf, 10)
        sstab = split_half_operator_stability(W1, W2)
        drep = direction_reproducibility(W1, W2)
        if name == "raw_|D|":
            recon = {"rank_corr": raw_recon_rank, "frob_rel": float("nan"), "row_cosine": float("nan")}
        else:
            recon = reconstruction_metrics(W1, D2)
        gma = global_mode_alignment(Wf, D)
        stab_rows.append({"operator": name, "participation_ratio": pr, "top10_density": dens,
                          "edge_rank_corr": sstab["edge_rank_corr"], "top2pct_overlap": sstab["top2pct_overlap"],
                          "direction_reproducibility": drep, "global_mode_align": gma})
        recon_rows.append({"operator": name, **recon})
        lines.append(f"| {name} | {fmt(pr,2)} | {fmt(dens)} | {fmt(sstab['edge_rank_corr'])} | "
                     f"{fmt(drep)} | {fmt(recon['rank_corr'])} | {fmt(gma)} |")

    pd.DataFrame(stab_rows).to_csv(TABLES_DIR / f"{PREFIX}_stability.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(TABLES_DIR / f"{PREFIX}_reconstruction.csv", index=False)
    # edges for the best-looking inverse operator (most split-half stable non-raw)
    nonraw = [r for r in stab_rows if r["operator"] != "raw_|D|"]
    best = max(nonraw, key=lambda r: r["edge_rank_corr"]) if nonraw else None
    if best is not None:
        Wbest = operators[best["operator"]][0]
        operator_edges(Wbest, P).sort_values("score", ascending=False).head(2000).to_csv(
            TABLES_DIR / f"{PREFIX}_edges.csv", index=False)

    # ---- verdict ----
    raw = next(r for r in stab_rows if r["operator"] == "raw_|D|")
    raw_dir = raw["direction_reproducibility"]
    # did ANY inverse operator beat raw on BOTH stability and reconstruction?
    better = [r for r in stab_rows if r["operator"] != "raw_|D|"
              and r["edge_rank_corr"] >= raw["edge_rank_corr"]
              and next(x for x in recon_rows if x["operator"] == r["operator"])["rank_corr"] >= raw_recon_rank]
    sparser = [r for r in stab_rows if r["operator"] != "raw_|D|" and r["participation_ratio"] < raw["participation_ratio"]]
    lines.append("\n## Verdict\n")
    lines.append(f"- synthetic noiseless recovery: {'PASS' if syn_ok else 'FAIL'}")
    lines.append(f"- raw |D| direction reproducibility (reference): {fmt(raw_dir)}")
    lines.append(f"- inverse operators that are SPARSER (lower participation ratio) than raw D: "
                 f"{[r['operator'] for r in sparser]}")
    lines.append(f"- inverse operators beating raw D on BOTH split-half stability AND reconstruction: "
                 f"{[r['operator'] for r in better] or 'NONE'}")
    if not syn_ok:
        verdict = "FAILED (synthetic control broke - do not trust real-data results)"
    elif better:
        verdict = "PROMISING (an inverse operator beat raw D on stability and reconstruction)"
    elif sparser:
        verdict = "MIXED (inverse can be sparser but does not beat raw D on stability+reconstruction)"
    else:
        verdict = "FAILED on real data (inverse neither sparser nor better than raw D)"
    lines.append(f"\n**VERDICT: {verdict}**\n")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(syn[syn.method == "ridge"]["noise"], syn[syn.method == "ridge"]["aupr"], "o-", label="inverse")
        ax[0].plot(syn[syn.method == "ridge"]["noise"], syn[syn.method == "ridge"]["aupr_raw_D"], "s--", label="raw D")
        ax[0].set_xlabel("noise"); ax[0].set_ylabel("edge-recovery AUPR"); ax[0].set_title("Synthetic recovery"); ax[0].legend()
        names = [r["operator"] for r in stab_rows]
        ax[1].bar(range(len(names)), [r["edge_rank_corr"] for r in stab_rows])
        ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax[1].set_title("Real: split-half edge stability")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"{PREFIX}.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        lines.append(f"(figure skipped: {e})")

    pd.DataFrame([{"verdict": verdict, "synthetic_pass": syn_ok, "raw_dir_repro": raw_dir,
                   "n_better": len(better), "n_sparser": len(sparser)}]).to_csv(
        TABLES_DIR / f"{PREFIX}_real_summary.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_debug_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
