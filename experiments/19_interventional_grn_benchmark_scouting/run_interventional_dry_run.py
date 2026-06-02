r"""Experiment 19 dry-run: exercise the interventional adapter shape with ZERO download.

Scouting deliverable (see interventional_benchmark_scouting.md). This does NOT download
CausalBench / Replogle data. It:

1. Reports whether a local CausalBench install / data dir exists (blocker check).
2. Builds the synthetic linear-SEM interventional fixture (known DAG, control + single
   knockdowns) matching the expected on-disk schema.
3. Loads it through :func:`load_interventional_frames`.
4. Computes the CausalBench-style interventional effect matrix (Wasserstein).
5. Runs the REBUILT orientation diagnostic (intervention asymmetry) and contrasts it
   with the observational static-score orientation control from experiments 17-18.

Run:
    $env:PYTHONPATH = "src"
    .\.venv\Scripts\python.exe -B experiments/19_interventional_grn_benchmark_scouting/run_interventional_dry_run.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    detect_causalbench,
    interventional_effect_matrix,
    interventional_orientation_asymmetry,
    load_interventional_frames,
    make_synthetic_interventional,
)

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"
PREFIX = "interventional_dry_run"


def observational_orientation_control(dataset, expr_corr: pd.DataFrame) -> dict:
    """The experiment-17/18 observational orientation instrument, for contrast.

    Symmetric association (|correlation|) cannot orient an edge, so on a symmetric score
    orientation-given-skeleton is exactly 0.5 by construction. We report it to make the
    point that the *observational* diagnostic is uninformative here and must be replaced
    by the interventional-asymmetry test."""
    truth = {
        (s, t)
        for s, t in zip(dataset.reference_edges["source"], dataset.reference_edges["target"])
    }
    seen, correct, n = set(), 0, 0
    for s, t in truth:
        key = frozenset((s, t))
        if key in seen or (t, s) in truth:  # skip reciprocal-true
            continue
        seen.add(key)
        fwd = abs(expr_corr.loc[s, t])
        rev = abs(expr_corr.loc[t, s])  # identical for correlation -> tie -> 0.5
        n += 1
        correct += 0.5 if np.isclose(fwd, rev) else float(fwd > rev)
    return {"accuracy": correct / n if n else float("nan"), "n_pairs": n}


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Experiment 19 dry-run: interventional adapter\n"]

    # 1. blocker check (no download)
    cb = detect_causalbench(
        search_dirs=[ROOT / "data" / "raw" / "causalbench", ROOT / "data" / "raw" / "CausalBench"]
    )
    lines.append("## Blocker check (no download performed)\n")
    lines.append(f"- causalscbench package installed: **{cb['package_installed']}**")
    lines.append(f"- local CausalBench data dirs found: {cb['data_dirs_found'] or 'none'}")
    lines.append(
        "- Action if absent: `pip install causalscbench`, then follow the dataset "
        "download in the scouting doc. Raw Replogle-scale data is NOT auto-downloaded.\n"
    )

    # 2-3. synthetic fixture -> dataset
    expr, perturb, true_edges = make_synthetic_interventional(
        n_genes=7, n_cells_per_condition=120, edge_density=0.4, seed=1
    )
    ds = load_interventional_frames(
        "synthetic-interventional",
        expr,
        perturb,
        reference_edges=true_edges,
        reference_kind="exact",
    )
    lines.append("## Synthetic fixture (matches expected schema)\n")
    lines.append(
        f"- genes={ds.metadata['n_genes']}, cells={ds.metadata['n_cells']}, "
        f"perturbations={ds.metadata['n_perturbations']}, "
        f"control_cells={ds.metadata['n_control_cells']}"
    )
    lines.append(
        f"- candidate_edges={ds.metadata['n_candidate_edges']} "
        f"(sources restricted to perturbed genes), true_edges={ds.metadata['n_true_edges']}\n"
    )

    # 4. interventional effect matrix (CausalBench-style)
    eff = interventional_effect_matrix(ds)
    eff_labeled = eff.merge(ds.edge_labels, on=["source", "target"], how="left")
    mean_true = eff_labeled.loc[eff_labeled["is_true"] == 1, "effect"].mean()
    mean_false = eff_labeled.loc[eff_labeled["is_true"] == 0, "effect"].mean()
    lines.append("## Interventional effect (Wasserstein perturb vs control)\n")
    lines.append(f"- mean effect on TRUE edges:  {mean_true:.3f}")
    lines.append(f"- mean effect on FALSE edges: {mean_false:.3f}")
    lines.append(
        f"- separation (true - false): **{mean_true - mean_false:.3f}** "
        "(positive => interventional signal ranks true edges above false)\n"
    )

    # 5. rebuilt orientation diagnostic vs observational control
    inter = interventional_orientation_asymmetry(ds, eff)
    corr = ds.expression.corr()
    obs = observational_orientation_control(ds, corr)
    lines.append("## Orientation: rebuilt (interventional) vs old (observational)\n")
    lines.append(
        "| diagnostic | orientation accuracy | n pairs | note |\n"
        "| --- | --- | --- | --- |"
    )
    lines.append(
        f"| interventional asymmetry (NEW) | {inter['accuracy']:.3f} | "
        f"{inter['n_pairs_both_perturbed']} | direction from effect(A->B) vs effect(B->A) |"
    )
    lines.append(
        f"| observational |correlation| (OLD) | {obs['accuracy']:.3f} | {obs['n_pairs']} | "
        "symmetric score cannot orient => 0.5 by construction |"
    )
    lines.append(
        f"\n- reciprocal-true pairs excluded: {inter['n_reciprocal_excluded']}"
    )
    lines.append(
        "- CAUTION: intervention asymmetry is directional evidence, not proof "
        "(indirect effects, compensation, off-target knockdown, cell-state shifts).\n"
    )

    # persist tables
    eff_labeled.to_csv(TABLES_DIR / f"{PREFIX}_effects.csv", index=False)
    inter["pairs"].to_csv(TABLES_DIR / f"{PREFIX}_orientation_pairs.csv", index=False)
    report = TABLES_DIR / f"{PREFIX}_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
