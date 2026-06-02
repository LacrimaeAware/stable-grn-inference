# Documentation Update Map

Use this as a lightweight checklist for keeping project notes current. Do not update every file after every small edit; update the smallest set that preserves the project trail.

## Public Docs

| Location | Update When | Purpose |
|---|---|---|
| `README.md` | Project scope, milestone, or top-level layout changes | Public entry point for the repo |
| `docs/project_plan.md` | Phase status or next research direction changes | Public roadmap for Track A |
| `docs/data_inventory.md` | Raw data files, file formats, selected benchmark files, or loader assumptions change | Data-understanding record |
| `docs/experiment_summary.md` | A completed experiment changes the cross-experiment interpretation | Central results and conclusions reference |
| `experiments/*/*.md` | An experiment is added, rerun with important results, or limitations change | Local guide for each experiment; use descriptive names, not `README.md` |
| `docs/update_map.md` | Documentation conventions change | Keeps this checklist useful |

## Generated Outputs

| Location | Update When | Notes |
|---|---|---|
| `results/tables/` | Scripts produce metrics, edge audits, or debug reports | Ignored by git; useful locally |
| experiment debug reports | A script adds interpretability/audit output | Keep generated reports in `results/`, not public docs |

## Private Notes

| Location | Update When | Purpose |
|---|---|---|
| `private_docs/dev_journal/YYYY-MM-DD.md` | A session reaches a checkpoint, changes direction, or clarifies next steps | Concise private work log |
| `private_docs/` planning docs | Only when explicitly working on private planning | Never copy private prose into public docs |

## Suggested Update Pattern

- New data or file-format insight: update `docs/data_inventory.md`; optionally update the private journal.
- New experiment script: add or update that experiment's descriptive Markdown note; update `docs/project_plan.md` if it changes phase status.
- New result that changes interpretation: update the relevant experiment note, `docs/experiment_summary.md`, and private journal; update top-level README only if it affects the current milestone.
- New generated metric table only: usually no public doc update unless the result changes the story.
- New next step: update `docs/project_plan.md` if public, and the private journal if it is session-specific.

## Current Important Locations

```text
README.md
docs/project_plan.md
docs/data_inventory.md
docs/experiment_summary.md
docs/update_map.md
experiments/01_dream4_size10_baseline/baseline_comparison.md
experiments/02_dream4_size10_stability/stability_audit.md
experiments/03_dream4_size10_data_regimes/data_regime_audit.md
experiments/04_dream4_genie3_baseline/genie3_baseline.md
experiments/06_dream4_size10_topology_evaluation/topology_evaluation.md
experiments/07_dream4_size10_lagged_timeseries/lagged_timeseries_audit.md
experiments/08_dream4_size10_dynamic_model_batch/dynamic_model_batch_audit.md
experiments/09_dream4_size10_dynamic_sparse_validation/dynamic_sparse_validation.md
experiments/10_dream4_size100_dynamic_sparse_scaling/size100_dynamic_sparse_scaling.md
experiments/11_dream4_dynamic_baseline_and_calibration/dynamic_baseline_and_calibration.md
experiments/12_gnw_sweep_design/gnw_sweep_design.md
experiments/13_dream4_mechanism_audit/mechanism_audit.md
experiments/14_dream4_calibrated_confidence/calibrated_confidence.md
experiments/15_modern_grn_benchmark_adapter/modern_grn_benchmark_adapter.md
experiments/16_beeline_adapter_smoke/beeline_adapter_smoke.md
experiments/17_dream4_stability_orientation_diagnostics/stability_orientation_diagnostics.md
experiments/18_beeline_diagnostics/beeline_diagnostics.md
experiments/19_interventional_grn_benchmark_scouting/interventional_benchmark_scouting.md
results/figures/                   # ignored generated figures (matplotlib)
private_docs/dev_journal/2026-06-01.md
private_docs/dev_journal/2026-06-02.md
results/tables/                    # ignored generated outputs
```
