# Experiment 25: counterfactual factor atlas

## The method

Describe an example as a class plus reusable sub-features that cut across classes (for example a digit with attributes like color, thickness, slant). A sub-feature is core to a class only if:
- removing it from the class's examples breaks the class (necessity), and
- adding it to a rival class converts the rival (sufficiency).

A feature that can be moved across classes without changing identity is a transferable nuisance/style factor. A shortcut is a nuisance factor that is spuriously correlated with a class in the training data (for example, all examples of one class share a color); the counterfactual test should separate it from the core. Separating core from nuisance is an anti-overfitting tool.

The experiment runs in two steps: validate the test on synthetic data with known ground truth, then apply the validated test to genes.

## Part A: synthetic positive control (ground truth known)

Three sub-features were planted into two-class data: a core factor, a nuisance factor, and a shortcut (a separate direction, on for 90% of class-1 and 10% of class-0 in training, so spuriously correlated).

| planted factor | necessity R (high = not needed) | sufficiency A (high = converts) | core_score |
| --- | --- | --- | --- |
| core | 0.04 | 0.96 | 0.92 |
| shortcut | 0.98 | 0.03 | 0.0006 |
| nuisance | ~1.00 | ~0.00 | ~0.000 |

- Sub-features were discovered from unlabeled deltas at ARI 1.0.
- The test marks the core factor as core (removing breaks it, adding converts) and assigns the shortcut a near-zero core_score despite its training correlation.
- Anti-overfitting check: a classifier trained where the shortcut is glued to class 1, then tested on the flipped combinations (class-1 without the shortcut, class-0 with it), reaches 0.89. Projecting out the discovered nuisance directions first raises this to 1.00.

Part A result: on planted ground truth the test recovers the factors, separates core from nuisance, and the factored representation generalizes to unseen class-by-factor combinations. This is the positive control for the real-data application.

## Part B: application to RPE1 genes

Examples are the perturbation responses (651 genes). The dominant shared program (53% of variance) is the cell-cycle / proliferation axis (H2AFZ, TUBA1B, RRM2, NASP).

1. Is the cell-cycle program nuisance-like with respect to response modules? The module counterfactual gave R=0.17. This test is artifact-prone for genes: removing a 53%-of-variance axis mechanically moves points off the original cluster centroids, so the reassignment is not a clean nuisance signal.
2. Does removing it reveal a cleaner core? A single seed suggested yes (module ARI 0.59 to 0.65). Across 15 (k, seed) settings this reverses: mean gain -0.12, residual wins 33% of the time. Removing the shared program does not reliably reveal a more reproducible core. This matches experiment 22, where removing the program reduced per-perturbation stability.

The single-seed positive was a seed artifact; the multi-seed check removes it.

Part B result: the decomposition does not transfer to genes. The dominant shared program is entangled with real gene function, so factoring it out does not leave a cleaner core. The decomposition works when the nuisance is orthogonal to identity; the cell-cycle axis is part of the biology.

## Summary

- The counterfactual test works as designed on synthetic data: discover factors without labels, separate core from nuisance, identify shortcuts, and reduce overfitting on unseen combinations.
- It requires factors that are separable from identity. It applies where that holds (the synthetic data here) and does not transfer to RPE1, where the dominant shared factor is part of the core biology.
- A natural application is a setting with a known, separable covariate (for example cell-line or batch) rather than the entangled cell cycle.

## Implementation

`src/stable_grn_inference/analysis/factor_atlas.py` (`make_factor_atlas_data`, `discover_factor_directions`, `counterfactual_necessity_sufficiency`, `held_out_combination_accuracy`, `project_out_directions`) with 4 tests asserting core/nuisance separation, shortcut identification, and anti-overfitting generalization on planted ground truth. `data/` and `results/` are git-ignored; tests use synthetic fixtures.
