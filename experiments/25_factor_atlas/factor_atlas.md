# Experiment 25 — Counterfactual factor atlas: your sub-feature idea, validated then applied

## The idea (yours), stated cleanly

Don't describe an example only by its class ("a 9"). Describe it as a **class plus reusable
sub-features that cut across classes** (red, thick, slanted). A sub-feature is **core** to a
class only if:
- **removing** it from the class's examples **breaks** the class (necessity), and
- **adding** it to a *rival* class **converts** the rival (sufficiency).

If a feature can be moved across classes without changing identity, it's a transferable
**nuisance/style** factor. A **shortcut** is a nuisance factor that's spuriously glued to a
class in the training data ("all 9s are red"); the counterfactual test should *see through* it.
Separating core from nuisance is an **anti-overfitting** tool.

This experiment does two things in order, and the order is the point: **prove the test is
faithful on ground truth first, then apply the validated test to genes.**

## Part A — synthetic positive control (ground truth known)

I planted three kinds of sub-feature into two-class data: a real **core** factor, an innocent
**nuisance** factor, and a **shortcut** (separate direction, but on for 90% of class-1 and 10%
of class-0 in training — spuriously correlated). Then I ran your test.

| planted factor | necessity R (high = not needed) | sufficiency A (high = converts) | core_score |
| --- | --- | --- | --- |
| **core** | **0.04** | **0.96** | **0.92** |
| shortcut | 0.98 | 0.03 | 0.0006 |
| nuisance | ~1.00 | ~0.00 | ~0.000 |

- Sub-features were **discovered from unlabeled deltas at ARI = 1.0** (recovered perfectly).
- The test **marks core as core** (removing breaks it, adding converts) and **sees through the
  shortcut** (core_score ≈ 0 despite the training correlation).
- **The anti-overfitting payoff:** trained where the shortcut is glued to class 1, then tested
  on the *flipped* combinations (class-1-without-shortcut, class-0-with-shortcut), the **raw
  classifier overfits → 0.89**, while **projecting out the discovered nuisance directions →
  1.00**. The factored representation generalizes to unseen class×factor combinations.

**Part A verdict: PASS.** Your idea is real, and the implementation is faithful — proven where
truth is known. This is the positive control that makes any real-data verdict trustworthy
(if it had failed here, the bug would be mine, not the data's).

## Part B — apply the validated tool to real RPE1 genes

Examples = perturbation responses Δ_g (651). The dominant shared program (53% of variance) is
the cell-cycle/proliferation axis (H2AFZ, TUBA1B, RRM2, NASP…). Two questions, mapping your
"true 9" idea onto genes:

1. **Is the cell-cycle a nuisance factor that cuts across response-modules (like redness)?**
   The module counterfactual gave R=0.17 — but I'm flagging this as **artifact-prone for genes**:
   removing a 53%-of-variance axis mechanically moves every point off the old cluster centroids,
   so the reassignment isn't a clean nuisance signal. *Don't over-read it.*
2. **Does removing it reveal a cleaner "true function" core?** A single seed suggested yes
   (module ARI 0.59 → 0.65). **Verified across 15 (k, seed) settings, that reverses:** mean gain
   **−0.12**, residual wins only **33%** of the time. So removing the shared program does **not**
   reliably reveal a more reproducible core — on average it's *worse*. (This matches exp22, where
   removing the program hurt per-perturbation stability.)

> The single-seed "positive" was a seed artifact; the multi-seed check killed it. I'm leaving
> this in the write-up on purpose — it's the exact failure mode you were worried about, and the
> verification is what makes the negative trustworthy.

**Part B verdict: the decomposition does NOT cleanly transfer to genes.** A dominant shared
program exists, but — unlike redness-and-a-9 — it is **entangled with real gene function**, so
factoring it out does not leave a cleaner core. The "true 9" move works when the nuisance is
genuinely orthogonal to identity; in genes the biggest shared axis is *part of* the biology.

## Bottom line (a precise map of where your idea applies)

- **Your sub-feature / counterfactual idea is genuinely real and powerful** — Part A is a clean,
  reproducible positive control, not hype: discover factors without labels, separate core from
  nuisance, see through shortcuts, and beat overfitting on unseen combinations.
- **It needs factors that are separable from identity.** It shines exactly where that holds
  (the synthetic world here; presumably digits/Track B). **Genes are a case where it partially
  breaks**, because the dominant shared factor is woven into the thing you'd call the "core."
- That's not a dead end for the idea — it's a *map*. The natural next homes are domains where the
  reusable factors really are separable from class identity (Track B's digits; or a gene setting
  with a *known, separable* covariate like cell-line/batch rather than the entangled cell cycle).

## Engineering
New tested library: `src/stable_grn_inference/analysis/factor_atlas.py`
(`make_factor_atlas_data`, `discover_factor_directions`, `counterfactual_necessity_sufficiency`,
`held_out_combination_accuracy`, `project_out_directions`) with correctness tests that assert the
core/nuisance separation, shortcut see-through, and anti-overfitting generalization on planted
ground truth (4 tests). `data/` and `results/` stay git-ignored; gene tests never touch the real file.
