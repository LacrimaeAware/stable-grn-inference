# Experiment 27: cascade-adjacent edges (RPE1)

## Hypothesis

Experiment 26 produced a reproducible cascade ordering (net_out, a per-gene upstream/downstream position). The hypothesis tested here: a direct edge A->B connects genes that are adjacent in that ordering, while the cascade connects distant genes (far-upstream to far-downstream). If true, ordering-adjacent pairs should be less explainable as a chain through an intermediate gene, and more reproducible as edges.

## Method

For each interacting pair (|D[A,B]| above a control-null), two quantities:
- ordering distance: |rank(A) - rank(B)| in the net_out ordering.
- mediation ratio: (strongest 2-step path through any middle gene C) / |D[A,B]|. A ratio above 1 means a chain explains the pair as well as the direct effect (indirect); below 1 means the direct effect exceeds any chain (direct).

Reproducibility: the ordering and an ordering-local, upstream-to-downstream edge score were computed on each cell half independently; the top-200 edge overlap across halves is compared to a raw |D| ranking and an observational correlation ranking.

## Result (651 genes)

Part 1, ordering distance vs mediation:
- Spearman(ordering distance, mediation ratio) = -0.060. Ordering distance does not predict how chain-explained a pair is.
- Mean mediation ratio: nearest-quartile pairs 3.33 versus farthest-quartile 3.28. Both are high and nearly equal: essentially every pair is well explained by a 2-step path through some gene, independent of ordering distance.
- Fraction of pairs where the direct effect exceeds the best chain (ratio < 1): 0.023 (near) versus 0.052 (far). Both small, and the direction is opposite to the hypothesis.

Part 2, split-half top-edge reproducibility (top-200 overlap):

| edge score | overlap across halves |
| --- | --- |
| observational correlation | 0.910 |
| raw |D| (total effect) | 0.805 |
| cascade-local |D| (ordering-adjacent) | 0.425 |

Restricting to cascade-local pairs reduces reproducibility rather than increasing it. The raw total-effect and the observational correlation rankings are both highly reproducible, but those rankings are dominated by the cascade.

## Summary

The hypothesis is not supported on RPE1. Ordering distance is uncorrelated with mediation, and the cascade-local restriction lowers edge reproducibility. The reason is the same convergent cascade: a strong 2-step path through the shared program exists for almost every pair (mediation ratio near 3.3 throughout), so a pair's ordering position does not change whether its effect is direct or chain-mediated. Separating direct from cascade-mediated effects on this data is not achieved by the ordering-adjacency restriction.

The observational correlation ranking being the most reproducible (0.91) is consistent with correlation remaining a strong baseline throughout the project. No external ground truth is used; reproducibility and mediation are internal checks.

Artifacts (git-ignored `results/`): `causalbench_local_edges_summary.csv`, `causalbench_local_edges_debug_report.md`.
