# Regimes and limits of directed gene-network recovery

Repository: `stable-grn-inference`. A controlled, regime-by-regime study of when a directed
gene-regulatory edge (gene A regulates gene B) can be recovered from expression data, and why it
usually cannot. Forty experiments span simulated networks (DREAM4), static single-cell data
(BEELINE), interventional CRISPR perturbation data (Replogle RPE1, RENGE), and time-resolved data
(BoolODE, RENGE, DREAM4 time-series). Every claim is graded against fair baselines and audited for
confounds. The contribution is not a new inference method; it is a map of where recovery is and is
not possible, with a controlled explanation for the failures.

## Headline conclusions

1. Direction requires time, intervention, or higher-order structure. A symmetric second-order
   statistic (correlation) cannot orient an edge (0.50 by construction). Lagged time-series orient
   well (0.88 to 0.96 on DREAM4); static single-cell data does not (mean 0.60, often 0.50);
   interventions make 61% of RPE1 pairs direction-decidable.

2. Simple established methods are not beaten, on fair benchmarks. Correlation, GENIE3, lagged LASSO,
   and self-persistence were not outperformed by any advanced method tried (a dynamical / DMD
   operator, spectral seriation and diffusion ordering, non-Gaussian LiNGAM orientation, higher-order
   correlation). On DREAM4 the dynamical operator ranks last of the orientable methods (directed AUPR
   0.37 versus lagged GENIE3 0.54); on BoolODE it is mid-pack. This matches the published field, where
   simple baselines match deep models on this task.

3. Real perturbation response is dominated by one convergent program. In RPE1, knockouts of essential
   genes trigger a shared cell-cycle program accounting for about 53% of response variance (about 4%
   of unperturbed control variance), which drowns gene-specific edges and is not linearly separable
   from them.

4. A separability phase diagram explains the failures (experiment 28, the conceptual capstone).
   Recovering specific structure from under a dominant shared mode has two axes: the dominant-mode
   fraction (rho), which is fixable by deflation, and a specific signal-to-noise floor, which is not.
   On a full synthetic grid (100 genes, 5 seeds), deflation is rho-invariant (recovery 0.67 across all
   rho), but no method clears a signal-to-noise floor near 0.2. RPE1 sits in the unrecoverable corner
   (rho about 0.53, low specific-SNR; best recovery 0.33). This reframes every RPE1 negative as one
   point on a map: the bottleneck is the SNR floor, not the dominant mode itself.

5. What is recoverable from real perturbation data is aggregate structure, not edges (experiment 26).
   Per-gene axes, response magnitude / breadth (essentiality) and a net-effect upstream / downstream
   ordering, are split-half reproducible (0.97 and 0.99) and recover known machinery (ribosome,
   spliceosome, proteasome, nuclear pore). This is a strong internal diagnostic, not proof of direct
   regulation.

6. Internal reproducibility is not biology. A "reproducible, structured, cell-state-aligned"
   single-cell response heterogeneity (experiment 37) turned out to be sequencing depth: its axis
   correlates with library size at 0.85 and is one global mode shared across all knockouts
   (experiment 38). Program discovery that survives reproducibility collapses to housekeeping under
   depth and external-coherence controls (experiment 39). Reproducibility reproduces technical
   artifacts too; a claim of structure must control for depth, show specificity, and validate against
   external biology.

## Capstones and their exact verification status

- Experiment 28, separability phase diagram. VERIFIED on the full grid (100 genes, 5 seeds, this
  checkout): the two-axis result and the SNR floor near 0.2 hold; deflation is rho-invariant. The
  system is synthetic with known ground truth. The placement of RPE1 on the map (measured rho about
  0.53, low SNR) is an interpretation, not a theorem.
- Experiment 26, essentiality and cascade axis. The reproducibility figures (0.97, 0.99) come from a
  prior run on the RPE1 `h5ad`, which is not present in this checkout (git-ignored, about 8.7 GB) and
  cannot be re-run here. The reproducibility is internal (split-half). External validation against
  DepMap essentiality, CORUM, or STRING was not performed. Status: a strong internal diagnostic, not
  externally validated, not reproducible from this checkout alone.
- Experiment 33, fair benchmark. VERIFIED in-repo: the dynamical operator loses to lagged GENIE3 and
  lagged LASSO on the same pairs and the same ground truth.
- Experiments 38 and 39, confound audits. VERIFIED in-repo on RENGE: the heterogeneity signal is
  library size; no externally-coherent, non-housekeeping, knockout-specific program survives.

## What did not work (and is not re-attempted)

- Removing or inverting the dominant program (experiments 22, 23): subtraction reduces stability;
  linear deconvolution recovers synthetic truth but gains nothing on RPE1.
- Transfer across perturbations (experiment 24): shared low-rank structure does not predict a held-out
  perturbation better than its own noisy half.
- Cascade-adjacency as an edge cue (experiment 27): ordering distance is uncorrelated with mediation.
- Dynamical / DMD operator (experiments 30 to 33): orients where a symmetric statistic cannot, but
  loses to lagged GENIE3 on fair benchmarks.
- Order from static geometry (experiment 34): the 1D order is recoverable (Spearman 0.83) but ties a
  plain PCA baseline and does not improve network recovery.
- Non-Gaussian orientation (experiment 35): provably correct on a planted chain, but on BoolODE it
  does not beat symmetric correlation (0.29 versus 0.36), unchanged at 5000 cells.
- Higher-order / iterated correlation (experiments 34, 36): adds spurious transitive edges.
- Program and heterogeneity discovery (experiments 37 to 39): collapses to depth and housekeeping
  confounds under audit.

## Lessons (methodological, transferable)

1. Beating chance, or beating a symmetric baseline that cannot do the task, is not a result. The bar
   is the established method (GENIE3, correlation, lagged regression) on the same data and truth.
2. Internal reproducibility reproduces technical artifacts (library size). External validation is
   required before a "structure" claim.
3. Identifiability is set by the regime: edge direction is recoverable only with time, intervention,
   or non-Gaussian structure, not from a static second-order statistic.
4. A clean negative with a controlled explanation (the phase diagram) is the durable contribution
   here, not a leaderboard number.

## Status and next direction

The directed-edge-recovery question is treated as concluded for this project: a mapped wall with a
diagnostic explanation (experiment 28) and a reproducible aggregate signal (experiment 26). The
forward direction, pursued separately, is parameter identifiability and inference for small
mechanistic adaptation models (`docs/roadmap.md`), which fits a statistics background and is not a
saturated benchmark. Its first tooling step (experiment 40) is built and its structural-identifiability
core is verified on a textbook gene-expression model. Combinatorial / non-additivity analysis on
combination-perturbation data is documented as a separate later option, not pursued simultaneously.

## Experiment log

| #     | experiment | method | result |
|-------|---|---|---|
| 01-04 | DREAM4 baselines | correlation, LASSO, Elastic Net, random forest | correlation AUPR 0.33 highest; LASSO 0.29, RF 0.30 |
| 07    | lagged time-series | source(t) to target(t+1) | AUPR 0.30 to 0.53 |
| 08-09 | dynamic sparse | LASSO level/delta, self in/out | best Size10 AUPR 0.65 / AUROC 0.82 |
| 10    | Size100 scaling | same model, 100 genes | did not scale; alpha 0.1 best; fusion AUPR 0.21 |
| 11-14 | calibration, fusion, mechanism | alpha sweep, CV/BIC, self-permutation | alpha tracks density; CV/BIC retain 96-100% of oracle |
| 15-16 | BEELINE adapter | static methods on single-cell | transfer confirmed; references are proxies |
| 17-18 | orientation diagnostics | skeleton vs orientation, sqrt-LASSO, stability | orientation 0.88-0.96 (DREAM4) vs 0.50-1.00 (BEELINE) |
| 19    | interventional scouting | benchmark selection | CausalBench / RPE1 selected |
| 20    | RPE1 diagnostics | Wasserstein effect, direction asymmetry | 61% decidable; observational AUROC 0.57 |
| 21    | response geometry | SVD, split-half stability | top component 53%; direction reproducible 0.70 |
| 22    | covariate cleaning | program / covariate residualization | removing cell-cycle reduces stability |
| 23    | inverse deconvolution | W = I - (I+D)^-1 | exact on synthetic; no gain on RPE1 |
| 24    | held-out perturbation | low-rank prediction | no transfer beyond self-estimate |
| 25    | counterfactual feature test | necessity / sufficiency | recovers synthetic truth; no transfer to RPE1 |
| 26    | essentiality and cascade position | response magnitude/breadth/centrality; net_out | reproducible 0.97 / 0.99, recovers known machinery (internal-only, prior run, not externally validated) |
| 27    | cascade-adjacent edges | ordering distance vs mediation | not supported (-0.06); local restriction lowers reproducibility (0.43 vs 0.81); correlation most reproducible (0.91) |
| 28    | separability phase diagram | synthetic dominant-mode + specific structure; sweep rho/SNR | VERIFIED full grid: rho fixable by deflation, specific-SNR a hard floor (~0.2); RPE1 in the unrecoverable corner (best 0.33) |
| 29    | whitened asymmetry | residual-asymmetry reproducibility, whitening sweep | whitening does not help (best alpha 0); synthetic control only |
| 30-32 | dynamical recovery | DMD operator vs static; DREAM4, BoolODE, RENGE timecourse | operator orients where static cannot; RENGE response builds over days, ordering reproducible 0.75 |
| 33    | dynamical baseline benchmark | DMD vs lagged GENIE3/LASSO/correlation, same pairs/truth | no win: last on DREAM4 (0.37 vs 0.54), 2nd on BoolODE (0.41 vs 0.45) |
| 34    | order from static | spectral / diffusion order; does order help; higher-order correlation | order recovered (0.83) ties PC1 (0.82); no network gain (0.35 vs 0.36); higher-order adds spurious edges |
| 35    | non-Gaussian orientation | LiNGAM direction-from-static; detectability map | orients a planted chain but fails on BoolODE (0.29 vs 0.36), unchanged at 5000 cells |
| 36    | queued directions | diversity-consensus; cycle 2D geometry | consensus does not beat best single lens (0.64 vs 0.67); 2D recovers the actual cycle (0.80 vs 0.55) |
| 37    | programs and heterogeneity | NMF vs PCA reproducibility; single-cell heterogeneity | heterogeneity "result" was a technical confound (corrected by exp 38) |
| 38    | heterogeneity audit | confound / specificity / residual tests | heterogeneity is library size (corr 0.85), one global axis (0.81); no knockout-specific residual (0.37) |
| 39    | validated programs | depth + housekeeping controls; external STRING | only the housekeeping cluster (GAPDH/ACTB) survives the filter; no specific biological program. Negative |
| 40    | identifiability pipeline (forward) | profile likelihood, Fisher information, MLE | validated on mRNA->protein: protein-only cannot separate transcription/translation rate (rank 3/4); both-observed can (4/4) |

## Reproduce

```bash
# Python 3.13, dependencies in requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests            # 207 core tests (exp-40 identifiability tests are additional and compute-heavy)
.\.venv\Scripts\python.exe -B experiments/<NN_name>/run_*.py --quick   # any experiment
.\.venv\Scripts\python.exe -B docs/figures/make_figures.py             # regenerate figures
```

Datasets (`data/`) and generated tables (`results/`) are git-ignored; the test suite uses synthetic
fixtures and does not depend on them. The RPE1 `h5ad` (experiments 20 to 26, 29, 37 to 39) is not in
this checkout; those experiments require the file from CausalBench / figshare.

## Layout

```text
stable-grn-inference/
├── src/stable_grn_inference/   # library: data adapters, inference, evaluation, analysis, dynamics
├── experiments/                # 40 experiments, each with a write-up, script, and tests
├── docs/                       # reports, figures, roadmap, and the literature review
└── tests/                      # synthetic fixtures only
```

## Further reading

- [`docs/experiment_summary.md`](docs/experiment_summary.md): per-experiment results.
- [`docs/research_directions.md`](docs/research_directions.md): the directions tried, the reconciliation
  with the literature, and the standing conclusions.
- [`docs/literature_review.md`](docs/literature_review.md): the external research base, with citations
  and verification status.
- [`docs/roadmap.md`](docs/roadmap.md): the forward direction (adaptation-model identifiability) and the
  comparison of options.
- Each `experiments/NN_*/` directory contains the full numbers and the per-experiment write-up.
