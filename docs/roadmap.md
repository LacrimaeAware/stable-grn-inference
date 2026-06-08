# Roadmap: a math/stats-fit direction in cellular adaptation

A plan toward a modest, real, publishable-type contribution that fits a statistics / mathematics /
quantitative background, aligned with a potential collaborator (Necmettin Yildirim) and away from the
saturated GRN edge-recovery benchmark this repo already mapped as a wall. Built from a verified
literature pass (docs/literature_review.md section 11).

## The collaborator and his world

Necmettin Yildirim, Professor of Mathematics (Soo Bong Chae Chair of Applied Math, New College of
Florida; formerly McGill, UNC). Method: mechanistic dynamical-systems modeling, writing and analyzing
ODE / delay-differential-equation (DDE) models and studying their dynamics (bifurcation, bistability).
No machine learning and no GRN edge recovery in his corpus.

- Yildirim & Mackey (2003), Biophysical J 84:2841-2851: 5-equation DDE model of the lac operon;
  bistability via a cusp bifurcation.
- Dyjack, Azeredo-Tseng & Yildirim (2017), Molecular BioSystems 13(7):1323-1335: cellular adaptation,
  yeast MAPK/pheromone response adapting to a persistent signal via two negative-feedback phosphatases
  (Ptp3 constant, Msg5 upregulated by pFus3).
- Violin et al. (2008), JBC 283(5):2949-2961: beta2-adrenergic receptor desensitization (adaptation);
  his most-cited (modeling contribution).

This is forward mechanistic modeling, the opposite end from data-driven network inference. The
benchmark wall (simple baselines beat everything) does not apply, because it is not a prediction
contest.

## The field (cellular adaptation), and the gap

Adaptation has a rigorous control-theory backbone:
- Sontag (2003/2004, Systems & Control Letters): the internal model principle, a system adapting to a
  class of signals must contain a subsystem generating those signals.
- Robust perfect adaptation (RPA) is integral feedback; antithetic integral feedback (Briat, Gupta &
  Khammash 2016, Cell Systems) is a single biomolecular topology realizing it; Aoki et al. (2019,
  Nature 570:533-537) prove it achieves RPA in arbitrary noisy networks.
- Araujo & Liotta (2018, Nat Commun 9:1757): complete topological classification of RPA networks into
  opposer / balancer modules; opposer nodes integrate a tracking error (a special case of the IMP).
- The stochastic turn: Gupta & Khammash (2022, PNAS 119:e2207802119) show RPA constraints differ
  between deterministic and stochastic descriptions; noise changes the rules.

The open gap (a verified open question from the research): nobody has done a rigorous parameter
identifiability and inference analysis of Yildirim's own adaptation / lac-operon models. He built and
analyzed the dynamics; the statistician's questions (are these parameters estimable from data, how
well, from what measurements) are unanswered. That gap is exactly the statistics/quant toolkit.

## The project (the candidate mild win)

A rigorous identifiability + inference re-analysis of one Yildirim adaptation model (the 2003
lac-operon DDE or the 2017 MAPK adaptation model):
1. Reimplement the published model (small ODE/DDE system).
2. Structural and practical identifiability: profile likelihood / Fisher information; which parameters
   are estimable from realistic time-course data and which are not.
3. Inference under noise: Bayesian/MCMC or particle filtering (the finance/filtering bridge) to fit
   the model to noisy single-cell or time-course data, with honest uncertainty quantification.
4. Optimal experimental design: what measurements would render the unidentifiable parameters
   identifiable.

Why it fits: it is individual-scale, it uses the likelihood/identifiability/filtering/stochastic
toolkit (the quant/stats edge), it directly engages the collaborator's models (a real reason to work
with him), it is an established publishable contribution type in mathematical biology, and it avoids
the saturated benchmark. The inference / detectability / stochastic-operator code from this repo
(experiments 23, 28, 30, 35, 38) transfers to steps 2-3.

Honest calibration: a modest contribution type, not a high-impact paper, and the result still depends
on execution. But it is well-matched, genuinely open, and tractable, which is more than the
edge-recovery and structure-discovery directions offered.

## Directions considered (not bulldozed)

All directions on the board, with an honest comparison so the chosen one does not erase the others.

- A. Identifiability / inference of adaptation models (this roadmap). RECOMMENDED first focus. Math/stats
  fit, collaborator-aligned (Yildirim), individual-scale, low confound risk; the bar is "correct,
  useful analysis," not beating a baseline.
- B. Non-additivity / epistasis on combination-perturbation data (Norman et al. 2019). Biologically the
  more interesting question, but in the crowded ML-benchmark space where simple additive baselines are
  hard to beat (same wall risk as edge recovery), not collaborator-aligned, and needs new data. A
  legitimate, separate project; keep it live as the backup, do not pursue simultaneously (one-person
  focus). The statistical toolkit built for A (identifiability, inference, signal-vs-noise) partly
  equips B later.
- Mapped walls (done, negative, documented in research_directions.md and experiment_summary.md): edge
  recovery (simple correlation/GENIE3 win); structure/program/heterogeneity discovery (collapses to
  depth/housekeeping/cell-cycle confounds, exp 37-39); dynamical-operator recovery (loses to lagged
  GENIE3, exp 30-33). These are settled negatives, not options.

Can A and B run at once: they are separate projects (different data, methods, collaborator), so no,
not without diluting a one-person effort. Focus A; keep B documented.

## Build status

- Step 1 (the identifiability pipeline) is built and validated on a textbook gene-expression model
  (mRNA -> protein), where transcription and translation rates are provably non-identifiable from
  protein data alone but identifiable when mRNA is also observed: experiment 40,
  `src/stable_grn_inference/dynamics/identifiability.py`, tested in `tests/test_identifiability.py`.
  This proves the tooling is correct before it touches a real model.
- Step 2 (port to Yildirim's actual 2003 lac-operon DDE) requires his published equations and
  parameters, to be taken from the paper rather than reconstructed from memory.

## The quant-to-biology bridge (for step 3, and as a theme)

Stochastic gene-expression noise, parameter inference / identifiability for dynamical models,
single-cell trajectory analysis, optimal experimental design, and model selection are where
probability / stochastic-process / time-series / filtering methods give a genuine edge, distinct from
edge recovery. Specific finance-style tools to map: stochastic differential equations, Kalman /
particle filtering, change-point / regime-switching detection, and signal-versus-noise detection,
applied to adaptation dynamics and gene-expression noise. (Which of these already have published
biology applications versus remain plausible bridges is an open question to check before committing.)
