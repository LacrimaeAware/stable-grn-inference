# Research directions (pre-research ideas)

Ideas generated from the project conversation BEFORE the literature pass, deliberately first so the
directions are not anchored on what the field already tried. To be compared against the research and
turned into experiments. Old directions are not discarded; these are additive, and the standing
baselines (direct correlation, GENIE3, self-persistence) are the bar each must beat.

Established context:
- Skeleton (who relates to whom) is recoverable; direct correlation is the strong baseline.
- Direction (who causes whom) does not come from a symmetric second-order statistic (correlation).
- Order is recoverable from static geometry (exp 34, absolute Spearman 0.83) but does not improve
  network recovery; higher-order iterated correlation adds spurious transitive edges.
- Small specific edges under the dominant cascade are bounded by signal-to-noise (exp 28).
- Detectability intuition: a signal is recoverable to the degree its statistic sits far from the null.

## Directions

1. Non-Gaussian orientation (LiNGAM). Correction to an earlier overstatement: direction is
   unidentifiable from CORRELATION (second-order, symmetric), but IS identifiable from static
   observational data when the noise is non-Gaussian, via higher-order moments (LiNGAM / ICA,
   Shimizu et al. 2006). Gene expression is non-Gaussian. This is the principled way to get arrows
   from static data, and it directly realizes the "break free from linearity to get more" intuition:
   the arrow lives in the third and higher moments, not the second. Untested here. Highest priority.
   Question: does LiNGAM-style non-Gaussian orientation recover edge direction from static data where
   correlation cannot, on BoolODE/DREAM4 (exact truth) and on RENGE (TRRUST proxy)?

2. Detectability map. Per candidate edge, measure the distance of its statistic from a permutation
   null; report which edges poke above the cascade noise and which are indistinguishable from random.
   Turns the SNR floor into a per-edge map of what is findable.
   Question: are there a few specific edges detectable above a matched null even under the cascade,
   and are they reproducible across cell halves?

3. Two-stage focus-then-refine. Stage 1: a cheap method finds the candidate skeleton (who is related
   at all). Stage 2: on just those pairs, apply an expensive method (LiNGAM orientation, conditional
   independence, biology priors).
   Question: does restricting the expensive method to a cheaply-found candidate set beat applying it
   to all pairs?

4. Koopman with real observables. Lift genes into nonlinear observables (dictionary / kernel) and fit
   the operator (extended DMD) on the time-series data (DREAM4, BoolODE, RENGE), the weird math done
   properly rather than plain linear DMD.
   Question: does extended-DMD / Koopman beat plain lagged GENIE3 at directed recovery, i.e. does
   nonlinearity buy anything?

5. Richer geometry for cycles/branches. Where 1D ordering failed (cycles, exp 34, 0.55), use a 2D+
   diffusion embedding and test whether cyclic/branching structure is recoverable as a cycle.
   Question: does a 2D embedding recover cyclic network structure that a 1D ordering missed?

6. Diversity-consensus. Run several genuinely different methods (correlation, LiNGAM, order, mutual
   information); keep only edges multiple methods agree on. Agreement across diverse lenses = signal;
   disagreement = drift (the translation-chain-with-drift intuition).
   Question: does a diversity-consensus core beat any single method on real data?

Each is testable against exact truth on BoolODE/DREAM4 and against TRRUST/STRING on RENGE, compared
to the standing baselines. After the literature pass, this list is reconciled with what the field has
tried (incorporate, keep-distinct, or compare) and turned into the next experiment set.

## Reconciliation with the literature (after the research pass)

Headline from the research: no verified published result shows any advanced method (Koopman/EDMD,
CCM/empirical dynamic modeling, non-Gaussian/LiNGAM, 2023-2025 deep/optimal-transport) beating simple
baselines (GENIE3, correlation) on REAL directed-GRN recovery. Every confirmed advantage was on
synthetic or physics/climate systems. The burden-of-proof gap is open. Per-edge detectability limits
were posed but unanswered (no surviving claim). Sources: DTIC AD1170384 / Nat Commun 2023
(PMC10229592); Kausal (arXiv:2505.14828 / Nat Commun Physics 2025); EggFinder (Sogawa-Shimizu 2011).

- Direction 1 (LiNGAM): KEEP, top priority. Confirmed as the principled way to get direction from
  static data via higher moments, and it works at p>>n. Caveats from the research: it assumes
  acyclicity and non-Gaussianity (the EggFinder variant recovers only root variables, not full
  edges); expect it to work on acyclic nets and degrade on cyclic ones and under Gaussian technical
  noise. Test the full pairwise/DirectLiNGAM version on small gene sets.
- Direction 2 (detectability map): KEEP, elevated. The research explicitly leaves per-edge
  detectability / information-theoretic limits unanswered, so this is genuinely open territory.
- Direction 4 (Koopman with observables): DEMOTE. On real bulk RNA-seq, plain linear DMD already
  fits well (R^2=0.93) so nonlinear lifting is not needed; the deep-Koopman advantage (Kausal) is
  physics/climate only with no biological demonstration. Consistent with exp 33 (DMD lost to GENIE3).
- Takens / CCM (background to direction 5): DEMOTE as a causal tool. Documented failure mode of
  misreading generalized synchrony as causation and inferring all-to-all networks on oscillatory
  data, exactly the cyclic case we care about.
- Directions 3 (two-stage), 5 (cycle 2D geometry), 6 (diversity-consensus): KEEP, queued; distinct
  and not addressed by the literature.

Next experiment: exp 35 combines directions 1 and 2 (non-Gaussian orientation + detectability map),
graded against truth on BoolODE/DREAM4 and against the standing baselines.

## Result of exp 35 (directions 1 and 2)

- Direction 1 (LiNGAM / non-Gaussian orientation): negative on real data. The orientation is provably
  correct (it orients a planted non-Gaussian Laplace chain from static data, unit-tested), but on
  BoolODE it does not beat symmetric correlation (directed AUPR 0.29 vs 0.36), is below GENIE3, and is
  worse on the acyclic networks where it should excel (0.23 vs 0.40). This is not a sample-size issue
  (unchanged at 5000 cells); BoolODE's noise is not the LiNGAM kind, so the measure mis-orients. The
  higher-moments idea is sound in theory and fails on this data.
- Direction 2 (detectability map): works as a diagnostic. True edges sit far from the permutation null
  (z=10.7) and 88% clear z>2, but transitive false edges also clear it (z=6.2), so detectability
  separates signal from pure noise but not direct edges from indirect. It quantifies the per-edge SNR
  floor honestly.

Queued and not yet run: direction 3 (two-stage focus-then-refine), 5 (cycle 2D geometry), 6
(diversity-consensus). Each must clear the standing baselines (correlation, GENIE3) on truth.
