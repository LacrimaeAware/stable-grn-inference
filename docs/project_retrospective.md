# Project Retrospective — what we did, what we learned, what we didn't

*A slow, human walk back through the whole project. Written to be read top to bottom by
someone who wants to actually understand it, not to impress anyone. I'll mark clearly where a
finding is **ours**, where it's **known-in-the-field-and-we-re-derived-it**, and where it's a
**dead end**. No hype. If something is modest, it says modest.*

---

## 0. A note on honesty before we start

For a stretch of this project I narrated ordinary results with triumphant words — "headline,"
"strongest result," "breakthrough-ish." That made the work feel like it was lurching between
"alive!" and "dead!" when really it was walking a steady, unglamorous line the whole time. So
this document does the opposite: it states what each thing *is*, flatly, and tells you when a
result is a genuine surprise versus a sanity check versus a known principle we confirmed.

The honest one-line frame for the whole project:

> We did **not** build a network-inference method that beats existing ones. We built a careful
> set of *diagnostic tools*, ran them across three kinds of data, and learned — correctly and
> concretely — *why* the easy hopes don't work and *what the real difficulty is*. The single
> most interesting thread (recovering hidden wiring from how the system responds when poked)
> is still open and untested.

That's not a triumph and it's not a failure. It's the middle of real work.

---

## 1. The 30-second version

- **The goal we started with:** given gene-expression data, figure out which genes regulate
  which (a directed network), and make that more *reliable* by using **stability** information
  (re-run on resampled data, trust edges that keep showing up).
- **What happened to that goal:** the stability idea was **tested and did not hold up.** Along
  the way we learned the deeper reason inference is hard: **you can only recover the *direction*
  of regulation if your data contains directional information** — time, or an actual
  intervention. Static "who-moves-with-whom" data fundamentally can't tell you who causes whom.
- **Where we ended:** on real CRISPR perturbation data, direction *does* come back (and is
  reproducible), but the perturbation response is dominated by a real broad biological program
  (the cell cycle), and standard observational methods barely predict the real effects.

---

## 2. The vocabulary you actually need (plain-language statistics)

This section is the part to *learn from*. Everything we did is built out of these pieces.

### The setup
- **Gene expression**: a big table, rows = cells (or samples), columns = genes, each number =
  how active that gene is.
- **Edge / directed edge**: a claim "gene A regulates gene B," written A → B. The arrow
  matters: A → B is a different claim than B → A.
- **Candidate edges**: every possible arrow we're willing to consider (e.g., every gene → every
  other gene). For 10 genes that's 90; for 100 genes, 9,900.
- **Gold standard**: the known true edges (only available in simulations or curated benchmarks),
  used **only to grade** our guesses, never to make them.

### The grading metrics (and why they disagree)
You rank all candidate edges by a score, then grade the ranking:
- **AUROC** — "if I pick a random true edge and a random non-edge, how often did I rank the true
  one higher?" 0.5 = coin flip, 1.0 = perfect. **Weakness:** it can look respectable even when
  true edges are rare, so it flatters you under imbalance.
- **AUPR** — rewards putting the *rare* true edges near the top of the list. Its "chance" level
  is the **density** (fraction of candidates that are true). This is the **honest** metric when
  true edges are rare. If 2% of edges are true, AUPR ≈ 0.02 is chance.
- **precision@k** — of your top *k* guesses, what fraction are actually true. Very practical:
  "if I act on my 10 best guesses, how many are right?"
- **EPR (early precision ratio)** — precision in your top-(number-of-true-edges) divided by the
  density. "How many times better than random are my top guesses?" 1.0 = no better than chance.

They disagree because they reward different behaviors (overall ranking vs. catching rare
positives vs. the very top of the list). A method can win one and lose another — that recurred
constantly, and it's a feature, not a bug: it means "best" depends on what you care about.

### The methods we used to score edges
- **Correlation** — the dumb, strong baseline. Score A→B by how much A and B move together. It
  is **symmetric** (corr(A,B) = corr(B,A)), so by construction it *cannot tell direction*. Keep
  this in mind — it becomes a key control later.
- **LASSO** — predict each gene from all the others with a linear model, plus a penalty
  (**alpha**) that forces most coefficients to exactly zero. The genes with non-zero
  coefficients are the "regulators," and the coefficient size is the edge score.
  - **alpha = the sparsity knob.** Bigger alpha → fewer edges survive. A big discovery-by-tuning:
    *the best alpha tracks how sparse the true network is.* Dense network → small alpha; sparse
    network → big alpha.
- **Elastic Net** — LASSO's cousin (mixes the L1 penalty with a smoother L2). We tried it; it
  was rarely the winner.
- **GENIE3** — the well-known "serious" method: for each gene, train a **random forest** (a
  bunch of decision trees) to predict it from all the others, and use each predictor's
  **importance** as the edge score. Non-linear, captures interactions. Treated as the method to
  beat.
- **Rank fusion** — combine several methods' rankings (e.g., average their ranks). The idea:
  if methods make *different* mistakes, combining them cancels noise.

### The "is it reliable?" machinery (the original thesis)
- **Bootstrap / subsampling** — re-run a method on many random sub-samples of the data.
- **Stability selection** — score an edge by how *often* it shows up across those re-runs
  (its "selection frequency"). The bet: real edges are stable, noise edges flicker.
  **Meinshausen–Bühlmann** is a famous theorem giving a formula that bounds how many false
  positives you should expect. This whole idea was the original Track A pitch.

### The diagnostic ideas we added (the actually-useful part)
- **Skeleton vs. orientation** — split the problem in two: "did I find the right *pair* of
  genes?" (skeleton, undirected) vs. "did I point the *arrow* the right way?" (orientation).
  This let us ask *where* errors come from. A clean trick: a symmetric method like correlation
  must score orientation at exactly **0.50** (it can't tell direction) — a built-in sanity check.
- **Square-root / theory LASSO** — instead of grid-searching alpha, set it from a formula
  √(2·log p / n), where *p* = number of genes, *n* = number of samples. No noise estimate needed.
  The lesson when it works: the right amount of regularization is **predictable from "how many
  things am I choosing among and how many samples do I have,"** not magic.
- **Paired confidence intervals** — with only 5 networks (DREAM4), a single average is fragile.
  Pairing the comparison network-by-network and bootstrapping gives an interval, so we can say
  "this difference is real (interval excludes 0)" vs. "this is noise."
- **Wasserstein distance** — a way to measure how different two whole *distributions* are (the
  "earth-mover" distance: how much work to reshape one pile of sand into the other). We used it
  to measure how much knocking out gene A shifts gene B's entire distribution vs. control.
- **SVD / low-rank / participation ratio** — tools to ask "is this big matrix really just a few
  repeated patterns?" If a few directions explain most of it, it's **low-rank** (lots of shared
  structure). Participation ratio = "effective number of things that matter" (1 = one thing
  dominates; n = everything matters equally).
- **Split-half stability** — split the cells in two at random, compute the thing on each half,
  check if the halves agree. High agreement = real; low = noise. This is how we *verified*
  results without a gold standard.
- **Residualization** — "regress out" a known factor (say, cell-cycle) and keep the leftover —
  "the part not explained by that factor."

### The biology words
- **knockout** = gene fully disabled. **knockdown / CRISPRi** = gene's activity strongly reduced.
- **multifactorial** = many small random nudges at once (a DREAM4 regime).
- **time-series** = repeated measurements over time.
- **control / non-targeting** = cells where nothing was perturbed (the baseline).
- **observational data** = you just watched the system. **interventional data** = you actively
  changed something (perturbed a gene) and watched what moved. This distinction turns out to be
  the whole story.

---

## 3. The story in three acts

### Act I — DREAM4: the simulator training ground (experiments 1–14)

DREAM4 is an old, *simulated* benchmark: 5 small networks (10 genes) and 5 bigger ones (100
genes), with the true edges known. It was where we built and sharpened all the machinery.

What we asked and found, cluster by cluster:
- **Baselines (exp 1–4):** plain **correlation** was a surprisingly strong baseline. Tuning
  LASSO's alpha mattered a lot. **GENIE3** (trees) became the strongest "serious" method by AUPR
  in several settings — but correlation stayed competitive, especially for finding hub genes.
  *(Known territory, cleanly reproduced.)*
- **Stability (exp 2–3):** stability-correlation gave a *small* AUPR bump in small-data settings,
  enough to justify continuing — but it never dominated. First yellow flag for the thesis.
- **Time helps (exp 7):** when we used *temporal order* (predict gene-at-t+1 from genes-at-t)
  instead of same-time data, recovery jumped a lot (AUPR ~0.30 → ~0.53). **Lesson: directional
  information in the data — here, time — is worth more than a fancier method.** This is the seed
  of the entire later story.
- **The Size10 winner that wasn't (exp 8–10):** a specific sparse model (`lasso, level target,
  include-self, alpha=0.03`) looked great on 10-gene networks — including a nice
  direction-accuracy advantage. Then we scaled to 100 genes and **it collapsed**: won 0 of 5
  networks, lost its direction advantage entirely. **Lesson: a result on tiny networks can be a
  small-network artifact.** *(This is an honest negative we caught ourselves.)*
- **Understanding *why* (exp 11–13):** we explained the winners instead of adding more. alpha is
  a **density knob** (best alpha rises as the network gets sparser). "Include-self" helps because
  a gene's own past strongly predicts its future (**self-persistence** — your "stays in motion
  until something interferes" intuition, made literal), but that term *dominates* and is
  dangerous to read as regulation. Fusion helps *only* when methods make different mistakes.
- **A deployable recipe (exp 14):** you can pick alpha **without** the gold standard (via CV or
  BIC) and keep 96–100% of the best-possible AUPR; method-agreement gives a *calibrated*
  confidence (high-confidence edges really are likelier true). Modest but genuinely useful.

**Act I verdict:** we learned the craft and drew real statistical lessons, but the headline
candidate didn't scale and stability wasn't winning. Useful, not exciting.

### Act II — BEELINE: the reality check (experiments 15–18)

BEELINE is a *real* single-cell benchmark (modern-ish, with curated true networks). We ported
the **exact** DREAM4 diagnostics to it to ask: *do the DREAM4 conclusions transfer?*

- The big DREAM4 finding had been "the hard part is finding the right *pair*; once found, the
  *direction* is basically free (orientation accuracy 0.88–0.96)."
- On BEELINE's **static** single-cell data, that **broke**: orientation accuracy was weak and
  wildly network-dependent (one network ~0.4, i.e. *worse than a coin flip given the pair*;
  another ~1.0). The symmetric-correlation control sat at exactly 0.50, as it must.
- The theory-alpha lesson held. The stability-selection negative held. Fusion's help did **not**
  transfer (helped one network, hurt others).

**Act II verdict (the real lesson of the whole project starts here):** *the DREAM4 "direction is
easy" result was an artifact of time-series data.* On static snapshots, direction is close to
**non-identifiable** — not because our method was bad, but because a snapshot of who-moves-with-whom
genuinely doesn't contain the answer. **Identifiability is set by the data regime, not the
algorithm.** *(This is known in causal inference, but we re-derived it concretely and it
reframed the project.)*

### Act III — CausalBench / RPE1: real intervention (experiments 19–22)

To get direction back, we needed data where someone *actually intervened*. CausalBench packages
real **CRISPR perturbation** data (Replogle RPE1: ~248k cells, ~8.7 GB raw). We wrote a
memory-careful loader and reduced it to a clean working set: **651 genes, ~140k cells, 11,485
untouched control cells.**

The conceptual shift (this is the Track-B bridge): stop obsessing over single edges, and look at
the whole **response vector**. Poke gene *g*, and the system shifts by

> Δ_g = (average expression when g is perturbed) − (average expression in controls).

Stack all those vectors into a **response matrix** D, and study its *geometry*. This is exactly
your water/stick intuition: poke the pond, watch the whole flow field move.

What we found:
- **(exp 20) Direction comes back under intervention.** Perturbing A moves B more than perturbing
  B moves A, decisively, for ~61% of gene pairs (vs. 50% undecidable for static scores). And —
  damningly for observational methods — the *observational* guess at direction was actually
  **anti-correlated** (0.33) with the interventional truth: static co-expression doesn't just
  fail to find direction, it points the wrong way.
- **(exp 21) The response is real and verifiable.** Quality check: knock down a gene and 99.7%
  of the time *that gene* drops — the experiment is real. The response matrix is **low-rank**
  (one dominant pattern explains 53% of all the variance). About half of perturbations give a
  **split-half-reproducible** response. And the directional asymmetry is **reproducible across
  independent halves of the cells (~0.64–0.70 vs. 0.5)** — the one place we could *verify*
  direction without an answer key. *(This reproducibility check is genuinely clean — but it's a
  verification that direction is stable, not a new method.)*
- **(exp 22) The broad response is real biology, not removable junk.** That dominant low-rank
  pattern is a **cell-cycle / proliferation program** (top genes: CCNB1, MCM3, RRM2, DNMT1,
  histones, tubulins). We tried to subtract it to reveal sparse "direct" effects — and it didn't
  work: removing it *hurts* (it's real signal), removing technical covariates does ~nothing. And
  the punchline: **no observational method predicts the real interventional response well** —
  correlation ρ≈0.13, sparse ≈0.04, **GENIE3 ≈ 0.** A standard, respected GRN method has
  essentially zero alignment with what the perturbations actually do.

**Act III verdict:** the response-geometry view is the most *interesting* shape the project has
had, and it produced two solid things — a verifiable directionality signal, and a clean negative
about observational methods. But it also hit a wall: the response is dominated by a broad real
biological program, and we could not yet isolate the sparse "direct wiring" from it.

---

## 4. The questions we asked, and the answers we got

- **Does stability-aware ranking make GRN inference more reliable?** → **No** (the original
  thesis). Small bumps in tiny data, but formal stability selection underperformed a single
  well-tuned fit, and its false-positive bound was too loose to use.
- **Is correlation too dumb to bother with?** → **No.** It's a strong baseline everywhere; never
  drop it from a comparison.
- **Does a fancier model (LASSO/GENIE3/MLP) reliably beat it?** → **Not reliably.** GENIE3 wins
  some AUPR comparisons; nothing dominates; an MLP didn't help.
- **Is the hard part finding the pair, or pointing the arrow?** → **Depends on the data.**
  Time-series: pointing the arrow is easy. Static snapshots: pointing the arrow is nearly
  impossible. Intervention: it comes back.
- **Is the right amount of regularization (alpha) magic, or predictable?** → **Predictable** from
  sample-complexity theory (√(log p / n)). One of our cleaner, transferable lessons.
- **Does fusion (combining methods) help?** → **Only when methods make complementary errors** —
  in the hard, high-dimensional regime, not universally.
- **Did the tiny-network winner scale up?** → **No.** Small-network artifact.
- **Does intervention give direction?** → **Yes, and it's reproducible** (~0.64–0.70 across cell
  halves). The strongest *positive* on real data.
- **Does observational co-expression predict what interventions do?** → **Barely.** Correlation
  ρ≈0.13, GENIE3≈0. The strongest *cautionary* result.
- **Is the "everything moves" broadness just noise we can clean off?** → **No.** It's a real
  cell-cycle program. Removing it hurts.

---

## 5. What we actually learned (the keepers)

These are true and they survive scrutiny. Marked by how novel they are.

1. **Identifiability is set by the data regime, not the estimator.** Direction needs time or
   intervention; static data can't supply it. *(Known in causal inference; we re-derived it
   concretely on three datasets — valuable as understanding, not as discovery.)*
2. **The regularization penalty is theory-predictable** (√(log p / n) ≈ tuned alpha), across all
   three regimes. *(A clean, transferable, slightly-nice result.)*
3. **The original stability-selection thesis does not hold** at these sample sizes. *(A real,
   honest falsification — see §7.)*
4. **Observational network methods barely predict interventional effects** (GENIE3 ≈ 0 on RPE1).
   *(Our cleanest negative finding; partly known motivation for Perturb-seq, but we measured it
   directly and bluntly. Has a caveat: our target was broad, see §6.)*
5. **Intervention direction is reproducible** across independent cells. *(A solid verification,
   not a method.)*
6. **A portable diagnostic toolkit** (skeleton/orientation split, theory-alpha, fusion-complementarity
   test, response geometry) that runs identically across regimes and is tested on synthetic data.
   *(Real engineering value; reusable.)*

---

## 6. What failed, and why

- **The stability thesis.** It was plausible but the math (MB bound) is too loose when you have
  far more genes than samples, and selection-frequency ranking just wasn't as good as one
  careful fit. Failed cleanly.
- **The Size10 sparse "winner."** Looked great at 10 genes, evaporated at 100. A small-network
  effect we mistook (briefly) for a real method.
- **Direct-effect cleaning (exp 22).** The hope that subtracting a global mode would reveal sparse
  direct wiring. It failed because the global mode is *real biology*, not an artifact — you can't
  subtract the cell cycle and have a clean network fall out.
- **The meta-failure: treating the benchmark leaderboard as the goal.** Chasing AUPR led to a lot
  of motion and little insight. The project only got interesting when it switched from "win the
  benchmark" to "understand what's identifiable."
- **The honesty failure (mine).** Overhyping ordinary results, which made the whole thing feel
  like a rollercoaster instead of a steady climb. Corrected by stating priors up front.
- **A caveat on finding #4:** the "observational methods don't predict interventional response"
  number is measured against a *broad* target (the response is dense and cell-cycle-dominated),
  so part of the low score is "the target is broad," not purely "the methods are useless." It's
  real but not as clean as a single number suggests.

---

## 7. Did we falsify anything? Did we discover anything?

**Falsify — yes, several things, cleanly:**
- stability-aware ranking as a reliability win (no),
- the tiny-network sparse candidate as a general method (no),
- "orientation is easy once you find the pair" as a universal claim (no — regime-specific),
- the global response as removable technical noise (no — it's biology).

Falsifications are real results. They're how a project earns the right to its eventual claims.

**Discover (new to the field) — honestly, no clear discovery.** What we have is: a correct
re-derivation of a known principle (identifiability ↔ data regime), one mildly nice transferable
result (theory-alpha), one blunt negative (observational ≠ interventional), and good infrastructure.
The response-geometry *framing* is interesting but it's an **existing active subfield** (Perturb-seq
modeling, CausalBench's own philosophy). So we're *joining* a conversation, not founding one. I
should have told you that earlier.

**Net:** intellectually we are in the best, clearest shape the project has had. In terms of "a
novel result that wins something," we are not there, and pretending otherwise was the mistake.

---

## 8. The lesson both of your projects share

Your Track B (representations) and this project (regulation) hit the **same shape of wall** at the
same time, and it's worth internalizing because it's a general research trap:

> **Don't test a sophisticated idea on a target that a dumb, global summary already solves.**

- Track B: "thickness" is a *global energy* factor → a dumb radial-energy summary beats
  scattering and a CNN. The fancy representation had nothing to beat.
- Track A: a knockout triggers a *global cell-cycle program* → "does A affect B at all" is
  dominated by that broad shift. The fancy structure had nothing to beat.

In both, the move forward is the same: **find the local, specific structure that the global
summary is blind to.** That's not failure; it's the design correction that makes the next
experiment meaningful. (And it's why your instinct to slow down and think is the right one — the
fix here is conceptual, not "run more methods.")

---

## 9. What's still alive (and your intuition is pointing right at it)

Read this part last, when you're not fried. It's the one genuinely open, genuinely interesting
thread, and it's *yours*.

You keep describing it: poke the system, the whole pond shimmers, you see the net flow but not
the little vectors that made it. Here is that intuition as a real, mostly-unexplored math problem
for this data:

- The response matrix **D** we measured is the **total** effect of each poke — direct wiring
  *plus* everything that propagated downstream *plus* the cell-cycle current. It's the shimmer.
- The thing we actually want — the sparse **direct** wiring (A literally touches B) — is *hidden
  inside* D. In a simple linear model of regulation, the total response is the direct network run
  through a feedback/propagation step: schematically **D ≈ (I − W)⁻¹**, where **W** is the sparse
  direct network. The shimmer is the wiring *after the wave spreads out.*
- So the real question — the stick-in-the-water question — is an **inverse problem**: *given the
  total response D, solve backward for the sparse generator W.* Find the stick that bent the flow,
  instead of describing the flow. There's existing math for this flavor (network deconvolution,
  recovering interaction matrices from response/covariance), but **we have not tried it on this
  data**, and it lines up exactly with your momentum/flow intuition.
- **Honest odds:** maybe a coin flip that it works on real, noisy, cell-cycle-dominated data.
  It might deflate like the others. But it is the only thread that is *both* genuinely interesting
  *and* not a re-confirmation of something known — and it came from your gut, not a benchmark.

Two smaller things are also not dead:
- The **verifiable-orientation** idea (check direction by cross-split reproducibility instead of a
  gold standard) is a legitimately reusable evaluation trick.
- **Self-persistence as the universal baseline** for time-series — your "stays in motion until
  something interferes" point. We saw it dominate the gene models; it's a sound default prior for
  any momentum-like system, and the *residual* (what persistence fails to predict) is exactly the
  "something pushed it" signal worth modeling.

**My honest recommendation, when you're ready (no rush):** if any thread pulls you, it should be
the inverse/deconvolution one — not because it's safe, but because it's the only one that's both
interesting and yours, it's cheap to try, and you can stop the moment it stops being fun.
Everything is committed and will be exactly here. Nothing evaporates.
