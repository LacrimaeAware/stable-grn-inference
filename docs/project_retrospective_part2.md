# Project Retrospective, Part 2 — the hands-on half (exp 23–25 + the theory-crafting)

*The first retrospective covered exp 17–22, mostly run heads-down. This half was different:
it was a back-and-forth — pushing on every step, theory-crafting the way forward together —
and it's where the project's real direction finally surfaced (which turned out not to be genes).
Plain language, honest sizes, no verdict-stamps.*

## What we ran (exp 23–25), plainly

- **Exp 23 — the inverse, "find the stick in the water."** Try to recover the sparse direct
  wiring **W** from the total response (D ≈ (I−W)⁻¹). On a *clean toy* where that assumption is
  exactly true, it worked perfectly. On real RPE1 it didn't help — sparser but noisier, no
  reconstruction gain. Killed cleanly, as pre-registered.
- **Exp 24 — transferable structure.** Can the other perturbations predict a held-out one's
  effect, beyond the cell-cycle average? No. Each perturbation is reproducible but
  *individualistic* — not a shared low-rank code.
- **Exp 25 — the counterfactual factor atlas (the sub-feature idea).** Built the idea as a real,
  tested tool and *proved it on planted ground truth*: it correctly separates a core feature
  from a nuisance, **sees through a shortcut** spuriously glued to a class, and beats overfitting
  on unseen combinations (a clean positive control). Applied to genes it did **not** transfer —
  the cell-cycle program is entangled with real function, so you can't subtract it to reveal a
  clean core. (A single seed faked a positive; a multi-seed check caught it.)

## What we worked out in the conversations (the real value of this half)

- **The third-gene / chain idea.** Pairwise features *provably* can't tell a direct wire from a
  chain A→C→B — a chain is invisible unless you bring in the third gene. On clean toy data, a
  mediation / transitive-reduction score using the third gene hit **~0.98**. On real RPE1 it fell
  apart (the "directness" score was *less* reproducible than the raw effect). The idea is sound;
  real data isn't clean enough for it.
- **The shape idea.** Don't collapse a relationship to one linear number — fit the *shape* of how
  B moves with A. A crude proxy found a faint, correctly-directed signal; the honest
  curve-fitting version found essentially **no** nonlinearity (a flexible curve beat a straight
  line by 0.002). Weak in this data — and in resting cells gene pairs barely predict each other
  at all (R² ≈ 0.05).
- **The cascade / "whirlpool" — the finding that explains all the faint signals.** Knocking out
  almost *any* essential gene triggers the **same convergent damage program** (cell-cycle arrest),
  which moves hundreds of genes together and drowns the specific A→B signal. It is real
  regulation, but *generic and convergent* — many causes, one shared downstream pattern. The
  specific wiring is a tiny signal riding on top, **entangled** with the cascade (a real A→B
  effect may act *through* the cell cycle), which is exactly why subtracting the cascade doesn't
  reveal a clean core. Also learned: the cascade is a *response* phenomenon — 53% of the
  perturbation response but only ~4% of resting-cell variation.
- **The field context (looked it up, not from memory).** RPE1/Perturb-seq is the modern frontier:
  a real competition (the GSK.ai **CausalBench Challenge**, won by PSGRN) and a standard benchmark
  (PerturBench). The best models are deep nets with gene **knowledge graphs** (GEARS) and
  foundation models (scGPT, Geneformer, CPA). But the field's own benchmarks find **simple
  baselines often match or beat them**, and the problem is **unsolved**. So our faint signals
  aren't incompetence — they're the actual state of the art. And the leading approach (knowledge
  graphs) is exactly the "use known structure" instinct that kept coming up.

## Honest bottom line on the gene project

- **No novel method, no competitive result.** On RPE1 specifically, that's because it's a
  genuinely unsolved frontier dominated by the cascade — something even foundation models can't
  cleanly crack with the tools available here.
- **On the clean simulator (DREAM4 Size10), edge recovery was decent** — AUPR ~0.65 vs ~0.33 for
  plain correlation, so the methods roughly doubled the baseline. Real, but *expected* on clean
  simulated data, not novel.
- **Durable products:** a portable diagnostic framework, a set of honest negatives that *match
  the field's own published conclusions*, and — most valuable — a clear map of *why* the problem
  is hard and *where the intuitions actually live in mathematics.*

## The pivot (the real outcome of this half)

The actual interest was never genes — it's **decomposing dynamic data into mathematical
components**. That is a real, named field, and the scattered intuitions map onto it precisely:

- "self-persistence + something pushed it" → **state-space / dynamical-systems models** (`dx/dt = f(x) + u`)
- "everything moves together = a frequency; nullify it, measure deviations" → **Dynamic Mode
  Decomposition (DMD) / Koopman operator theory** — *born from fluid dynamics; the water
  metaphors weren't metaphors*
- "classify the components / recover the rule" → **SINDy** (sparse identification of nonlinear
  dynamics — literally "recover the differential equation from movement data")
- "chain reactions" → **network dynamics / transfer operators**

The gene cascade was one ugly, real-world instance of "subtract the dominant mode, find the small
structure underneath." The clean place to *learn* that is synthetic dynamical systems where the
components are known and you can grade yourself — the "LeetCode for dynamic decomposition" idea,
which is a genuinely shippable project in its own right.

## Meta-lessons (carry these forward)

1. **Reason about geometry/identifiability BEFORE running.** Predict whether a thing can possibly
   work given what the data actually contains; test only to confirm. (The biggest process flaw of
   this round — running, then being surprised, instead of predicting.)
2. **Killing a direction in two days is fast, good science**, not failure.
3. **Intuition is real but only counts once formalized** against a known theorem — and that
   bridge (words ↔ established math) is where the AI is most useful here. Tonight's
   whirlpool → DMD was a clean example.
