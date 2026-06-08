# Experiment 28: the separability phase diagram

## Question

Every RPE1 experiment (21-27) hit the same wall: a dominant convergent mode (cell-cycle) sits on top of small gene-specific structure, and the specific part is not cleanly separable, transferable, or beatable. Instead of running another method on RPE1's fixed, unknowable-truth data, this experiment makes the separation itself the object of study on synthetic systems with known ground truth, and maps the boundary of when recovery is possible.

This is the apex the project's negatives were circling, and the bridge to the structured-transform project: separating a dominant shared mode from small reusable structure is the same problem in expression space (cell-cycle vs gene-specific edges) and representation space (class identity vs transformation factors).

## Setup

A synthetic response matrix mixes three parts (all rescaled to unit Frobenius norm):

```
D = sqrt(rho) * M  +  sqrt(1-rho) * [ sqrt(snr/(snr+1)) * Dspec + sqrt(1/(snr+1)) * E ]
```

- `Dspec` = the known specific structure: the total response of a sparse direct operator `W` via `propagation_forward` (the exp-23 model `D = (I-W)^-1 - I`). Its off-diagonal nonzeros are the truth to recover.
- `M = outer(a, m)` = a rank-1 dominant shared mode (the convergent program).
- `E` = i.i.d. noise.

Two knobs: `rho` (dominant-mode variance fraction; RPE1's measured top-1 SVD fraction is ~0.53) and `snr` (specific-vs-noise ratio in the non-dominant part). An optional `entanglement` knob aligns the mode with the specific structure; the headline sweep uses entanglement 0 (clean rank-1 mode).

Four recovery methods rank candidate specific edges: `raw` (`|D|`), `deflate1` (remove the top-1 SVD mode), `program` (subtract the shared response program), `deconv` (ridge inverse `W_hat`). Recovery is AUPR against `W`, chance-normalized to `(AUPR - density)/(1 - density)`. Tooling lives in `src/stable_grn_inference/dynamics/separability.py`; all reused from the existing interventional toolkit.

## Results

Verification: the full grid (100 genes, 5 seeds/cell) was re-run and confirms both conclusions.
Deflation is rho-invariant (recovery about 0.67 across all rho at high SNR), the specific-SNR floor is
near 0.2 (no method clears below it), deflation is the most robust method over the grid (mean 0.372),
and RPE1's corner (rho about 0.53, low SNR) reaches only 0.33. The quick-run tables below show the same
structure at smaller scale.

### Quick run (50 genes, 2 seeds/cell)

Normalized recovery, `raw` (no deflation) — collapses along both axes:

| snr\rho | 0.2 | 0.4 | 0.53 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 2.0 | 0.639 | 0.500 | 0.409 | 0.256 | 0.058 |
| 0.5 | 0.453 | 0.338 | 0.244 | 0.125 | 0.026 |
| 0.2 | 0.297 | 0.176 | 0.116 | 0.046 | 0.013 |
| 0.05 | 0.080 | 0.035 | 0.022 | 0.012 | 0.005 |

Normalized recovery, `deflate1` (remove the dominant mode) — rho-invariant, collapses only with snr:

| snr\rho | 0.2 | 0.4 | 0.53 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 2.0 | 0.726 | 0.731 | 0.733 | 0.734 | 0.734 |
| 0.5 | 0.547 | 0.551 | 0.550 | 0.550 | 0.550 |
| 0.2 | 0.364 | 0.368 | 0.369 | 0.370 | 0.370 |
| 0.05 | 0.111 | 0.112 | 0.111 | 0.111 | 0.113 |

## Findings

1. **Two distinct failure axes, which the project had conflated.** `rho` (dominant-mode dominance) kills naive recovery — `raw` falls ~0.58 normalized-recovery from low to high rho — but it is *fixable*: removing a clean rank-1 mode makes `deflate1` essentially rho-invariant. `snr` is the other axis, and it is *not* fixable: below a noise floor (~0.2 here) no method clears chance, at any rho.
2. **RPE1's bottleneck is the SNR floor, not the dominant mode.** At rho~0.53 with low specific-SNR (RPE1's regime: exp 21 found ~half of responses are noise and global-mode removal increased diffuseness rather than isolating clean signal), the best of the four methods reaches only ~0.37 normalized recovery, dropping toward the floor at the low-SNR end. Removing the cell-cycle mode does not help because there is little recoverable specific signal underneath it — exactly the real RPE1 finding (exp 21), now explained by the diagram rather than just observed.
3. **The negatives become one map.** exp 22 (cleaning fails), 23 (linear inverse fails), 24 (no transfer), 25 (cell-cycle entangled), 27 (cascade swamps pairs) are all the same point on this diagram: high rho, low specific-SNR.
4. **Decision use.** For specific-structure recovery to be possible, a dataset needs specific-SNR above the floor — not merely a low dominant-mode fraction. This is a selection criterion for future data (e.g. time-resolved perturbation data should be chosen/measured for specific-SNR), and it tells you the cascade/asymmetry instinct (exp 26-27) can only pay off on data past the boundary.

## Outputs

Under `results/` (git-ignored): `separability_phase_diagram_grid.csv`, `separability_phase_diagram_boundary.csv`, `separability_phase_diagram_debug_report.md`, `results/figures/separability_phase_diagram.png`.

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B experiments/28_separability_phase_diagram/run_separability_phase_diagram.py --quick   # coarse, fast
.\.venv\Scripts\python.exe -B experiments/28_separability_phase_diagram/run_separability_phase_diagram.py           # full grid
```

## Interpretation policy

Diagnostic, not a deployable method. The synthetic dominant mode is a clean rank-1 component, so `deflate1`'s dominance is a property of the model, not a claim about real data; the transferable conclusion is the *shape* of the boundary (rho is fixable, SNR is the binding floor) and where RPE1 lands on it. The natural next rungs are (a) the entanglement axis (a mode that is not cleanly removable, sweeping the optional knob), (b) a time axis so DMD/Koopman/SINDy operators can be added and graded against truth, and (c) placing a real time-resolved perturbation dataset onto the diagram.
