# Notes and next steps

Short author notes on what this project was for and where it goes next.

## Purpose

The underlying interest is decomposing dynamic data into reusable components, not gene biology specifically. GRN inference was a concrete test case: a system where many variables move together and the goal is to recover which variable drives which. The three datasets (DREAM4, BEELINE, RPE1) were chosen to vary one factor: how much directional information the data contains.

## Lessons

- Whether direction is recoverable is set by the data structure, not the method. Time-series and interventions contain directional information; static co-expression does not. A method cannot recover what the data does not encode.
- A result on a clean simulator does not imply a result on real data. Several methods that scored near-perfect on synthetic systems dropped to near-random on RPE1.
- The bottleneck on RPE1 is a dominant convergent component (cell-cycle), not a missing method. Specific effects are small relative to it and not cleanly separable.
- Reason about the data geometry before running a test. Predict whether a method can work given what the data contains, then test to confirm.

## Direction going forward

The general problem (separating a dominant shared mode from smaller specific structure in dynamic data) has established machinery: Dynamic Mode Decomposition, Koopman operator theory, and SINDy. The plan is to build fluency in these on controlled synthetic dynamical systems where the components are known and recovery can be graded directly:

1. Linear systems: recover modes with SVD and DMD.
2. Nonlinear systems: SINDy and Koopman.
3. Driven systems: separate the autonomous dynamics from an external input.
4. Noisy and dominant-mode systems: recover small structure under a strong shared component (the RPE1 case, with ground truth).

A graded set of such problems, with known answers, is a self-contained side project (a practice ladder for dynamic-system decomposition).
