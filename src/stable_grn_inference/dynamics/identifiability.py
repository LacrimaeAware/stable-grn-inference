"""Parameter identifiability and inference for dynamical (ODE) models (experiment 40, roadmap step 1).

The math/stats contribution type in mathematical biology that fits a statistics background: given a
mechanistic model and data, which parameters can actually be estimated, how well, and from what
measurements? This module is the pipeline (simulate, fit by maximum likelihood, profile likelihood,
Fisher information), validated on a textbook gene-expression model where the answer is known
analytically: a two-state mRNA -> protein cascade

    dm/dt = k_m - d_m m,    dp/dt = k_p m - d_p p,

observed in protein only, makes the transcription rate k_m and translation rate k_p NON-identifiable
(protein depends on them only through the product k_m k_p), while observing mRNA as well makes them
identifiable. The pipeline must recover that known fact, which is the correctness check before the
tooling is pointed at a real model (e.g. Yildirim's lac-operon DDE).

Parameters are handled in log space (positivity, and the mRNA->protein degeneracy is exactly the
log(k_m) - log(k_p) direction).
"""

from __future__ import annotations

import numpy as np


def simulate_mrna_protein(theta, t, *, m0: float = 0.0, p0: float = 0.0,
                          rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Simulate the mRNA->protein cascade. ``theta`` = log([k_m, d_m, k_p, d_p]). Returns (len(t), 2).

    ``rtol`` / ``atol`` are the ODE solver tolerances; the structural-identifiability answer is
    insensitive to them, so tests can pass a looser tolerance to run the inner optimizers cheaply.
    """
    from scipy.integrate import solve_ivp

    k_m, d_m, k_p, d_p = np.exp(np.asarray(theta, dtype=float))

    def rhs(_t, y):
        m, p = y
        return [k_m - d_m * m, k_p * m - d_p * p]

    t = np.asarray(t, dtype=float)
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), [m0, p0], t_eval=t, rtol=rtol, atol=atol)
    return sol.y.T


def observed(traj, channels) -> np.ndarray:
    """Flatten the observed channels (0 = mRNA, 1 = protein) of a trajectory into a vector."""
    return np.asarray(traj)[:, list(channels)].ravel()


def neg_log_likelihood(theta, t, data, channels, sigma, *, simulate=simulate_mrna_protein) -> float:
    """Gaussian negative log-likelihood of the observed channels (up to a constant)."""
    pred = observed(simulate(theta, t), channels)
    return float(np.sum((pred - np.asarray(data)) ** 2) / (2.0 * sigma ** 2))


def fit_mle(t, data, channels, sigma, theta0, *, simulate=simulate_mrna_protein,
            maxiter: int = 2000, xatol: float = 1e-6, fatol: float = 1e-9):
    """Maximum-likelihood fit (Nelder-Mead on the log-parameters). Returns the MLE theta.

    ``maxiter`` bounds the runtime. On a non-identifiable direction the objective is flat, so the
    simplex never meets the tolerance and would otherwise run to the iteration cap; keep the cap modest.
    """
    from scipy.optimize import minimize

    def obj(theta):
        return neg_log_likelihood(theta, t, data, channels, sigma, simulate=simulate)

    res = minimize(obj, np.asarray(theta0, dtype=float), method="Nelder-Mead",
                   options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter})
    return res.x


def profile_likelihood(index, theta_mle, t, data, channels, sigma, *, grid=None,
                       span: float = 2.0, n: int = 21, simulate=simulate_mrna_protein,
                       maxiter: int = 2000, xatol: float = 1e-6, fatol: float = 1e-9):
    """Profile likelihood for parameter ``index``: fix it across a grid, re-optimize the rest, record
    the negative log-likelihood. Returns (grid in log space, nll profile).

    ``maxiter`` bounds the inner re-optimization (see :func:`fit_mle`); a flat profile is the signal of
    non-identifiability, so the cap controls runtime without changing the flat-versus-bounded verdict.
    """
    from scipy.optimize import minimize

    theta_mle = np.asarray(theta_mle, dtype=float)
    if grid is None:
        grid = np.linspace(theta_mle[index] - span, theta_mle[index] + span, n)
    free = [j for j in range(len(theta_mle)) if j != index]
    nll = []
    for val in grid:
        def obj(free_theta):
            th = theta_mle.copy()
            th[index] = val
            th[free] = free_theta
            return neg_log_likelihood(th, t, data, channels, sigma, simulate=simulate)
        res = minimize(obj, theta_mle[free], method="Nelder-Mead",
                       options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter})
        nll.append(float(res.fun))
    return np.asarray(grid), np.asarray(nll)


def is_identifiable(nll_profile, *, delta: float = 1.92) -> bool:
    """Practically identifiable if the profile rises by more than ``delta`` (95% CI for one parameter)
    above its minimum on BOTH sides within the scanned range, i.e. a finite confidence interval."""
    nll = np.asarray(nll_profile, dtype=float)
    k = int(np.argmin(nll))
    left = nll[:k + 1].max() - nll[k] if k > 0 else 0.0
    right = nll[k:].max() - nll[k] if k < len(nll) - 1 else 0.0
    return bool(left > delta and right > delta)


def fisher_information(theta, t, channels, sigma, *, simulate=simulate_mrna_protein, eps: float = 1e-5):
    """Fisher information matrix FIM = J^T J / sigma^2 (J = sensitivity of the observed output to the
    log-parameters), by central finite differences. A near-zero eigenvalue is a non-identifiable
    direction."""
    theta = np.asarray(theta, dtype=float)
    base = observed(simulate(theta, t), channels)
    J = np.zeros((base.size, theta.size))
    for j in range(theta.size):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        J[:, j] = (observed(simulate(tp, t), channels) - observed(simulate(tm, t), channels)) / (2 * eps)
    return (J.T @ J) / (sigma ** 2)


def identifiability_report(theta, t, channels, sigma, *, param_names=None, simulate=simulate_mrna_protein):
    """FIM eigenvalues and rank for a parameter set and observation scheme (structural-identifiability
    proxy). Returns a dict with eigenvalues, condition number, and the rank deficiency."""
    F = fisher_information(theta, t, channels, sigma, simulate=simulate)
    ev = np.linalg.eigvalsh(F)
    ev = np.clip(ev, 0, None)
    top = ev.max() if ev.size else 0.0
    rank = int(np.sum(ev > 1e-9 * (top + 1e-30)))
    return {
        "eigenvalues": ev,
        "condition_number": float(top / (ev[ev > 0].min() if np.any(ev > 0) else np.nan)),
        "rank": rank,
        "n_params": len(theta),
        "rank_deficient": rank < len(theta),
        "param_names": list(param_names) if param_names else None,
    }
