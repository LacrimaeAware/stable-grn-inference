"""Correctness tests for the identifiability pipeline (experiment 40, roadmap step 1).

The pipeline must recover the textbook fact: in the mRNA -> protein cascade, observing protein only
makes the transcription rate k_m and translation rate k_p non-identifiable (Fisher matrix
rank-deficient; profile likelihood flat), while observing mRNA as well makes them identifiable. If the
tooling does not get this known answer right, it must not be pointed at a real model.
"""

import unittest

import numpy as np

from stable_grn_inference.dynamics import (
    fisher_information,
    fit_mle,
    identifiability_report,
    is_identifiable,
    profile_likelihood,
    simulate_mrna_protein,
)

# true log-parameters: k_m, d_m, k_p, d_p
THETA = np.log([2.0, 0.5, 3.0, 0.8])
T = np.linspace(0.0, 12.0, 40)
SIGMA = 0.05


def _data(channels, seed=0):
    rng = np.random.default_rng(seed)
    traj = simulate_mrna_protein(THETA, T)
    clean = traj[:, list(channels)].ravel()
    return clean + rng.normal(scale=SIGMA, size=clean.size)


class StructuralIdentifiabilityTest(unittest.TestCase):
    def test_protein_only_is_rank_deficient(self):
        rep = identifiability_report(THETA, T, channels=(1,), sigma=SIGMA,
                                     param_names=["k_m", "d_m", "k_p", "d_p"])
        self.assertTrue(rep["rank_deficient"])
        self.assertEqual(rep["rank"], 3)   # one non-identifiable direction (k_m vs k_p)

    def test_both_channels_full_rank(self):
        rep = identifiability_report(THETA, T, channels=(0, 1), sigma=SIGMA)
        self.assertFalse(rep["rank_deficient"])
        self.assertEqual(rep["rank"], 4)


# Bounded settings keep these correctness checks fast: a flat profile (non-identifiable) stays flat at
# any resolution, and a bounded profile rises well above the delta threshold on a coarse grid. The
# structural Fisher-rank tests above are the decisive check; these confirm it via profile likelihood.
FIT = dict(maxiter=500)
PROF = dict(span=2.0, n=7, maxiter=500)


class ProfileLikelihoodTest(unittest.TestCase):
    def test_k_m_flat_with_protein_only(self):
        data = _data((1,), seed=1)
        mle = fit_mle(T, data, (1,), SIGMA, THETA + 0.1, **FIT)
        _, nll = profile_likelihood(0, mle, T, data, (1,), SIGMA, **PROF)  # index 0 = k_m
        self.assertFalse(is_identifiable(nll))            # flat profile -> non-identifiable

    def test_k_m_bounded_with_both_channels(self):
        data = _data((0, 1), seed=2)
        mle = fit_mle(T, data, (0, 1), SIGMA, THETA + 0.1, **FIT)
        _, nll = profile_likelihood(0, mle, T, data, (0, 1), SIGMA, **PROF)
        self.assertTrue(is_identifiable(nll))             # bounded profile -> identifiable

    def test_product_k_m_k_p_is_preserved_under_protein_only(self):
        # the MLE may move k_m and k_p individually but should preserve their product
        data = _data((1,), seed=3)
        mle = fit_mle(T, data, (1,), SIGMA, THETA + np.array([0.5, 0.0, -0.5, 0.0]), **FIT)
        true_prod = THETA[0] + THETA[2]
        self.assertAlmostEqual(mle[0] + mle[2], true_prod, places=1)


class InferenceTest(unittest.TestCase):
    def test_recovers_all_params_when_both_observed(self):
        data = _data((0, 1), seed=4)
        mle = fit_mle(T, data, (0, 1), SIGMA, THETA + 0.2, **FIT)
        np.testing.assert_allclose(np.exp(mle), np.exp(THETA), rtol=0.15)


if __name__ == "__main__":
    unittest.main()
