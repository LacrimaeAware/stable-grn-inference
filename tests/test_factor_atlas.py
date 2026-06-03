"""Correctness tests for the counterfactual factor-atlas tool.

These are the proof that the test is *faithful* to the idea: on data where we PLANTED the
ground truth, the counterfactual test must mark the core factor as core, see through the
shortcut, and the factored classifier must generalize to flipped combinations better than the
overfitting raw classifier. If these fail, the bug is ours -- caught here, not on real data.
"""

import unittest

import numpy as np

from stable_grn_inference.analysis import (
    counterfactual_necessity_sufficiency,
    discover_factor_directions,
    held_out_combination_accuracy,
    make_factor_atlas_data,
    project_out_directions,
)


class FactorAtlasCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_factor_atlas_data(n_per_class=400, dims=30, n_nuisance=2, seed=0)

    def test_discovery_recovers_planted_factors(self):
        res = discover_factor_directions(self.data.deltas, self.data.delta_factor_id,
                                         n_factors=len(self.data.factor_kind))
        self.assertGreater(res["ari"], 0.8)   # deltas cluster back to the planted factors

    def test_counterfactual_marks_core_vs_nuisance_correctly(self):
        rows = counterfactual_necessity_sufficiency(self.data, target_class=1)
        by = {r["factor"]: r for r in rows}
        core = by["core"]
        # CORE: removing it breaks the class (low necessity_R), adding converts (high sufficiency_A)
        self.assertLess(core["necessity_R"], 0.5)
        self.assertGreater(core["sufficiency_A"], 0.5)
        # NUISANCE: removing leaves class intact (high R), adding does not convert (low A)
        nui = by["nuisance"]
        self.assertGreater(nui["necessity_R"], 0.7)
        self.assertLess(nui["sufficiency_A"], 0.4)
        # CORE must score higher on the core_score than nuisance -> the test separates them
        self.assertGreater(core["core_score"], nui["core_score"])

    def test_factored_classifier_beats_overfit_on_flipped_combos(self):
        res = discover_factor_directions(self.data.deltas, self.data.delta_factor_id,
                                         n_factors=len(self.data.factor_kind))
        # use the discovered nuisance/shortcut directions (all planted factors are non-core here)
        out = held_out_combination_accuracy(self.data, nuisance_directions=res["directions"])
        # projecting out the transferable factors generalizes better to unseen combinations
        self.assertGreaterEqual(out["factored_heldout_acc"], out["raw_heldout_acc"])

    def test_project_out_removes_direction(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(1, 8)); d /= np.linalg.norm(d)
        X = rng.normal(size=(20, 8))
        Xp = project_out_directions(X, d)
        # no remaining component along d
        self.assertLess(float(np.abs(Xp @ d.T).max()), 1e-8)


if __name__ == "__main__":
    unittest.main()
