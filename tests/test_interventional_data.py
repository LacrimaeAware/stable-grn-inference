"""Tests for the interventional (perturbation) adapter shape (experiment 19).

Synthetic fixtures only; no real CausalBench/Replogle download required.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stable_grn_inference.data import (
    build_candidate_edges_from_perturbations,
    direct_effect_filter,
    interventional_effect_matrix,
    interventional_orientation_asymmetry,
    load_causalbench,
    load_interventional_frames,
    load_replogle_raw_h5ad,
    make_synthetic_interventional,
    perturbation_response_matrix,
    response_low_rank,
    response_sparsity,
    split_half_stability,
)


class CandidateEdgeTest(unittest.TestCase):
    def test_sources_restricted_to_perturbed(self):
        genes = ["A", "B", "C"]
        edges = build_candidate_edges_from_perturbations(genes, ["A"])
        self.assertEqual(set(edges["source"]), {"A"})
        self.assertEqual(set(zip(edges["source"], edges["target"])), {("A", "B"), ("A", "C")})
        self.assertNotIn("A", set(edges["target"]))  # no self edges

    def test_tf_list_further_restricts_sources(self):
        genes = ["A", "B", "C"]
        edges = build_candidate_edges_from_perturbations(genes, ["A", "B"], tf_list=["B"])
        self.assertEqual(set(edges["source"]), {"B"})


class LoadFramesTest(unittest.TestCase):
    def test_control_mask_and_metadata(self):
        expr = pd.DataFrame(
            np.arange(12, dtype=float).reshape(4, 3),
            index=["c0", "c1", "c2", "c3"],
            columns=["A", "B", "C"],
        )
        perturb = pd.Series(["control", "A", "B", "control"], index=expr.index)
        ref = pd.DataFrame([("A", "B")], columns=["source", "target"])
        ds = load_interventional_frames("d", expr, perturb, reference_edges=ref, reference_kind="exact")
        self.assertEqual(ds.metadata["n_control_cells"], 2)
        self.assertEqual(ds.perturbed_genes, ["A", "B"])
        self.assertEqual(ds.metadata["n_true_edges"], 1)
        self.assertTrue(ds.is_control.loc["c0"])
        self.assertFalse(ds.is_control.loc["c1"])
        # only perturbed genes are sources
        self.assertEqual(set(ds.candidate_edges["source"]), {"A", "B"})

    def test_no_reference_gives_zero_true(self):
        expr = pd.DataFrame(np.ones((3, 2)), index=["c0", "c1", "c2"], columns=["A", "B"])
        perturb = pd.Series(["control", "A", "A"], index=expr.index)
        ds = load_interventional_frames("d", expr, perturb)
        self.assertFalse(ds.metadata["has_reference"])
        self.assertEqual(int(ds.edge_labels["is_true"].sum()), 0)


class SyntheticSemTest(unittest.TestCase):
    def test_fixture_schema(self):
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=5, n_cells_per_condition=40, seed=0
        )
        # control + one block per gene
        self.assertEqual(expr.shape, (40 * (5 + 1), 5))
        self.assertEqual(set(perturb.unique()) - {"control"}, set(expr.columns))
        self.assertTrue(set(zip(true_edges["source"], true_edges["target"])))

    def test_interventional_effect_separates_true_from_false(self):
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=6, n_cells_per_condition=100, edge_density=0.4, seed=2
        )
        ds = load_interventional_frames("d", expr, perturb, reference_edges=true_edges, reference_kind="exact")
        eff = interventional_effect_matrix(ds).merge(ds.edge_labels, on=["source", "target"])
        mean_true = eff.loc[eff["is_true"] == 1, "effect"].mean()
        mean_false = eff.loc[eff["is_true"] == 0, "effect"].mean()
        self.assertGreater(mean_true, mean_false)

    def test_orientation_asymmetry_recovers_direction(self):
        # On the acyclic SEM fixture, intervention asymmetry should orient ~perfectly,
        # while a symmetric observational score cannot (this is the whole point of exp19).
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=7, n_cells_per_condition=120, edge_density=0.4, seed=1
        )
        ds = load_interventional_frames("d", expr, perturb, reference_edges=true_edges, reference_kind="exact")
        result = interventional_orientation_asymmetry(ds)
        self.assertGreaterEqual(result["n_pairs_both_perturbed"], 1)
        self.assertGreaterEqual(result["accuracy"], 0.9)


class CausalBenchLoaderTest(unittest.TestCase):
    def test_load_causalbench_from_synthetic_h5ad(self):
        try:
            import anndata as ad
        except Exception:  # pragma: no cover
            self.skipTest("anndata not installed")
        rng = np.random.default_rng(0)
        genes = ["GA", "GB", "GC", "GD"]
        # cells: 150 control + 150 each perturbing GA, GB (GC/GD never perturbed)
        obs_gene, X = [], []
        for label, n in [("non-targeting", 150), ("GA", 150), ("GB", 150)]:
            obs_gene += [label] * n
            X.append(rng.poisson(5, size=(n, len(genes))))
        X = np.vstack(X).astype(float)
        import pandas as pd

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"gene": obs_gene}),
            var=pd.DataFrame({"gene_name": genes}, index=genes),
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mini.h5ad"
            adata.write_h5ad(p)
            ds = load_causalbench(p, min_cells_per_perturbation=100, perturbed_genes_only=True)
        # only GA, GB pass the >100-cell floor and are measured -> kept as perturbed cols
        self.assertEqual(set(ds.perturbed_genes), {"GA", "GB"})
        self.assertEqual(ds.metadata["n_control_cells"], 150)
        self.assertTrue(set(ds.candidate_edges["source"]) <= {"GA", "GB"})
        self.assertGreater(ds.metadata["n_cells"], 0)


class ReplogleRawLoaderTest(unittest.TestCase):
    def test_chunked_raw_loader_on_synthetic_dense_h5ad(self):
        try:
            import anndata as ad
        except Exception:  # pragma: no cover
            self.skipTest("anndata not installed")
        rng = np.random.default_rng(1)
        genes = ["GA", "GB", "GC", "GD"]
        obs_gene, blocks = [], []
        for label, n in [("non-targeting", 40), ("GA", 80), ("GB", 80)]:
            obs_gene += [label] * n
            blocks.append(rng.poisson(4, size=(n, len(genes))).astype("float32"))
        X = np.vstack(blocks)
        obs = pd.DataFrame({"gene": obs_gene, "UMI_count": X.sum(axis=1)})
        var = pd.DataFrame({"gene_name": genes}, index=genes)
        adata = ad.AnnData(X=X, obs=obs, var=var)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.h5ad"
            adata.write_h5ad(p)
            # chunk smaller than n to exercise the chunked-read path
            ds = load_replogle_raw_h5ad(p, min_cells=50, chunk=37)
        self.assertEqual(set(ds.perturbed_genes), {"GA", "GB"})       # >50 cells & measured
        self.assertEqual(list(ds.expression.columns), ["GA", "GB"])    # perturbed&measured block
        self.assertEqual(ds.metadata["n_control_cells"], 40)
        self.assertEqual(ds.metadata["n_cells"], 200)                  # 40 + 80 + 80
        self.assertTrue(np.isfinite(ds.expression.to_numpy()).all())   # normalized + log1p


class ResponseGeometryTest(unittest.TestCase):
    def _dataset(self):
        expr, perturb, true_edges = make_synthetic_interventional(
            n_genes=6, n_cells_per_condition=100, edge_density=0.4, seed=3
        )
        return load_interventional_frames("d", expr, perturb, reference_edges=true_edges)

    def test_response_matrix_self_knockdown_is_negative(self):
        ds = self._dataset()
        D = perturbation_response_matrix(ds)
        self.assertEqual(D.shape, (len(ds.perturbed_genes), len(ds.genes)))
        # the synthetic fixture forces the perturbed gene low -> its own response is negative
        diag = [D.loc[g, g] for g in ds.perturbed_genes]
        self.assertLess(float(np.mean(diag)), 0.0)

    def test_split_half_matrices_returned(self):
        ds = self._dataset()
        D, da, db = perturbation_response_matrix(ds, split_half=True, seed=0)
        self.assertEqual(da.shape, D.shape)
        stab = split_half_stability(da, db)
        self.assertEqual(len(stab), len(ds.perturbed_genes))
        self.assertTrue((stab.dropna() <= 1.0001).all())

    def test_response_low_rank_on_rank1(self):
        u = np.arange(1, 8, dtype=float)[:, None]
        v = np.array([[1.0, -2.0, 0.5, 3.0]])
        D = pd.DataFrame(u @ v)
        res = response_low_rank(D, var_cutoff=0.9)
        self.assertEqual(res["rank_at_cutoff"], 1)
        self.assertGreater(res["top1_var"], 0.999)

    def test_direct_effect_filter_removes_rank1(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=(10, 1)); v = rng.normal(size=(1, 5))
        D = pd.DataFrame(u @ v)
        direct, broad = direct_effect_filter(D, n_modes=1)
        self.assertLess(float(np.abs(direct.to_numpy()).max()), 1e-8)

    def test_response_sparsity_bounds(self):
        D = pd.DataFrame([[5.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        eff = response_sparsity(D)
        self.assertAlmostEqual(eff.iloc[0], 1.0, places=5)   # one responder
        self.assertAlmostEqual(eff.iloc[1], 4.0, places=5)   # fully diffuse

    def test_split_half_stability_identical_and_orthogonal(self):
        a = pd.DataFrame([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(split_half_stability(a, a).iloc[0], 1.0, places=6)
        b = pd.DataFrame([[0.0, 0.0, 1.0]]); c = pd.DataFrame([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(split_half_stability(b, c).iloc[0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
