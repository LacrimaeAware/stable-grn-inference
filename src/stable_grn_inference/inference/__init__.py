"""Baseline sparse inference routines."""

from .baselines import (
    rank_edges_by_correlation,
    rank_edges_by_elastic_net,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
)
from .dynamic import (
    build_dynamic_sparse_linear_grid,
    fit_dynamic_linear_coefficients,
    rank_edges_by_dynamic_correlation,
    rank_edges_by_dynamic_elastic_net,
    rank_edges_by_dynamic_lasso,
    rank_edges_by_dynamic_mlp_permutation,
    rank_edges_by_dynamic_tree_ensemble,
    rank_fusion,
    summarize_resampled_dynamic_linear_coefficients,
    summarize_resampled_dynamic_scores,
)
from .genie3 import (
    rank_edges_by_genie3,
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
)
from .lagged import (
    rank_edges_by_lagged_correlation,
    rank_edges_by_lagged_extra_trees,
    rank_edges_by_lagged_lasso,
    rank_edges_by_lagged_random_forest,
    rank_edges_by_lagged_tree_ensemble,
)

__all__ = [
    "rank_edges_by_genie3",
    "rank_edges_by_genie3_extra_trees",
    "rank_edges_by_genie3_random_forest",
    "rank_edges_by_lagged_correlation",
    "rank_edges_by_lagged_extra_trees",
    "rank_edges_by_lagged_lasso",
    "rank_edges_by_lagged_random_forest",
    "rank_edges_by_lagged_tree_ensemble",
    "build_dynamic_sparse_linear_grid",
    "fit_dynamic_linear_coefficients",
    "rank_edges_by_correlation",
    "rank_edges_by_dynamic_correlation",
    "rank_edges_by_dynamic_elastic_net",
    "rank_edges_by_dynamic_lasso",
    "rank_edges_by_dynamic_mlp_permutation",
    "rank_edges_by_dynamic_tree_ensemble",
    "rank_edges_by_elastic_net",
    "rank_edges_by_lasso",
    "rank_edges_by_random_forest",
    "rank_fusion",
    "summarize_resampled_dynamic_linear_coefficients",
    "summarize_resampled_dynamic_scores",
]
