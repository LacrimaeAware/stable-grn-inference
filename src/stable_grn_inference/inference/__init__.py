"""Baseline sparse inference routines."""

from .baselines import (
    rank_edges_by_correlation,
    rank_edges_by_elastic_net,
    rank_edges_by_lasso,
    rank_edges_by_random_forest,
)
from .genie3 import (
    rank_edges_by_genie3,
    rank_edges_by_genie3_extra_trees,
    rank_edges_by_genie3_random_forest,
)

__all__ = [
    "rank_edges_by_genie3",
    "rank_edges_by_genie3_extra_trees",
    "rank_edges_by_genie3_random_forest",
    "rank_edges_by_correlation",
    "rank_edges_by_elastic_net",
    "rank_edges_by_lasso",
    "rank_edges_by_random_forest",
]
