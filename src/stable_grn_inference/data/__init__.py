"""Data loading helpers for DREAM4/GeneNetWeaver-style experiments."""

from .dream4 import (
    SIZE10_DATA_REGIMES,
    SIZE100_DATA_REGIMES,
    dream4_size10_expression_path,
    dream4_size10_gold_standard_path,
    dream4_size100_expression_path,
    dream4_size100_gold_standard_path,
    load_expression_matrix,
    load_gold_standard_edges,
)
from .timeseries import (
    build_dynamic_target,
    build_lagged_samples,
    moving_average_smooth_trajectories,
    split_trajectories_by_time_reset,
    trajectory_bootstrap_indices,
)

__all__ = [
    "SIZE10_DATA_REGIMES",
    "SIZE100_DATA_REGIMES",
    "build_dynamic_target",
    "build_lagged_samples",
    "dream4_size10_expression_path",
    "dream4_size10_gold_standard_path",
    "dream4_size100_expression_path",
    "dream4_size100_gold_standard_path",
    "load_expression_matrix",
    "load_gold_standard_edges",
    "moving_average_smooth_trajectories",
    "split_trajectories_by_time_reset",
    "trajectory_bootstrap_indices",
]
