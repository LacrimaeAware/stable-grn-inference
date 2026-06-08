"""Dynamical-structure recovery under a dominant shared mode.

This subpackage studies the problem every RPE1 experiment (21-27) kept circling
and the structured-transform project states in representation space: separating a
dominant shared mode (cell-cycle program / class identity) from small, specific,
reusable structure (gene-specific edges / transformation factors), and knowing
*when* that separation is identifiable.

`separability` provides a controlled, ground-truthed generator and a recovery
harness so the boundary of recoverability can be mapped as a phase diagram, onto
which real datasets (e.g. RPE1, dominant-mode fraction ~0.53) can be placed.
"""

from .separability import (
    SeparableSystem,
    make_separable_system,
    recover_specific,
    RECOVERY_METHODS,
    specific_recovery_aupr,
    normalized_recovery,
    separability_grid,
    recoverability_boundary,
)
from .temporal import (
    DYNAMICAL_METHODS,
    DynamicalSystem,
    dmd_edges,
    dmd_operator,
    dynamical_recovery_grid,
    edges_to_operator,
    make_dynamical_system,
    pseudotime_ordered_pairs,
    skeleton_recovery_aupr,
    static_correlation_edges,
)

__all__ = [
    "SeparableSystem",
    "make_separable_system",
    "recover_specific",
    "RECOVERY_METHODS",
    "specific_recovery_aupr",
    "normalized_recovery",
    "separability_grid",
    "recoverability_boundary",
    "DYNAMICAL_METHODS",
    "DynamicalSystem",
    "dmd_edges",
    "dmd_operator",
    "dynamical_recovery_grid",
    "edges_to_operator",
    "make_dynamical_system",
    "pseudotime_ordered_pairs",
    "skeleton_recovery_aupr",
    "static_correlation_edges",
]
