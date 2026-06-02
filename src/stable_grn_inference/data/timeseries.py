"""Time-series utilities for lagged DREAM4-style inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def split_trajectories_by_time_reset(
    timeseries: pd.DataFrame,
    *,
    time_column: str = "Time",
) -> list[pd.DataFrame]:
    """Split a time-series table into trajectories when time resets.

    Parameters
    ----------
    timeseries:
        Data frame containing a time column and gene-expression columns.
    time_column:
        Name of the time column used to detect trajectory boundaries.

    Returns
    -------
    list[pandas.DataFrame]
        One data frame per trajectory, preserving row order.
    """
    if time_column not in timeseries.columns:
        raise ValueError(f"missing time column: {time_column}")
    if timeseries.empty:
        return []

    table = timeseries.copy()
    table[time_column] = pd.to_numeric(table[time_column])
    reset_points = [0]
    previous_time = float(table.iloc[0][time_column])
    for row_index in range(1, len(table)):
        current_time = float(table.iloc[row_index][time_column])
        if current_time <= previous_time:
            reset_points.append(row_index)
        previous_time = current_time

    trajectories: list[pd.DataFrame] = []
    for start, stop in zip(reset_points, [*reset_points[1:], len(table)]):
        trajectory = table.iloc[start:stop].reset_index(drop=True)
        if not trajectory.empty:
            trajectories.append(trajectory)
    return trajectories


def build_lagged_samples(
    trajectories: Sequence[pd.DataFrame],
    *,
    time_column: str = "Time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build adjacent-time lagged samples within each trajectory.

    For each adjacent pair in each trajectory, predictors are gene expression
    at time ``t`` and targets are gene expression at time ``t + 1``. No lagged
    pair is created across trajectory boundaries.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``X_t`` predictors, ``Y_t1`` targets, and metadata with trajectory id
        plus the source and next time values.
    """
    x_rows: list[pd.Series] = []
    y_rows: list[pd.Series] = []
    meta_rows: list[dict[str, float | int]] = []

    for trajectory_id, trajectory in enumerate(trajectories, start=1):
        if time_column not in trajectory.columns:
            raise ValueError(f"missing time column: {time_column}")
        gene_columns = [column for column in trajectory.columns if column != time_column]
        for row_index in range(len(trajectory) - 1):
            current_row = trajectory.iloc[row_index]
            next_row = trajectory.iloc[row_index + 1]
            x_rows.append(current_row[gene_columns])
            y_rows.append(next_row[gene_columns])
            meta_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "time_t": float(current_row[time_column]),
                    "time_t1": float(next_row[time_column]),
                }
            )

    if not x_rows:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(columns=["trajectory_id", "time_t", "time_t1"]),
        )

    x = pd.DataFrame(x_rows).reset_index(drop=True).apply(pd.to_numeric)
    y = pd.DataFrame(y_rows).reset_index(drop=True).apply(pd.to_numeric)
    metadata = pd.DataFrame(meta_rows)
    return x, y, metadata


def build_dynamic_target(
    x_t: pd.DataFrame,
    y_t1: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    target_type: str,
) -> pd.DataFrame:
    """Return level, delta, or derivative targets for lagged inference.

    ``level`` predicts expression at ``t+1``. ``delta`` predicts
    ``expression(t+1) - expression(t)``. ``derivative`` divides the delta by
    ``time_t1 - time_t`` for each lagged sample.
    """
    if list(x_t.columns) != list(y_t1.columns):
        raise ValueError("x_t and y_t1 must have matching gene columns")
    if len(x_t) != len(y_t1) or len(x_t) != len(metadata):
        raise ValueError("x_t, y_t1, and metadata must have matching rows")

    if target_type == "level":
        return y_t1.copy()
    if target_type == "delta":
        return y_t1.subtract(x_t)
    if target_type == "derivative":
        if "time_t" not in metadata.columns or "time_t1" not in metadata.columns:
            raise ValueError("metadata must include time_t and time_t1 for derivative targets")
        delta_time = metadata["time_t1"].to_numpy(dtype=float) - metadata["time_t"].to_numpy(dtype=float)
        if np.any(delta_time <= 0):
            raise ValueError("all derivative target time steps must be positive")
        return y_t1.subtract(x_t).div(delta_time, axis=0)
    raise ValueError("target_type must be 'level', 'delta', or 'derivative'")


def residualize_target_on_self(
    x_t: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Remove each gene's autoregressive self-persistence from its target.

    For every gene ``j``, this regresses the target column ``target[j]`` on the
    same gene's predictor column ``x_t[j]`` with an intercept (ordinary least
    squares, closed form) and returns the residual. The residual target keeps
    only the variation in ``G_j(t+1)`` (or its delta) that is not explained by
    ``G_j(t)`` alone, so a downstream exclude-self model fit on these residuals
    scores non-self regulators after self-persistence has been controlled for.

    Constant predictor columns contribute no slope, so the residual is simply
    the mean-centered target for those genes.
    """
    if list(x_t.columns) != list(target.columns):
        raise ValueError("x_t and target must have matching gene columns")
    if len(x_t) != len(target):
        raise ValueError("x_t and target must have matching rows")

    x = x_t.apply(pd.to_numeric)
    y = target.apply(pd.to_numeric)
    residuals = {}
    for gene in x.columns:
        predictor = x[gene].to_numpy(dtype=float)
        response = y[gene].to_numpy(dtype=float)
        predictor_centered = predictor - predictor.mean()
        response_centered = response - response.mean()
        denominator = float(np.dot(predictor_centered, predictor_centered))
        if denominator == 0.0:
            residuals[gene] = response - response.mean()
            continue
        slope = float(np.dot(predictor_centered, response_centered)) / denominator
        intercept = response.mean() - slope * predictor.mean()
        residuals[gene] = response - (intercept + slope * predictor)
    return pd.DataFrame(residuals, index=target.index)


def trajectory_bootstrap_indices(
    metadata: pd.DataFrame,
    n_resamples: int,
    *,
    random_seed: int = 0,
    trajectory_column: str = "trajectory_id",
) -> list[np.ndarray]:
    """Generate reproducible lagged-row indices by bootstrapping trajectories."""
    if trajectory_column not in metadata.columns:
        raise ValueError(f"missing trajectory column: {trajectory_column}")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    trajectory_ids = sorted(metadata[trajectory_column].unique())
    if not trajectory_ids:
        raise ValueError("metadata must contain at least one trajectory")

    rows_by_trajectory = {
        trajectory_id: metadata.index[metadata[trajectory_column] == trajectory_id].to_numpy()
        for trajectory_id in trajectory_ids
    }
    rng = np.random.default_rng(random_seed)
    resamples: list[np.ndarray] = []
    for _ in range(n_resamples):
        sampled_trajectories = rng.choice(trajectory_ids, size=len(trajectory_ids), replace=True)
        resamples.append(np.concatenate([rows_by_trajectory[trajectory] for trajectory in sampled_trajectories]))
    return resamples


def moving_average_smooth_trajectories(
    trajectories: Sequence[pd.DataFrame],
    *,
    window: int = 3,
    time_column: str = "Time",
) -> list[pd.DataFrame]:
    """Apply centered moving-average smoothing to each gene in each trajectory."""
    if window <= 0:
        raise ValueError("window must be positive")

    smoothed: list[pd.DataFrame] = []
    for trajectory in trajectories:
        if time_column not in trajectory.columns:
            raise ValueError(f"missing time column: {time_column}")
        result = trajectory.copy()
        gene_columns = [column for column in trajectory.columns if column != time_column]
        result[gene_columns] = (
            trajectory[gene_columns]
            .apply(pd.to_numeric)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        smoothed.append(result)
    return smoothed
