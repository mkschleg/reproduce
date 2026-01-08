"""
Data transformation utilities for converting experiment data to Altair-ready tidy format.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union
import numpy as np
import pandas as pd


@dataclass
class TransformSpec:
    """Specification for data transformation from YAML."""
    metric: str
    agg_func: str = "median"  # "mean", "median"
    ci_type: str = "quantile"  # "quantile", "sem", "std"
    ci_lower: float = 0.25
    ci_upper: float = 0.75
    group_by: List[str] = field(default_factory=list)
    # For individual_returns transformation
    individual_returns: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TransformSpec":
        """Create TransformSpec from YAML dict."""
        return cls(
            metric=d.get("metric", "returns:avg_end"),
            agg_func=d.get("agg_func", "median"),
            ci_type=d.get("ci_type", "quantile"),
            ci_lower=d.get("ci_lower", 0.25),
            ci_upper=d.get("ci_upper", 0.75),
            group_by=d.get("group_by", []),
            individual_returns=d.get("individual_returns"),
        )


def get_agg_function(name: str) -> Callable:
    """Get aggregation function by name."""
    funcs = {
        "mean": np.mean,
        "median": np.median,
    }
    return funcs.get(name, np.median)


def compute_ci(
    values: List[float],
    ci_type: str,
    ci_lower: float = 0.25,
    ci_upper: float = 0.75,
    center_value: Optional[float] = None
) -> tuple:
    """
    Compute confidence interval bounds.

    Args:
        values: List of raw values
        ci_type: Type of CI ("quantile", "sem", "std")
        ci_lower: Lower quantile (for quantile type)
        ci_upper: Upper quantile (for quantile type)
        center_value: Aggregated value to center sem/std around (if None, uses mean)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    arr = np.array(values)

    if ci_type == "quantile":
        return (np.quantile(arr, ci_lower), np.quantile(arr, ci_upper))
    elif ci_type == "sem":
        sem = np.std(arr) / np.sqrt(len(arr))
        center = center_value if center_value is not None else np.mean(arr)
        return (center - sem, center + sem)
    elif ci_type == "std":
        std = np.std(arr)
        center = center_value if center_value is not None else np.mean(arr)
        return (center - std, center + std)
    else:
        # Default to quantile
        return (np.quantile(arr, 0.25), np.quantile(arr, 0.75))


def aggregate_with_ci(
    df: pd.DataFrame,
    metric: str,
    group_cols: List[str],
    agg_func: Union[str, Callable] = "median",
    ci_type: str = "quantile",
    ci_lower: float = 0.25,
    ci_upper: float = 0.75
) -> pd.DataFrame:
    """
    Aggregate metric by group and compute confidence intervals.

    Expects the metric column to contain lists of values (one per seed).

    Args:
        df: DataFrame with grouped experiment data
        metric: Name of metric column containing list values
        group_cols: Columns to keep in output (hyperparameters)
        agg_func: Aggregation function name or callable
        ci_type: Type of CI ("quantile", "sem", "std")
        ci_lower: Lower quantile bound
        ci_upper: Upper quantile bound

    Returns:
        DataFrame with columns: group_cols + [value, ci_lower, ci_upper]
    """
    if isinstance(agg_func, str):
        agg_func = get_agg_function(agg_func)

    # Ensure we have the metric column
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")

    # Filter to only requested columns
    keep_cols = [c for c in group_cols if c in df.columns]

    # Build output data
    rows = []
    for _, row in df.iterrows():
        raw_values = row[metric]

        # Handle case where values might be tuple or list
        if isinstance(raw_values, (list, tuple, np.ndarray)):
            values = list(raw_values)
        else:
            values = [raw_values]

        # Skip if no values
        if len(values) == 0:
            continue

        # Compute aggregated value and CI
        agg_value = float(agg_func(values))
        ci_low, ci_high = compute_ci(values, ci_type, ci_lower, ci_upper, center_value=agg_value)

        # Build row
        out_row = {col: row[col] for col in keep_cols}
        out_row["value"] = agg_value
        out_row["ci_lower"] = float(ci_low)
        out_row["ci_upper"] = float(ci_high)
        rows.append(out_row)

    return pd.DataFrame(rows)


def explode_metric_lists(
    df: pd.DataFrame,
    metric: str,
    group_cols: List[str]
) -> pd.DataFrame:
    """
    Explode list-valued metric column into tidy format.

    Input: Each row has metric as a list [v1, v2, v3, ...] (multiple seeds)
    Output: One row per seed value, with 'seed_idx' column

    Useful for scatter plots or jitter plots showing raw data.

    Args:
        df: DataFrame with grouped experiment data
        metric: Name of metric column containing list values
        group_cols: Columns to keep in output

    Returns:
        Tidy DataFrame with one row per observation
    """
    keep_cols = [c for c in group_cols if c in df.columns]

    rows = []
    for _, row in df.iterrows():
        raw_values = row[metric]

        if isinstance(raw_values, (list, tuple, np.ndarray)):
            values = list(raw_values)
        else:
            values = [raw_values]

        for idx, val in enumerate(values):
            out_row = {col: row[col] for col in keep_cols}
            out_row["value"] = val
            out_row["seed_idx"] = idx
            rows.append(out_row)

    return pd.DataFrame(rows)


def apply_transform_spec(
    df: pd.DataFrame,
    spec: TransformSpec
) -> pd.DataFrame:
    """
    Apply TransformSpec to DataFrame, returning Altair-ready tidy format.

    Args:
        df: DataFrame with grouped experiment data (metrics as lists)
        spec: TransformSpec defining the transformation

    Returns:
        DataFrame with columns: group_by cols + [value, ci_lower, ci_upper]
    """
    # Handle individual_returns transformation
    if spec.individual_returns:
        num_agents = spec.individual_returns.get("num_agents", 2)
        metric_prefix = spec.individual_returns.get("metric_prefix", spec.metric)

        # Prepare data for each agent
        all_agent_data = []
        for agent_idx in range(num_agents):
            agent_df = prepare_individual_returns(
                df=df,
                metric_prefix=metric_prefix,
                agent_idx=agent_idx,
                group_cols=spec.group_by,
                agg_func=spec.agg_func,
                ci_type=spec.ci_type,
                ci_lower=spec.ci_lower,
                ci_upper=spec.ci_upper,
            )
            all_agent_data.append(agent_df)

        return pd.concat(all_agent_data, ignore_index=True)

    # Standard aggregation
    return aggregate_with_ci(
        df=df,
        metric=spec.metric,
        group_cols=spec.group_by,
        agg_func=spec.agg_func,
        ci_type=spec.ci_type,
        ci_lower=spec.ci_lower,
        ci_upper=spec.ci_upper,
    )


def prepare_individual_returns(
    df: pd.DataFrame,
    metric_prefix: str,
    agent_idx: int,
    group_cols: List[str],
    agg_func: Union[str, Callable] = "median",
    ci_type: str = "quantile",
    ci_lower: float = 0.25,
    ci_upper: float = 0.75
) -> pd.DataFrame:
    """
    Prepare individual agent returns for plotting.

    The metric column contains lists of tuples, where each tuple has
    one value per agent. This extracts and aggregates values for a
    specific agent.

    Args:
        df: DataFrame with grouped experiment data
        metric_prefix: Column containing per-agent returns
        agent_idx: Which agent's returns to extract
        group_cols: Columns to keep in output
        agg_func: Aggregation function
        ci_type: CI type
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound

    Returns:
        DataFrame with agent's aggregated returns and CI
    """
    if isinstance(agg_func, str):
        agg_func = get_agg_function(agg_func)

    keep_cols = [c for c in group_cols if c in df.columns]

    rows = []
    for _, row in df.iterrows():
        raw_values = row[metric_prefix]

        # Extract agent_idx values from each tuple
        try:
            agent_values = [v[agent_idx] for v in raw_values]
        except (IndexError, TypeError):
            continue

        if len(agent_values) == 0:
            continue

        agg_value = float(agg_func(agent_values))
        ci_low, ci_high = compute_ci(agent_values, ci_type, ci_lower, ci_upper, center_value=agg_value)

        out_row = {col: row[col] for col in keep_cols}
        out_row["value"] = agg_value
        out_row["ci_lower"] = float(ci_low)
        out_row["ci_upper"] = float(ci_high)
        out_row["agent_idx"] = agent_idx
        rows.append(out_row)

    return pd.DataFrame(rows)
