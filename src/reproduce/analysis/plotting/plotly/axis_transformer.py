"""
Axis transformation logic for Plotly plots.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from .utils import log_transform, format_tick_value


@dataclass
class TransformedAxis:
    """Result of axis transformation."""

    df: pd.DataFrame
    y_field: str
    tickvals: Optional[List[float]]
    ticktext: Optional[List[str]]
    y_axis_settings: Dict[str, Any]


class AxisTransformer:
    """Handle axis transformations."""

    def transform_y_axis(
        self, df: pd.DataFrame, y_spec: Dict[str, Any], y_field: str
    ) -> TransformedAxis:
        """Apply log transform if specified."""
        if not isinstance(y_spec, dict):
            return TransformedAxis(
                df=df,
                y_field=y_field,
                tickvals=None,
                ticktext=None,
                y_axis_settings={},
            )

        scale_spec = y_spec.get("scale", {})
        scale_type = scale_spec.get("type")

        # Check if we need log transform
        if scale_type not in {"log", "symlog"} or y_field not in df.columns:
            return TransformedAxis(
                df=df,
                y_field=y_field,
                tickvals=None,
                ticktext=None,
                y_axis_settings={},
            )

        # Apply log transform
        df = df.copy()
        y_trans_field = f"{y_field}__log"
        df[y_trans_field] = log_transform(df[y_field].to_numpy())

        # Transform CI bounds if present
        if "ci_lower" in df.columns and "ci_upper" in df.columns:
            df["ci_lower"] = log_transform(df["ci_lower"].to_numpy())
            df["ci_upper"] = log_transform(df["ci_upper"].to_numpy())

        # Compute tick values
        tickvals, ticktext = self._compute_log_ticks(df, y_field, scale_spec)

        # Build axis settings
        y_axis_settings = {"type": "linear"}
        if "domain" in scale_spec:
            domain_vals = np.array(scale_spec["domain"], dtype=float)
            y_axis_settings["range"] = log_transform(domain_vals).tolist()

        if tickvals and ticktext:
            y_axis_settings["tickvals"] = tickvals
            y_axis_settings["ticktext"] = ticktext

        return TransformedAxis(
            df=df,
            y_field=y_trans_field,
            tickvals=tickvals,
            ticktext=ticktext,
            y_axis_settings=y_axis_settings,
        )

    def _compute_log_ticks(
        self, df: pd.DataFrame, y_field: str, scale_spec: Dict[str, Any]
    ) -> Tuple[Optional[List[float]], Optional[List[str]]]:
        """Compute tick values for log scale."""
        # Get domain
        domain = scale_spec.get("domain")
        if domain:
            domain_vals = np.array(domain, dtype=float)
        elif y_field in df.columns:
            domain_vals = df[y_field].to_numpy()
        else:
            return None, None

        # Filter valid values
        domain_vals = domain_vals[np.isfinite(domain_vals)]
        if domain_vals.size == 0:
            return None, None

        # Compute tick positions
        min_val = np.min(domain_vals)
        max_val = np.max(domain_vals)
        min_abs = np.min(np.abs(domain_vals[np.abs(domain_vals) > 0]))
        max_abs = np.max(np.abs(domain_vals))

        min_exp = int(np.floor(np.log10(min_abs)))
        max_exp = int(np.ceil(np.log10(max_abs)))

        tick_values = []
        if max_val <= 0:
            for exp in range(min_exp, max_exp + 1):
                tick_values.append(-10**exp)
        elif min_val >= 0:
            for exp in range(min_exp, max_exp + 1):
                tick_values.append(10**exp)
        else:
            for exp in range(min_exp, max_exp + 1):
                tick_values.append(-10**exp)
                tick_values.append(10**exp)

        tickvals = log_transform(np.array(tick_values, dtype=float)).tolist()
        ticktext = [format_tick_value(val) for val in tick_values]

        return tickvals, ticktext
