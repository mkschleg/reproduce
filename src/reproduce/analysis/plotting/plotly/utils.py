"""
Utility functions for Plotly plotting.
"""

from typing import Any, Optional, Iterable
import pandas as pd
import numpy as np


_PLOTLY_DASH_MAP = {
    (1, 0): "solid",
    (4, 4): "dash",
    (2, 2): "dot",
    (8, 4): "longdash",
    (4, 2, 2, 2): "dashdot",
}
_DEFAULT_PLOTLY_DASHES = ["solid", "dash", "dot", "longdash", "dashdot"]


def normalize_dash(value: Any) -> Optional[str]:
    """Normalize dash pattern value to Plotly dash string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        try:
            key = tuple(value)
            return _PLOTLY_DASH_MAP.get(key, "dash")
        except TypeError:
            return None
    return None


def hex_to_rgba(color: str, opacity: float) -> str:
    """Convert hex color to rgba format with opacity."""
    if not isinstance(color, str):
        return color
    if color.startswith("#") and len(color) == 7:
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            return f"rgba({r},{g},{b},{opacity})"
        except ValueError:
            return color
    return color


def log_transform(values: np.ndarray) -> np.ndarray:
    """Apply log10 transform preserving sign."""
    values = values.astype(float)
    sign = np.sign(values)
    abs_vals = np.abs(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.log10(abs_vals)
    log_vals = np.where(abs_vals > 0, log_vals, 0.0)
    return sign * log_vals


def format_tick_value(value: float) -> str:
    """Format tick value for display."""
    abs_val = abs(value)
    if abs_val >= 1:
        return f"{value:.0f}"
    return f"{value:.2g}"


def metric_columns_contain_lists(df: pd.DataFrame) -> bool:
    """Check if metric columns contain list values."""
    metric_cols = [
        c for c in df.columns if "returns" in c or ":" in c or c.startswith("metric_")
    ]
    for col in metric_cols:
        if col not in df.columns:
            continue
        series = df[col]
        sample = series.dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return True
    return False
