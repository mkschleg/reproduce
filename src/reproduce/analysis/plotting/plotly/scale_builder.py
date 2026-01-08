"""
Scale mapping builder for Plotly plots.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from ..altair_plotter import ScaleRegistry
from .utils import normalize_dash, _DEFAULT_PLOTLY_DASHES
from .encoding_parser import ParsedEncoding


@dataclass
class ScaleMaps:
    """Computed scale mappings."""

    color_map: Dict[Any, str]
    dash_map: Dict[Any, str]


class PlotlyScaleBuilder:
    """Build Plotly-specific scale mappings."""

    def __init__(self, scales: ScaleRegistry, labels: Dict[str, str]):
        self.scales = scales
        self.labels = labels

    def build_scales(
        self, df: pd.DataFrame, encoding: ParsedEncoding
    ) -> ScaleMaps:
        """Build all scale mappings for the plot."""
        # Get scale specs
        color_scale_spec = self._get_scale_spec(encoding.color_spec)
        dash_scale_spec = self._get_scale_spec(encoding.dash_spec)

        # Apply fallback for agent_dash
        if encoding.dash_field and not dash_scale_spec:
            if encoding.dash_field == "combo_agent.type":
                dash_scale_spec = self.scales.get_scale_spec("agent_dash")

        # Build maps
        color_map = self._build_color_map(df, encoding.color_field, color_scale_spec)
        dash_map = self._build_dash_map(df, encoding.dash_field, dash_scale_spec)

        return ScaleMaps(color_map=color_map, dash_map=dash_map)

    def _get_scale_spec(self, channel_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get scale spec from channel spec."""
        if not isinstance(channel_spec, dict):
            return None

        scale = channel_spec.get("scale")
        if isinstance(scale, str):
            return self.scales.get_scale_spec(scale)
        if isinstance(scale, dict):
            return scale
        return None

    def _build_color_map(
        self,
        df: pd.DataFrame,
        field: Optional[str],
        scale_spec: Optional[Dict[str, Any]],
    ) -> Dict[Any, str]:
        """Build color mapping."""
        if not field or not scale_spec:
            return {}

        domain = scale_spec.get("domain")
        range_vals = scale_spec.get("range")

        if not domain:
            domain = list(pd.unique(df[field]))
        if not range_vals:
            return {}

        return {k: range_vals[idx % len(range_vals)] for idx, k in enumerate(domain)}

    def _build_dash_map(
        self,
        df: pd.DataFrame,
        field: Optional[str],
        scale_spec: Optional[Dict[str, Any]],
    ) -> Dict[Any, str]:
        """Build dash mapping."""
        if not field:
            return {}

        if scale_spec:
            dash_map_raw = self._build_scale_map(df, field, scale_spec)
        else:
            # Use default dashes
            values = list(pd.unique(df[field]))
            dash_map_raw = {
                val: _DEFAULT_PLOTLY_DASHES[idx % len(_DEFAULT_PLOTLY_DASHES)]
                for idx, val in enumerate(values)
            }

        # Normalize dash values
        return {k: normalize_dash(v) for k, v in dash_map_raw.items()}

    def _build_scale_map(
        self, df: pd.DataFrame, field: str, scale_spec: Dict[str, Any]
    ) -> Dict[Any, Any]:
        """Generic scale map builder."""
        domain = scale_spec.get("domain")
        range_vals = scale_spec.get("range")

        if not domain:
            domain = list(pd.unique(df[field]))
        if not range_vals:
            return {}

        return {k: range_vals[idx % len(range_vals)] for idx, k in enumerate(domain)}
