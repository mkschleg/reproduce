"""
Trace builder for Plotly plots.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .scale_builder import ScaleMaps
from .encoding_parser import ParsedEncoding
from .utils import hex_to_rgba


class TraceBuilder:
    """Build Plotly traces for different mark types."""

    def __init__(self, scale_maps: ScaleMaps, labels: Dict[str, str]):
        self.scale_maps = scale_maps
        self.labels = labels

    def build_traces_for_facet(
        self,
        facet_df: pd.DataFrame,
        encoding: ParsedEncoding,
        mark_spec: Dict[str, any],
        error_band_spec: Optional[Dict[str, any]],
        hovertemplate: Optional[str],
        hover_fields: Optional[List[str]],
        show_legend: bool,
    ) -> List[Union[go.Scatter, go.Bar]]:
        """Build all traces for a facet."""
        traces = []

        # Group data
        if encoding.group_fields:
            grouped = facet_df.groupby(encoding.group_fields, dropna=False)
        else:
            grouped = [((), facet_df)]

        # Build trace for each group
        for group_values, group_df in grouped:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            # Build group label
            group_label = self._build_group_label(
                encoding.group_fields, group_values, encoding.color_field
            )

            # Sort by x
            if encoding.x_field:
                group_df = group_df.sort_values(encoding.x_field)

            # Get styling
            line_color = self._get_color(group_df, encoding.color_field)
            line_dash = self._get_dash(group_df, encoding.dash_field)

            # Build error band traces if specified
            if (
                error_band_spec
                and "ci_lower" in group_df
                and "ci_upper" in group_df
            ):
                band_traces = self._build_error_band(
                    group_df, encoding.x_field, line_color, error_band_spec
                )
                traces.extend(band_traces)

            # Build main trace
            mark_type = mark_spec.get("type", "line")
            show_points = bool(mark_spec.get("point", False))

            customdata = None
            if hover_fields:
                customdata = group_df[hover_fields].to_numpy()

            main_trace = self._build_trace(
                group_df=group_df,
                x_field=encoding.x_field,
                y_field=encoding.y_field,
                mark_type=mark_type,
                show_points=show_points,
                group_label=group_label,
                line_color=line_color,
                line_dash=line_dash,
                show_legend=show_legend,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
            traces.append(main_trace)

        return traces

    def _build_group_label(
        self,
        group_fields: List[str],
        group_values: Tuple,
        color_field: Optional[str],
    ) -> str:
        """Build label for a group."""
        parts = []
        for field, value in zip(group_fields, group_values):
            if field == color_field and self.labels.get(field) == "Agent Type":
                parts.append(str(value))
            else:
                parts.append(f"{self.labels.get(field, field)}: {value}")
        return ", ".join(parts) if parts else ""

    def _get_color(
        self, group_df: pd.DataFrame, color_field: Optional[str]
    ) -> Optional[str]:
        """Get color for group."""
        if not color_field:
            return None
        color_value = group_df[color_field].iloc[0]
        return self.scale_maps.color_map.get(color_value)

    def _get_dash(
        self, group_df: pd.DataFrame, dash_field: Optional[str]
    ) -> Optional[str]:
        """Get dash pattern for group."""
        if not dash_field:
            return None
        dash_value = group_df[dash_field].iloc[0]
        return self.scale_maps.dash_map.get(dash_value)

    def _build_error_band(
        self,
        group_df: pd.DataFrame,
        x_field: str,
        line_color: Optional[str],
        error_band_spec: Dict[str, any],
    ) -> List[go.Scatter]:
        """Build error band traces."""
        opacity = error_band_spec.get("opacity", 0.2)
        band_color = (
            hex_to_rgba(line_color, opacity)
            if line_color
            else f"rgba(0,0,0,{opacity})"
        )

        lower_trace = go.Scatter(
            x=group_df[x_field],
            y=group_df["ci_lower"],
            mode="lines",
            line=dict(color=band_color),
            hoverinfo="skip",
            showlegend=False,
        )

        upper_trace = go.Scatter(
            x=group_df[x_field],
            y=group_df["ci_upper"],
            mode="lines",
            line=dict(color=band_color),
            fill="tonexty",
            fillcolor=band_color,
            hoverinfo="skip",
            showlegend=False,
        )

        return [lower_trace, upper_trace]

    def _build_trace(
        self,
        group_df: pd.DataFrame,
        x_field: str,
        y_field: str,
        mark_type: str,
        show_points: bool,
        group_label: str,
        line_color: Optional[str],
        line_dash: Optional[str],
        show_legend: bool,
        customdata: Optional[np.ndarray],
        hovertemplate: Optional[str],
    ) -> Union[go.Scatter, go.Bar]:
        """Build single trace."""
        line_kwargs = {}
        if line_color:
            line_kwargs["color"] = line_color
        if line_dash:
            line_kwargs["dash"] = line_dash

        mode = "lines"
        if mark_type == "point":
            mode = "markers"
        elif mark_type == "line" and show_points:
            mode = "lines+markers"

        if mark_type == "bar":
            return go.Bar(
                x=group_df[x_field],
                y=group_df[y_field],
                name=group_label,
                marker_color=line_color,
                showlegend=show_legend,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        elif mark_type == "area":
            return go.Scatter(
                x=group_df[x_field],
                y=group_df[y_field],
                mode=mode,
                name=group_label,
                line=line_kwargs,
                fill="tozeroy",
                showlegend=show_legend,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        else:  # line or point
            return go.Scatter(
                x=group_df[x_field],
                y=group_df[y_field],
                mode=mode,
                name=group_label,
                line=line_kwargs,
                marker=dict(color=line_color),
                showlegend=show_legend,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
