"""
Altair-based plotter for RL experiments.

Provides declarative, Vega-Lite-style plotting as an alternative to matplotlib.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import altair as alt
import pandas as pd
import numpy as np

from ..utils import data_utils
from .data_transforms import (
    TransformSpec,
    aggregate_with_ci,
    prepare_individual_returns,
    apply_transform_spec,
)

if TYPE_CHECKING:
    from ..loading.experiment import Experiment


def _sanitize_field_name(name: str) -> str:
    """Sanitize field name for Vega-Lite (dots cause issues)."""
    return name.replace(".", "_")


def _sanitize_dataframe(df: pd.DataFrame) -> tuple:
    """
    Sanitize DataFrame column names for Vega-Lite compatibility.

    Returns:
        (sanitized_df, column_map) where column_map is {original: sanitized}
    """
    column_map = {col: _sanitize_field_name(col) for col in df.columns}
    sanitized_df = df.rename(columns=column_map)
    return sanitized_df, column_map


def _sanitize_spec_fields(spec_dict: Dict[str, Any], column_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Update field references in a spec dict to use sanitized names.
    """
    import copy
    result = copy.deepcopy(spec_dict)

    def update_field(obj):
        if isinstance(obj, dict):
            if "field" in obj and obj["field"] in column_map:
                obj["field"] = column_map[obj["field"]]
            for v in obj.values():
                update_field(v)
        elif isinstance(obj, list):
            for item in obj:
                update_field(item)

    update_field(result)
    return result


def _metric_columns_contain_lists(df: pd.DataFrame) -> bool:
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


# Default color palette (tol_colors bright equivalent)
DEFAULT_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

# Default stroke dash patterns for Altair
# Format: [dash_length, gap_length] - [1, 0] is solid
DEFAULT_DASHES = [[1, 0], [4, 4], [2, 2], [8, 4], [4, 2, 2, 2]]


class ScaleRegistry:
    """
    Altair scale definitions loaded from YAML.

    Replaces matplotlib StyleRegistry with Altair-native scales.
    """

    def __init__(self, scale_defs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize from YAML scales section.

        Args:
            scale_defs: Dictionary of {scale_name: scale_spec}
        """
        self._scales = scale_defs or {}
        self._init_default_scales()

    def _init_default_scales(self):
        """Initialize default scales for agent types if not provided."""
        if "agent_color" not in self._scales:
            self._scales["agent_color"] = {
                "domain": [
                    "Centralized",
                    "Decentralized",
                    "DecentralizedJointObs",
                    "DecentralizedSumRew",
                    "DecentralizedJointObsSumRew",
                ],
                "range": [
                    "#4477AA",  # blue
                    "#EE6677",  # red
                    "#EE6677",  # red (same as Decentralized)
                    "#228833",  # green
                    "#228833",  # green (same as DecentralizedSumRew)
                ],
            }

        if "agent_dash" not in self._scales:
            self._scales["agent_dash"] = {
                "domain": [
                    "Centralized",
                    "Decentralized",
                    "DecentralizedJointObs",
                    "DecentralizedSumRew",
                    "DecentralizedJointObsSumRew",
                ],
                "range": [
                    [1, 0],  # solid
                    [4, 4],  # dashed
                    [1, 0],  # solid
                    [4, 4],  # dashed
                    [1, 0],  # solid
                ],
            }

    def get_scale_spec(self, name: str) -> Dict[str, Any]:
        """Get raw scale specification."""
        return self._scales.get(name, {})

    def get_altair_scale(self, name: str) -> alt.Scale:
        """Convert scale definition to Altair Scale object."""
        spec = self._scales.get(name, {})
        if not spec:
            return alt.Scale()
        return alt.Scale(
            domain=spec.get("domain"),
            range=spec.get("range"),
        )

    def register(self, name: str, spec: Dict[str, Any]):
        """Register a new scale definition."""
        self._scales[name] = spec

    def list_scales(self) -> List[str]:
        """List all registered scale names."""
        return list(self._scales.keys())


@dataclass
class AltairPlotSpec:
    """Parsed plot specification from YAML."""

    name: str
    source: str = "combined"
    mark: Dict[str, Any] = field(default_factory=lambda: {"type": "line"})
    encoding: Dict[str, Any] = field(default_factory=dict)
    error_band: Optional[Dict[str, Any]] = None
    facet: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    transform: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "AltairPlotSpec":
        """Create AltairPlotSpec from YAML dict."""
        return cls(
            name=d.get("name", "unnamed"),
            source=d.get("source", "combined"),
            mark=d.get("mark", {"type": "line"}),
            encoding=d.get("encoding", {}),
            error_band=d.get("error_band"),
            facet=d.get("facet"),
            properties=d.get("properties", {}),
            transform=d.get("transform", {}),
        )


class AltairPlotter:
    """
    Altair-based plotter for RL experiments.

    Parallel to SensitivityPlotter but uses Altair/Vega-Lite.
    """

    _TYPE_MAP = {
        "quantitative": "Q",
        "nominal": "N",
        "ordinal": "O",
        "temporal": "T",
        "Q": "Q",
        "N": "N",
        "O": "O",
        "T": "T",
    }

    def __init__(
        self,
        experiment: "Experiment",
        scales: Optional[ScaleRegistry] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize plotter with experiment data.

        Args:
            experiment: Experiment object containing data
            scales: Scale definitions for colors, dashes, etc.
            labels: Label mappings for axis titles
        """
        self.experiment = experiment
        self.scales = scales or ScaleRegistry()
        self.labels = labels or {}

    def get_label(self, field_name: str) -> str:
        """Get human-readable label for a field."""
        return self.labels.get(field_name, field_name)

    def _get_short_type(self, field_type: str) -> str:
        return self._TYPE_MAP.get(field_type, "Q")

    def _resolve_scale(self, scale_spec: Optional[Union[str, Dict[str, Any]]]) -> alt.Scale:
        if isinstance(scale_spec, str):
            return self.scales.get_altair_scale(scale_spec)
        if isinstance(scale_spec, dict):
            return alt.Scale(**scale_spec)
        return alt.Scale()

    def _field_from_spec(self, channel_spec: Any) -> Optional[str]:
        if isinstance(channel_spec, dict):
            return channel_spec.get("field")
        if isinstance(channel_spec, str):
            return channel_spec
        return None

    def _build_axis_channel(
        self,
        channel: str,
        field_name: str,
        short_type: str,
        channel_spec: Dict[str, Any],
        title: str,
    ) -> Union[alt.X, alt.Y]:
        if channel == "x":
            enc = alt.X(f"{field_name}:{short_type}", title=title)
        else:
            enc = alt.Y(f"{field_name}:{short_type}", title=title)
        if "scale" in channel_spec:
            enc = enc.scale(**self._parse_scale_spec(channel_spec["scale"]))
        return enc

    def _build_line_with_ci(
        self,
        df: pd.DataFrame,
        x_field: str,
        y_field: str,
        x_title: str,
        y_title: str,
        group_fields: Optional[List[str]] = None,
        group_titles: Optional[List[str]] = None,
        color_scale: Optional[str] = None,
        y_scale_kwargs: Optional[Dict[str, Any]] = None,
        show_points: bool = True,
        point_size: int = 50,
        band_opacity: float = 0.2,
    ) -> alt.Chart:
        encoding = {
            "x": alt.X(f"{x_field}:Q", title=x_title),
            "y": alt.Y(
                f"{y_field}:Q",
                title=y_title,
                scale=alt.Scale(**(y_scale_kwargs or {})),
            ),
        }

        if group_fields:
            group_titles = group_titles or group_fields
            encoding["color"] = alt.Color(
                f"{group_fields[0]}:N",
                title=self.get_label(group_titles[0]),
                scale=self.scales.get_altair_scale(color_scale) if color_scale else alt.Scale(),
            )
            if len(group_fields) > 1:
                encoding["strokeDash"] = alt.StrokeDash(
                    f"{group_fields[1]}:O",
                    title=self.get_label(group_titles[1]),
                    legend=alt.Legend(
                        symbolType="stroke",
                        symbolStrokeWidth=2,
                        symbolSize=100,
                    ),
                )
                encoding["detail"] = [alt.Detail(f"{f}:N") for f in group_fields]

        line = alt.Chart(df).mark_line().encode(**encoding)

        band_encoding = {
            "x": alt.X(f"{x_field}:Q"),
            "y": alt.Y("ci_lower:Q"),
            "y2": alt.Y2("ci_upper:Q"),
        }
        if group_fields:
            band_encoding["color"] = alt.Color(
                f"{group_fields[0]}:N",
                scale=self.scales.get_altair_scale(color_scale) if color_scale else alt.Scale(),
                legend=None,
            )
            if len(group_fields) > 1:
                band_encoding["detail"] = [alt.Detail(f"{f}:N") for f in group_fields]

        band = alt.Chart(df).mark_area(opacity=band_opacity).encode(**band_encoding)

        chart = band + line
        if show_points:
            point_encoding = {k: v for k, v in encoding.items() if k != "strokeDash"}
            points = alt.Chart(df).mark_point(size=point_size).encode(**point_encoding)
            chart = chart + points

        return chart

    def from_yaml_spec(self, plot_spec: Dict[str, Any]) -> alt.Chart:
        """
        Build chart directly from YAML plot specification.

        Main entry point for yaml_plot_runner.py

        Args:
            plot_spec: Dictionary from YAML plots list

        Returns:
            Altair Chart object
        """
        spec = AltairPlotSpec.from_dict(plot_spec)

        # Get grouped data (avoid regrouping already-aggregated frames)
        if _metric_columns_contain_lists(self.experiment.df):
            df = self.experiment.df
        else:
            df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)

        # Apply transform to get tidy data
        transform_spec = TransformSpec.from_dict(spec.transform)
        tidy_df = apply_transform_spec(df, transform_spec)

        # Sanitize column names for Vega-Lite (dots cause issues)
        tidy_df, column_map = _sanitize_dataframe(tidy_df)

        # Update spec to use sanitized field names
        sanitized_encoding = _sanitize_spec_fields(spec.encoding, column_map)
        sanitized_facet = _sanitize_spec_fields(spec.facet, column_map) if spec.facet else None

        # Create sanitized spec
        sanitized_spec = AltairPlotSpec(
            name=spec.name,
            source=spec.source,
            mark=spec.mark,
            encoding=sanitized_encoding,
            error_band=spec.error_band,
            facet=sanitized_facet,
            properties=spec.properties,
            transform=spec.transform,
        )

        # Build the chart with sanitized data and spec
        chart = self._build_chart(tidy_df, sanitized_spec, column_map)

        return chart

    def _build_chart(
        self,
        df: pd.DataFrame,
        spec: AltairPlotSpec,
        column_map: Optional[Dict[str, str]] = None,
    ) -> alt.Chart:
        """Build complete chart from spec and data."""
        # Reverse column map for looking up original names for labels
        reverse_map = {v: k for k, v in (column_map or {}).items()}

        # Build encoding
        encoding = self._build_encoding(spec.encoding, reverse_map)

        # Build mark
        mark_type = spec.mark.get("type", "line")
        mark_kwargs = {k: v for k, v in spec.mark.items() if k != "type"}

        # Create base chart
        chart = alt.Chart(df)
        mark_method = getattr(chart, f"mark_{mark_type}", chart.mark_line)
        base = mark_method(**mark_kwargs)

        # Apply encoding
        base = base.encode(**encoding)

        # Add error band if specified
        if spec.error_band:
            band = self._build_error_band(df, spec.encoding, spec.error_band)
            chart = band + base
        else:
            chart = base

        # Add points on top if mark specifies point=True
        if spec.mark.get("point"):
            point_encoding = {k: v for k, v in encoding.items() if k != "strokeDash"}
            points = alt.Chart(df).mark_point(size=50).encode(**point_encoding)
            chart = chart + points

        # Apply properties before faceting
        if spec.properties:
            chart = chart.properties(**spec.properties)

        # Apply faceting
        if spec.facet:
            chart = self._apply_facet(chart, spec.facet, reverse_map)

        return chart

    def _build_encoding(
        self,
        encoding_spec: Dict[str, Any],
        reverse_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Convert YAML encoding spec to Altair encoding kwargs."""
        encoding = {}
        reverse_map = reverse_map or {}

        for channel, channel_spec in encoding_spec.items():
            if channel == "tooltip":
                if isinstance(channel_spec, list):
                    encoding["tooltip"] = [
                        alt.Tooltip(f) if isinstance(f, str) else alt.Tooltip(**f)
                        for f in channel_spec
                    ]
                elif isinstance(channel_spec, dict):
                    field_name = channel_spec.get("field")
                    field_type = channel_spec.get("type", "quantitative")
                    short_type = self._get_short_type(field_type)
                    encoding["tooltip"] = alt.Tooltip(f"{field_name}:{short_type}")
                else:
                    encoding["tooltip"] = alt.Tooltip(channel_spec)
                continue

            if isinstance(channel_spec, str):
                encoding[channel] = channel_spec
                continue

            field_name = channel_spec.get("field")
            if not field_name:
                continue

            original_field = reverse_map.get(field_name, field_name)
            field_type = channel_spec.get("type", "quantitative")
            short_type = self._get_short_type(field_type)

            # Build the encoding (use original_field for labels)
            if channel == "x":
                enc = self._build_axis_channel(
                    "x",
                    field_name,
                    short_type,
                    channel_spec,
                    channel_spec.get("title", self.get_label(original_field)),
                )
                encoding["x"] = enc

            elif channel == "y":
                enc = self._build_axis_channel(
                    "y",
                    field_name,
                    short_type,
                    channel_spec,
                    channel_spec.get("title", self.get_label(original_field)),
                )
                encoding["y"] = enc

            elif channel == "color":
                legend_config = None
                if channel_spec.get("legend", True):
                    legend_config = alt.Legend(title=channel_spec.get("title", self.get_label(original_field)))

                encoding["color"] = alt.Color(
                    f"{field_name}:{short_type}",
                    title=channel_spec.get("title", self.get_label(original_field)),
                    scale=self._resolve_scale(channel_spec.get("scale")),
                    legend=legend_config,
                )

            elif channel == "strokeDash":
                # Configure legend to show line patterns instead of circles
                encoding["strokeDash"] = alt.StrokeDash(
                    f"{field_name}:{short_type}",
                    title=channel_spec.get("title", self.get_label(original_field)),
                    scale=self._resolve_scale(channel_spec.get("scale")),
                    legend=alt.Legend(
                        symbolType="stroke",
                        symbolStrokeWidth=2,
                        symbolSize=100,
                    ),
                )

            elif channel == "detail":
                encoding["detail"] = alt.Detail(f"{field_name}:{short_type}")

        return encoding

    def _parse_scale_spec(self, scale_spec: Union[str, Dict]) -> Dict[str, Any]:
        """Parse scale specification into kwargs for Altair scale."""
        if isinstance(scale_spec, str):
            return {"type": scale_spec}

        result = {}
        if "type" in scale_spec:
            result["type"] = scale_spec["type"]
        if "domain" in scale_spec:
            result["domain"] = scale_spec["domain"]
        if "range" in scale_spec:
            result["range"] = scale_spec["range"]
        if "zero" in scale_spec:
            result["zero"] = scale_spec["zero"]

        return result

    def _build_error_band(
        self,
        df: pd.DataFrame,
        encoding_spec: Dict[str, Any],
        error_band_spec: Dict[str, Any],
    ) -> alt.Chart:
        """Build error band layer."""
        opacity = error_band_spec.get("opacity", 0.2)

        # Get x encoding
        x_spec = encoding_spec.get("x", {})
        x_field = self._field_from_spec(x_spec)

        # Get color encoding for the band
        color_spec = encoding_spec.get("color", {})
        color_field = self._field_from_spec(color_spec) if isinstance(color_spec, dict) else None

        # Build band encoding
        band_encoding = {
            "x": alt.X(f"{x_field}:Q"),
            "y": alt.Y("ci_lower:Q", title=""),
            "y2": alt.Y2("ci_upper:Q"),
        }

        if color_field:
            band_encoding["color"] = alt.Color(
                f"{color_field}:N",
                scale=self._resolve_scale(color_spec.get("scale")),
                legend=None,  # Hide legend for band
            )

        # Add detail for proper grouping
        group_fields = []
        for channel in ["color", "strokeDash", "detail"]:
            ch_spec = encoding_spec.get(channel, {})
            if isinstance(ch_spec, dict) and "field" in ch_spec:
                group_fields.append(ch_spec["field"])

        if len(group_fields) > 1:
            band_encoding["detail"] = [alt.Detail(f"{f}:N") for f in group_fields]

        return alt.Chart(df).mark_area(opacity=opacity).encode(**band_encoding)

    def _apply_facet(
        self,
        chart: alt.Chart,
        facet_spec: Dict[str, Any],
        reverse_map: Optional[Dict[str, str]] = None,
    ) -> alt.Chart:
        """Apply faceting to chart."""
        facet_kwargs = {}
        reverse_map = reverse_map or {}

        if "column" in facet_spec:
            col_spec = facet_spec["column"]
            field = col_spec.get("field")
            original_field = reverse_map.get(field, field)
            title = col_spec.get("title", self.get_label(original_field))
            sort = col_spec.get("sort")

            column = alt.Column(
                f"{field}:N",
                title=title,
                sort=sort,
            )
            facet_kwargs["column"] = column

        if "row" in facet_spec:
            row_spec = facet_spec["row"]
            field = row_spec.get("field")
            original_field = reverse_map.get(field, field)
            title = row_spec.get("title", self.get_label(original_field))
            sort = row_spec.get("sort")

            row = alt.Row(
                f"{field}:N",
                title=title,
                sort=sort,
            )
            facet_kwargs["row"] = row

        return chart.facet(**facet_kwargs)

    def sensitivity(
        self,
        x_key: str,
        metric: str = "returns:avg_end",
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        agg_func: str = "median",
        ci_type: str = "quantile",
        ci_lower: float = 0.25,
        ci_upper: float = 0.75,
        log_scale: bool = False,
        ylims: Optional[tuple] = None,
        show_points: bool = True,
        width: int = 300,
        height: int = 200,
    ) -> alt.Chart:
        """
        Create sensitivity curve using Altair.

        Convenience method that mirrors the matplotlib API.

        Args:
            x_key: Hyperparameter for x-axis
            metric: Metric for y-axis
            group_by: Keys to group series by (different lines/colors)
            facet_by: Key to create subplot facets
            agg_func: Aggregation function name
            ci_type: CI type (quantile, sem, std)
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            log_scale: Use symlog y-scale
            ylims: Y-axis limits
            show_points: Show point markers
            width: Chart width
            height: Chart height

        Returns:
            Altair Chart object
        """
        # Get grouped data (avoid regrouping already-aggregated frames)
        if _metric_columns_contain_lists(self.experiment.df):
            df = self.experiment.df
        else:
            df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)
        df = df.sort_values(x_key)

        # Build group columns for transform
        group_cols = [x_key]
        if group_by:
            group_cols.extend(group_by)
        if facet_by:
            group_cols.append(facet_by)

        # Transform data
        tidy_df = aggregate_with_ci(
            df=df,
            metric=metric,
            group_cols=group_cols,
            agg_func=agg_func,
            ci_type=ci_type,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

        # Sanitize column names for Vega-Lite
        tidy_df, _ = _sanitize_dataframe(tidy_df)

        # Create sanitized field names (keeping original for labels)
        s_x_key = _sanitize_field_name(x_key)
        s_group_by = [_sanitize_field_name(f) for f in group_by] if group_by else None
        s_facet_by = _sanitize_field_name(facet_by) if facet_by else None

        # Build y scale
        y_scale_kwargs = {}
        if log_scale:
            y_scale_kwargs["type"] = "symlog"
        if ylims:
            y_scale_kwargs["domain"] = list(ylims)

        chart = self._build_line_with_ci(
            df=tidy_df,
            x_field=s_x_key,
            y_field="value",
            x_title=self.get_label(x_key),
            y_title=self.get_label(metric),
            group_fields=s_group_by,
            group_titles=group_by,
            color_scale="agent_color",
            y_scale_kwargs=y_scale_kwargs,
            show_points=show_points,
            point_size=50,
            band_opacity=0.2,
        )

        # Apply properties before faceting
        chart = chart.properties(width=width, height=height)

        # Apply faceting (use sanitized field name, original for label)
        if s_facet_by:
            chart = chart.facet(
                column=alt.Column(f"{s_facet_by}:N", title=self.get_label(facet_by))
            )

        return chart

    def individual_returns(
        self,
        x_key: str,
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        num_agents: int = 2,
        metric_prefix: str = "sep_returns:end",
        agg_func: str = "median",
        ci_type: str = "quantile",
        ci_lower: float = 0.25,
        ci_upper: float = 0.75,
        log_scale: bool = False,
        width: int = 200,
        height: int = 150,
    ) -> alt.Chart:
        """
        Create per-agent return sensitivity plots using Altair.

        Args:
            x_key: Hyperparameter for x-axis
            group_by: Keys to group series by
            facet_by: Key for column facets
            num_agents: Number of agents
            metric_prefix: Prefix for individual return metrics
            agg_func: Aggregation function
            ci_type: CI type
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            log_scale: Use symlog y-scale
            width: Chart width per facet
            height: Chart height per row

        Returns:
            Altair Chart with row faceting by agent
        """
        # Get grouped data (avoid regrouping already-aggregated frames)
        if _metric_columns_contain_lists(self.experiment.df):
            df = self.experiment.df
        else:
            df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)
        df = df.sort_values(x_key)

        # Build group columns
        group_cols = [x_key]
        if group_by:
            group_cols.extend(group_by)
        if facet_by:
            group_cols.append(facet_by)

        # Prepare data for each agent
        all_agent_data = []
        for agent_idx in range(num_agents):
            agent_df = prepare_individual_returns(
                df=df,
                metric_prefix=metric_prefix,
                agent_idx=agent_idx,
                group_cols=group_cols,
                agg_func=agg_func,
                ci_type=ci_type,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
            all_agent_data.append(agent_df)

        tidy_df = pd.concat(all_agent_data, ignore_index=True)

        # Sanitize column names for Vega-Lite
        tidy_df, _ = _sanitize_dataframe(tidy_df)

        # Create sanitized field names (keeping original for labels)
        s_x_key = _sanitize_field_name(x_key)
        s_group_by = [_sanitize_field_name(f) for f in group_by] if group_by else None
        s_facet_by = _sanitize_field_name(facet_by) if facet_by else None

        # Build y scale
        y_scale_kwargs = {}
        if log_scale:
            y_scale_kwargs["type"] = "symlog"

        chart = self._build_line_with_ci(
            df=tidy_df,
            x_field=s_x_key,
            y_field="value",
            x_title=self.get_label(x_key),
            y_title="Return",
            group_fields=s_group_by,
            group_titles=group_by,
            color_scale="agent_color",
            y_scale_kwargs=y_scale_kwargs,
            show_points=True,
            point_size=30,
            band_opacity=0.2,
        )

        # Apply faceting: rows by agent, columns by facet_by (use sanitized names)
        facet_kwargs = {
            "row": alt.Row("agent_idx:O", title="Agent"),
        }
        if s_facet_by:
            facet_kwargs["column"] = alt.Column(
                f"{s_facet_by}:N", title=self.get_label(facet_by)
            )

        chart = chart.properties(width=width, height=height).facet(**facet_kwargs)

        return chart
