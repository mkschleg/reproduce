"""
Plotly-based plotter for RL experiments.

Refactored version with modular components.
"""

from typing import Dict, Optional, Any, TYPE_CHECKING
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import rl4psjoint.utils.data_utils as data_utils
from ..data_transforms import TransformSpec, apply_transform_spec
from ..altair_plotter import ScaleRegistry
from .spec import PlotlyPlotSpec
from .encoding_parser import EncodingParser
from .scale_builder import PlotlyScaleBuilder
from .axis_transformer import AxisTransformer
from .facet_layout import FacetLayoutBuilder
from .trace_builder import TraceBuilder
from .hover_builder import HoverTemplateBuilder
from .utils import metric_columns_contain_lists

if TYPE_CHECKING:
    from ...loading.experiment import Experiment


class PlotlyPlotter:
    """Plotly-based plotter for RL experiments."""

    # Base size for a single subplot
    _BASE_SUBPLOT_WIDTH = 400
    _BASE_SUBPLOT_HEIGHT = 300

    # Additional space for elements
    _LEGEND_WIDTH = 200
    _MARGIN_LEFT = 80
    _MARGIN_RIGHT = 20
    _MARGIN_TOP = 60
    _MARGIN_BOTTOM = 60

    def __init__(
        self,
        experiment: "Experiment",
        scales: Optional[ScaleRegistry] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize plotter with experiment data."""
        self.experiment = experiment
        self.scales = scales or ScaleRegistry()
        self.labels = labels or {}

    def get_label(self, field_name: str) -> str:
        """Get human-readable label for a field."""
        return self.labels.get(field_name, field_name)

    def from_yaml_spec(self, plot_spec: Dict[str, Any]) -> go.Figure:
        """Build figure from YAML v2 plot specification."""
        spec = PlotlyPlotSpec.from_dict(plot_spec)

        # Get grouped data
        if metric_columns_contain_lists(self.experiment.df):
            df = self.experiment.df
        else:
            df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)

        # Apply transform
        transform_spec = TransformSpec.from_dict(spec.transform)
        tidy_df = apply_transform_spec(df, transform_spec)

        # Build and return figure
        fig = self._build_figure(tidy_df, spec)
        return fig

    def _build_figure(self, df: pd.DataFrame, spec: PlotlyPlotSpec) -> go.Figure:
        """Build figure by orchestrating components."""
        # 1. Parse encoding
        encoding_parser = EncodingParser()
        encoding = encoding_parser.parse(spec.encoding or {})

        # 2. Build scales
        scale_builder = PlotlyScaleBuilder(self.scales, self.labels)
        scale_maps = scale_builder.build_scales(df, encoding)

        # 3. Transform axes (handle log scales)
        axis_transformer = AxisTransformer()
        axis_result = axis_transformer.transform_y_axis(
            df, encoding.y_spec, encoding.y_field
        )
        df = axis_result.df
        y_field = axis_result.y_field

        # Update encoding with transformed field
        if y_field != encoding.y_field:
            encoding.y_field = y_field

        # 4. Compute facet layout
        layout_builder = FacetLayoutBuilder(self.labels)
        layout = layout_builder.build_layout(df, spec.facet, spec.name)

        # 5. Build hover template
        hover_builder = HoverTemplateBuilder(self.labels)
        hovertemplate, hover_fields = hover_builder.build_hovertemplate(
            encoding.tooltip_spec,
            extra_fields=[encoding.x_field, encoding.y_field]
            if encoding.x_field and encoding.y_field
            else None,
        )

        # 6. Create figure with subplots
        fig = make_subplots(
            rows=len(layout.row_values),
            cols=len(layout.col_values),
            subplot_titles=layout.subplot_titles
            if len(layout.row_values) * len(layout.col_values) > 1
            else None,
            shared_xaxes=len(layout.row_values) * len(layout.col_values) > 1,
            shared_yaxes=len(layout.row_values) * len(layout.col_values) > 1,
        )

        # 7. Build and add traces for each facet
        trace_builder = TraceBuilder(scale_maps, self.labels)

        for row_idx, row_val in enumerate(layout.row_values, start=1):
            for col_idx, col_val in enumerate(layout.col_values, start=1):
                # Filter to facet
                facet_df = df.copy()
                if layout.row_field:
                    facet_df = facet_df[facet_df[layout.row_field] == row_val]
                if layout.col_field:
                    facet_df = facet_df[facet_df[layout.col_field] == col_val]

                if facet_df.empty:
                    continue

                # Build traces for this facet
                show_legend = row_idx == 1 and col_idx == 1
                traces = trace_builder.build_traces_for_facet(
                    facet_df=facet_df,
                    encoding=encoding,
                    mark_spec=spec.mark,
                    error_band_spec=spec.error_band,
                    hovertemplate=hovertemplate,
                    hover_fields=hover_fields,
                    show_legend=show_legend,
                )

                # Add traces to figure
                for trace in traces:
                    fig.add_trace(trace, row=row_idx, col=col_idx)

        # 8. Apply axis settings and layout
        self._apply_axis_settings(fig, encoding, axis_result, layout)
        self._apply_layout(fig, spec, layout, encoding)

        return fig

    def _apply_axis_settings(
        self, fig: go.Figure, encoding: Any, axis_result: Any, layout: Any
    ):
        """Apply axis settings to figure."""
        # Build axis settings from encoding
        x_axis_settings = self._axis_settings_from_spec(encoding.x_spec, "x")
        y_axis_settings = self._axis_settings_from_spec(encoding.y_spec, "y")

        # Merge with transformed axis settings
        y_axis_settings.update(axis_result.y_axis_settings)

        # Apply to all subplots
        fig.update_xaxes(**x_axis_settings)
        fig.update_yaxes(**y_axis_settings)

        # Remove redundant titles in multi-facet plots
        if len(layout.row_values) * len(layout.col_values) > 1:
            x_title = x_axis_settings.get("title_text")
            y_title = y_axis_settings.get("title_text")
            if x_title or y_title:
                for row_idx in range(1, len(layout.row_values) + 1):
                    for col_idx in range(1, len(layout.col_values) + 1):
                        if x_title and row_idx != len(layout.row_values):
                            fig.update_xaxes(
                                title_text=None, row=row_idx, col=col_idx
                            )
                        if y_title and col_idx != 1:
                            fig.update_yaxes(
                                title_text=None, row=row_idx, col=col_idx
                            )

        # Apply styling
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="#000000",
            ticks="outside",
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor="#000000",
            ticks="outside",
            showgrid=False,
            zeroline=False,
        )

    def _axis_settings_from_spec(
        self, axis_spec: Any, axis: str
    ) -> Dict[str, Any]:
        """Extract axis settings from encoding spec."""
        if not isinstance(axis_spec, dict):
            return {}

        settings = {}

        title = axis_spec.get("title")
        if title:
            settings["title_text"] = title

        scale_spec = axis_spec.get("scale", {})
        if isinstance(scale_spec, dict):
            if "domain" in scale_spec:
                settings["range"] = scale_spec["domain"]

        return settings

    def _compute_default_size(
        self, layout: Any, encoding: Any
    ) -> tuple[int, int]:
        """
        Compute smart default figure size based on facets and legend.

        Args:
            layout: Facet layout info
            encoding: Encoding specification

        Returns:
            (width, height) in pixels
        """
        n_rows = len(layout.row_values)
        n_cols = len(layout.col_values)

        # Check if legend will be shown (color encoding present)
        has_legend = encoding.color_spec is not None

        # Base plot area size
        plot_width = self._BASE_SUBPLOT_WIDTH * n_cols
        plot_height = self._BASE_SUBPLOT_HEIGHT * n_rows

        # Add margins
        total_width = plot_width + self._MARGIN_LEFT + self._MARGIN_RIGHT
        total_height = plot_height + self._MARGIN_TOP + self._MARGIN_BOTTOM

        # Add space for legend on the right
        if has_legend:
            total_width += self._LEGEND_WIDTH

        # Add extra space for subplot titles if multi-facet
        if n_rows * n_cols > 1:
            total_height += 40  # Extra space for subplot titles

        return (total_width, total_height)

    def _apply_layout(
        self, fig: go.Figure, spec: PlotlyPlotSpec, layout: Any, encoding: Any
    ):
        """Apply layout properties to figure."""
        layout_kwargs = {}

        # Size from properties (explicit override)
        if spec.properties:
            width = spec.properties.get("width")
            height = spec.properties.get("height")
            if width:
                layout_kwargs["width"] = width * max(1, len(layout.col_values))
            if height:
                layout_kwargs["height"] = height * max(1, len(layout.row_values))

        # Legend title from color encoding
        if isinstance(encoding.color_spec, dict) and encoding.color_spec.get("title"):
            layout_kwargs["legend_title_text"] = encoding.color_spec.get("title")

        # Margins for multi-facet
        if (
            len(layout.row_values) * len(layout.col_values) > 1
            and "margin" not in layout_kwargs
        ):
            layout_kwargs["margin"] = dict(
                l=self._MARGIN_LEFT,
                r=self._MARGIN_RIGHT,
                t=self._MARGIN_TOP,
                b=self._MARGIN_BOTTOM,
            )

        # Default styling
        layout_kwargs.setdefault("template", "simple_white")
        layout_kwargs.setdefault(
            "font",
            dict(
                family="DejaVu Sans, Arial, sans-serif",
                size=14,
                color="#000000",
            ),
        )
        layout_kwargs.setdefault("paper_bgcolor", "white")
        layout_kwargs.setdefault("plot_bgcolor", "white")

        # Compute smart default size if not explicitly set
        if "width" not in layout_kwargs or "height" not in layout_kwargs:
            default_width, default_height = self._compute_default_size(layout, encoding)
            layout_kwargs.setdefault("width", default_width)
            layout_kwargs.setdefault("height", default_height)

        fig.update_layout(**layout_kwargs)
