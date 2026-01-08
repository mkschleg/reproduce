"""
Facet layout computation for Plotly subplots.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd


@dataclass
class FacetLayout:
    """Computed facet layout."""

    row_values: List[Any]
    col_values: List[Any]
    row_field: Optional[str]
    col_field: Optional[str]
    subplot_titles: List[str]


class FacetLayoutBuilder:
    """Compute subplot layout from facet spec."""

    def __init__(self, labels: Dict[str, str]):
        self.labels = labels

    def get_label(self, field_name: str) -> str:
        """Get human-readable label for field."""
        return self.labels.get(field_name, field_name)

    def build_layout(
        self, df: pd.DataFrame, facet_spec: Optional[Dict[str, Any]], plot_name: str
    ) -> FacetLayout:
        """Determine facet grid layout."""
        facet_spec = facet_spec or {}

        # Parse row/column specs
        row_spec = facet_spec.get("row")
        col_spec = facet_spec.get("column")

        row_field = row_spec.get("field") if isinstance(row_spec, dict) else None
        col_field = col_spec.get("field") if isinstance(col_spec, dict) else None

        # Get facet values
        row_values = [None]
        col_values = [None]

        if row_field:
            row_values = self._facet_values(df, row_field, row_spec.get("sort"))
        if col_field:
            col_values = self._facet_values(df, col_field, col_spec.get("sort"))

        # Build subplot titles
        subplot_titles = []
        for row_val in row_values:
            for col_val in col_values:
                parts = []
                if row_field:
                    parts.append(f"{self.get_label(row_field)}: {row_val}")
                if col_field:
                    parts.append(f"{self.get_label(col_field)}: {col_val}")
                subplot_titles.append(" | ".join(parts) if parts else plot_name)

        return FacetLayout(
            row_values=row_values,
            col_values=col_values,
            row_field=row_field,
            col_field=col_field,
            subplot_titles=subplot_titles,
        )

    def _facet_values(
        self, df: pd.DataFrame, field: str, sort: Optional[List[Any]] = None
    ) -> List[Any]:
        """Get unique values for faceting."""
        if sort:
            return [v for v in sort if v in set(df[field].unique())]
        return sorted(df[field].unique())
