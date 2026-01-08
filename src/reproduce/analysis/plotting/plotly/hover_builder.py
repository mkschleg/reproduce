"""
Hover template builder for Plotly plots.
"""

from typing import Dict, List, Optional, Union, Any, Tuple


class HoverTemplateBuilder:
    """Build hover templates for Plotly."""

    def __init__(self, labels: Dict[str, str]):
        self.labels = labels

    def get_label(self, field_name: str) -> str:
        """Get human-readable label for field."""
        return self.labels.get(field_name, field_name)

    def build_hovertemplate(
        self,
        tooltip_spec: Optional[Union[List[Any], Dict[str, Any]]],
        extra_fields: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Build hover template and field list."""
        if not tooltip_spec:
            return None, None

        fields = []
        titles = []

        if isinstance(tooltip_spec, list):
            for item in tooltip_spec:
                if isinstance(item, str):
                    fields.append(item)
                    titles.append(self.get_label(item))
                elif isinstance(item, dict) and "field" in item:
                    fields.append(item["field"])
                    titles.append(
                        item.get("title", self.get_label(item["field"]))
                    )
        elif isinstance(tooltip_spec, dict) and "field" in tooltip_spec:
            fields.append(tooltip_spec["field"])
            titles.append(
                tooltip_spec.get("title", self.get_label(tooltip_spec["field"]))
            )

        if extra_fields:
            for field in extra_fields:
                if field not in fields:
                    fields.append(field)
                    titles.append(self.get_label(field))

        if not fields:
            return None, None

        hover_lines = [
            f"{titles[idx]}: %{{customdata[{idx}]}}" for idx in range(len(fields))
        ]
        return "<br>".join(hover_lines) + "<extra></extra>", fields
