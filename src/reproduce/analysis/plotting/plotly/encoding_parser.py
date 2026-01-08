"""
Encoding specification parser for Plotly plots.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ParsedEncoding:
    """Structured encoding specification."""

    x_field: Optional[str]
    y_field: Optional[str]
    color_field: Optional[str]
    dash_field: Optional[str]
    detail_field: Optional[str]

    x_spec: Dict[str, Any]
    y_spec: Dict[str, Any]
    color_spec: Dict[str, Any]
    dash_spec: Dict[str, Any]
    detail_spec: Dict[str, Any]
    tooltip_spec: Optional[Any]

    @property
    def group_fields(self) -> List[str]:
        """Get all fields used for grouping."""
        fields = []
        for field in [self.color_field, self.dash_field, self.detail_field]:
            if field and field not in fields:
                fields.append(field)
        return fields


class EncodingParser:
    """Parse YAML encoding specs into structured format."""

    def parse(self, encoding: Dict[str, Any]) -> ParsedEncoding:
        """Extract all channel specs and field names."""
        return ParsedEncoding(
            x_field=self._get_field_name(encoding.get("x", {})),
            y_field=self._get_field_name(encoding.get("y", {})),
            color_field=self._get_field_name(encoding.get("color", {})),
            dash_field=self._get_field_name(encoding.get("strokeDash", {})),
            detail_field=self._get_field_name(encoding.get("detail", {})),
            x_spec=encoding.get("x", {}),
            y_spec=encoding.get("y", {}),
            color_spec=encoding.get("color", {}),
            dash_spec=encoding.get("strokeDash", {}),
            detail_spec=encoding.get("detail", {}),
            tooltip_spec=encoding.get("tooltip"),
        )

    def _get_field_name(self, spec: Any) -> Optional[str]:
        """Extract field name from channel spec."""
        if isinstance(spec, dict):
            return spec.get("field")
        if isinstance(spec, str):
            return spec
        return None
