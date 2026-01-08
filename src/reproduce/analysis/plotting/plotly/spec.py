"""
Plot specification dataclasses for Plotly.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PlotlyPlotSpec:
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
    def from_dict(cls, d: dict) -> "PlotlyPlotSpec":
        """Create PlotlyPlotSpec from YAML dict."""
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
