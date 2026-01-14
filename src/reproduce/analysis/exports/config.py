"""
Configuration dataclasses for experiment config exports.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ExportConfig:
    """
    Configuration for exporting experiment sweep configs.

    Mirrors the pattern used in plot specifications, supporting
    source selection, filtering, and best_hypers mode.
    """

    name: str
    """Name identifier for this export."""

    source: str
    """Source experiment name or 'combined'."""

    output: str
    """Output file path for the generated YAML config."""

    source_mode: str = "raw"
    """
    How to process the source data:
    - "raw": Use the data as-is
    - "best_hypers": Apply best hyperparameter selection
    - "grouped": Group by hyperparameters
    """

    filter_query: Optional[str] = None
    """Pandas query string to filter data before export."""

    filter_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Key-value filters for Experiment.filter()."""

    # best_hypers mode options
    best_over: Optional[List[str]] = None
    """Columns to find best hyperparameters over (for best_hypers mode)."""

    sort_key: str = "returns:avg_end_mean"
    """Metric key to sort by when finding best hyperparameters."""

    ascending: bool = False
    """Sort order for best hyperparameter selection."""

    @classmethod
    def from_dict(cls, spec: Dict[str, Any]) -> "ExportConfig":
        """
        Create ExportConfig from a dictionary specification.

        Args:
            spec: Dictionary with export configuration

        Returns:
            ExportConfig instance
        """
        return cls(
            name=spec["name"],
            source=spec.get("source", "combined"),
            output=spec["output"],
            source_mode=spec.get("source_mode", "raw"),
            filter_query=spec.get("filter_query"),
            filter_kwargs=spec.get("filter_kwargs") or {},
            best_over=spec.get("best_over"),
            sort_key=spec.get("sort_key", "returns:avg_end_mean"),
            ascending=spec.get("ascending", False),
        )


def build_export_configs(exports_list: Optional[List[Dict]]) -> Optional[List[ExportConfig]]:
    """
    Build list of ExportConfig objects from YAML section.

    Args:
        exports_list: List of export config dicts from YAML

    Returns:
        List of ExportConfig objects, or None if input is None/empty

    Example:
        >>> exports_list = [
        ...     {"name": "best_hypers", "source": "exp1", "output": "config.yaml",
        ...      "source_mode": "best_hypers", "best_over": ["agent.type"]},
        ... ]
        >>> configs = build_export_configs(exports_list)
        >>> len(configs)
        1
    """
    if not exports_list:
        return None

    return [ExportConfig.from_dict(spec) for spec in exports_list]
