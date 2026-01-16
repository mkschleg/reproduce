"""YAML parsing helpers for loading and cleaning configs."""

from typing import Any, Dict, List, Optional
import numpy as np
import yaml

from ..notebook_config import DataCleaningConfig, ColumnCleanSpec
from ..plotting.migrate import migrate_v1_to_v2
from ..exports.config import ExportConfig


def parse_yaml_config(yaml_str: str, auto_migrate: bool = True) -> Dict[str, Any]:
    """
    Parse a YAML config string into a dict.

    Handles version routing:
    - Version 1 configs with backend: altair are auto-migrated to v2
    - Version 2 configs are returned as-is

    Args:
        yaml_str: YAML string to parse
        auto_migrate: If True, auto-migrate v1 configs that request altair backend

    Returns:
        Parsed config dict (possibly migrated to v2)
    """
    config = yaml.safe_load(yaml_str) or {}

    version = config.get("version", 1)
    backend = config.get("plotting", {}).get("backend", "matplotlib")

    # Auto-migrate v1 configs that want altair backend
    if auto_migrate and version == 1 and backend == "altair":
        config = migrate_v1_to_v2(config)

    return config


def get_config_backend(config: Dict[str, Any]) -> str:
    """
    Get the plotting backend from a config.

    Args:
        config: Parsed config dict

    Returns:
        Backend name ("matplotlib", "altair", or "plotly")
    """
    return config.get("plotting", {}).get("backend", "matplotlib")


def _translate_replace(replace_map: Optional[Dict[Any, Any]]) -> Optional[Dict[Any, Any]]:
    if not replace_map:
        return None
    translated = {}
    for key, value in replace_map.items():
        if isinstance(key, str) and key.strip().lower() in {".nan", "nan"}:
            translated[np.nan] = value
        else:
            translated[key] = value
    return translated


def build_cleaning_config(
    cleaning_cfg: Optional[Dict[str, Any]]
) -> DataCleaningConfig:
    """Build a DataCleaningConfig from a YAML-parsed dict section."""
    cleaning_cfg = cleaning_cfg or {}
    columns = {}

    for col, spec in (cleaning_cfg.get("columns") or {}).items():
        columns[col] = ColumnCleanSpec(
            replace=_translate_replace(spec.get("replace")),
            cast=spec.get("cast"),
            filter_eq=spec.get("filter_eq"),
            filter_in=spec.get("filter_in"),
            dropna=bool(spec.get("dropna", False)),
        )

    return DataCleaningConfig(
        columns=columns,
        drop_columns=tuple(cleaning_cfg.get("drop_columns", ("env.render_mode",))),
        filter_query=cleaning_cfg.get("filter_query"),
    )


def build_aggregation_configs(
    agg_cfg_list: Optional[list]
) -> Optional[list]:
    """
    Build list of AggregationConfig objects from YAML section.

    Args:
        agg_cfg_list: List of aggregation config dicts from YAML

    Returns:
        List of AggregationConfig objects, or None if input is None

    Example:
        >>> agg_cfg_list = [
        ...     {"type": "avg_end", "n": 5, "name": "short_avg"},
        ...     {"type": "median_end", "n": 5}
        ... ]
        >>> configs = build_aggregation_configs(agg_cfg_list)
        >>> len(configs)
        2
    """
    if not agg_cfg_list:
        return None

    from .aggregation import AggregationConfig
    return [AggregationConfig.from_config(cfg) for cfg in agg_cfg_list]


def build_export_configs(
    exports_list: Optional[List[Dict[str, Any]]]
) -> Optional[List[ExportConfig]]:
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
