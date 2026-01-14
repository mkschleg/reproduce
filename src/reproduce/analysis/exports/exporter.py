"""
Export functions for generating experiment sweep configs from analysis results.
"""

import os
from typing import Dict, Any, Tuple, Optional, List

from ..loading.experiment import Experiment
from ..utils.data_utils import create_params_for_final_sweep
from .config import ExportConfig, build_export_configs


def export_single_config(
    export_spec: Dict[str, Any],
    loaded: Dict[str, Any],
) -> Tuple[Dict, Dict]:
    """
    Process a single export specification.

    Takes an export specification dict and loaded experiment data,
    applies any filtering/transformations, and generates a sweep config.

    Args:
        export_spec: Dictionary with export configuration (name, source, output, etc.)
        loaded: Dictionary from load_from_config with keys: lake, experiments, combined

    Returns:
        Tuple of (base_config, sweep_config)

    Raises:
        ValueError: If source experiment not found
    """
    config = ExportConfig.from_dict(export_spec)
    return _process_export(config, loaded)


def _process_export(
    config: ExportConfig,
    loaded: Dict[str, Any],
) -> Tuple[Dict, Dict]:
    """
    Internal function to process an ExportConfig.

    Args:
        config: ExportConfig instance
        loaded: Dictionary from load_from_config

    Returns:
        Tuple of (base_config, sweep_config)
    """
    # Get source experiment
    if config.source == "combined":
        exp = loaded.get("combined")
    else:
        exp = loaded.get("experiments", {}).get(config.source)

    if exp is None:
        raise ValueError(f"No experiment found for source: {config.source}")

    # Apply source_mode transformations
    if config.source_mode == "best_hypers":
        df = exp.best_hypers(
            sort_key=config.sort_key,
            ascending=config.ascending,
            best_over=config.best_over or [],
        )
        exp = Experiment(df, f"{exp.name}_best", exp.metadata)
    elif config.source_mode == "grouped":
        df = exp.group_by()
        exp = Experiment(df, f"{exp.name}_grouped", exp.metadata)

    # Apply filters
    if config.filter_query or config.filter_kwargs:
        exp = exp.filter(query=config.filter_query, **(config.filter_kwargs or {}))

    # Generate sweep config
    base, sweep = create_params_for_final_sweep(exp.df, save_file=config.output)

    return base, sweep


def export_sweep_configs(
    config: Dict[str, Any],
    loaded: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple[Dict, Dict]]:
    """
    Process all exports from a config dict.

    Args:
        config: Parsed YAML config dict containing 'exports' section
        loaded: Dictionary from load_from_config with keys: lake, experiments, combined
        output_dir: Optional base directory for output files (prepended to export output paths)

    Returns:
        Dictionary mapping export names to (base_config, sweep_config) tuples

    Example:
        >>> from reproduce.analysis.loading import load_from_config, parse_yaml_config
        >>> config = parse_yaml_config(yaml_str)
        >>> loaded = load_from_config(config)
        >>> results = export_sweep_configs(config, loaded)
        >>> for name, (base, sweep) in results.items():
        ...     print(f"{name}: {len(sweep.get('+params', {}))} sweep params")
    """
    exports_list = config.get("exports", [])
    if not exports_list:
        return {}

    export_configs = build_export_configs(exports_list)
    if not export_configs:
        return {}

    results = {}
    for export_config in export_configs:
        # Handle output_dir prefix
        if output_dir:
            original_output = export_config.output
            export_config.output = os.path.join(output_dir, original_output)

        # Ensure output directory exists
        output_path = export_config.output
        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

        try:
            base, sweep = _process_export(export_config, loaded)
            results[export_config.name] = (base, sweep)
            print(f"Exported: {export_config.name} -> {export_config.output}")
        except Exception as e:
            print(f"Error exporting {export_config.name}: {e}")
            raise

    return results
