"""Centralized experiment loading/cleaning helpers."""

from typing import Any, Dict, Optional

from .datalake import DataLake
from .experiment import Experiment
from ..notebook_config import clean_experiment
from .yaml_utils import parse_yaml_config, build_cleaning_config


def load_from_config(
    config: Dict[str, Any],
    lake: Optional[DataLake] = None
) -> Dict[str, Any]:
    """
    Load experiments, apply cleaning, and optionally combine using config dict.

    Returns a dict with keys: lake, experiments, combined.
    """
    load_cfg = config.get("load", {})

    lake = lake or DataLake(load_cfg.get("base_path", "datalake"))

    # Parse default aggregations (applies to ALL experiments)
    agg_funcs = None
    if "default_aggregations" in load_cfg:
        from .yaml_utils import build_aggregation_configs
        from .aggregation import aggregation_configs_to_dict
        agg_configs = build_aggregation_configs(load_cfg["default_aggregations"])
        agg_funcs = aggregation_configs_to_dict(agg_configs) if agg_configs else None

    # Load experiments (all experiments are just strings)
    experiment_names = load_cfg.get("experiments", [])
    experiments = {}
    for exp_name in experiment_names:
        experiments[exp_name] = lake.load(exp_name, agg_funcs=agg_funcs)

    cleaning_cfg = build_cleaning_config(config.get("cleaning"))
    to_clean = set((config.get("cleaning") or {}).get("apply_to", []))

    cleaned = {}
    for name, exp in experiments.items():
        if name in to_clean:
            cleaned[name] = clean_experiment(exp, cleaning_cfg)
        else:
            cleaned[name] = exp

    combine_name = load_cfg.get("combine_name")
    combined = None
    if combine_name and len(cleaned) > 1:
        combined = Experiment.combine_multiple(list(cleaned.values()), name=combine_name)

    return {
        "lake": lake,
        "experiments": cleaned,
        "combined": combined,
    }


def load_from_yaml(
    yaml_str: str,
    lake: Optional[DataLake] = None
) -> Dict[str, Any]:
    """Parse YAML and load experiments via load_from_config."""
    config = parse_yaml_config(yaml_str)
    return load_from_config(config, lake=lake)
