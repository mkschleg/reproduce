"""
Migration utilities for converting v1 YAML configs to v2 Altair format.
"""

from typing import Dict, Any, List, Optional
import copy


def migrate_v1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert version 1 YAML config to version 2 Altair format.

    Transformation rules:
    - plots[].type: "sensitivity" -> mark: {type: line, point: true}
    - plots[].x_key -> encoding.x.field
    - plots[].metric -> transform.metric
    - plots[].group_by -> encoding.color/strokeDash + transform.group_by
    - plots[].facet_by -> facet.column.field
    - config.overrides.log_scale -> encoding.y.scale.type: symlog
    - config.overrides.ylims -> encoding.y.scale.domain

    Args:
        config: Version 1 config dictionary

    Returns:
        Version 2 config dictionary
    """
    result = copy.deepcopy(config)
    result["version"] = 2

    # Migrate plotting section
    plotting = result.get("plotting", {})
    plotting["backend"] = "altair"

    # Add default scales
    if "scales" not in plotting:
        plotting["scales"] = _get_default_scales()

    # Add default labels
    if "labels" not in plotting:
        plotting["labels"] = _get_default_labels()

    result["plotting"] = plotting

    # Migrate each plot
    new_plots = []
    for plot in result.get("plots", []):
        new_plot = _migrate_plot(plot, plotting)
        new_plots.append(new_plot)

    result["plots"] = new_plots

    return result


def _migrate_plot(plot: Dict[str, Any], plotting: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a single plot specification from v1 to v2."""
    result = {
        "name": plot.get("name", "unnamed"),
        "source": plot.get("source", "combined"),
    }

    # Get plot type
    plot_type = plot.get("type", "sensitivity")

    # Migrate mark
    if plot_type == "sensitivity":
        result["mark"] = {"type": "line", "point": True}
    elif plot_type == "individual_returns":
        result["mark"] = {"type": "line", "point": True}
    else:
        result["mark"] = {"type": "line"}

    # Build transform
    transform = {
        "metric": plot.get("metric", "returns:avg_end"),
        "agg_func": plot.get("agg_func", "median"),
        "ci_type": plot.get("ci_type", "quantile"),
        "ci_lower": plot.get("ci_lower", 0.25),
        "ci_upper": plot.get("ci_upper", 0.75),
    }

    # Build group_by for transform (includes all grouping keys)
    group_cols = []
    x_key = plot.get("x_key")
    if x_key:
        group_cols.append(x_key)

    group_by = plot.get("group_by", [])
    if isinstance(group_by, str):
        group_by = [group_by]
    group_cols.extend(group_by)

    facet_by = plot.get("facet_by")
    if facet_by:
        group_cols.append(facet_by)

    transform["group_by"] = group_cols
    result["transform"] = transform

    # Build encoding
    encoding = {}

    # X encoding
    if x_key:
        encoding["x"] = {
            "field": x_key,
            "type": "quantitative",
        }

    # Y encoding
    encoding["y"] = {
        "field": "value",
        "type": "quantitative",
    }

    # Apply config overrides to encoding
    plot_config = plot.get("config", {})
    overrides = plot_config.get("overrides", {})

    if overrides.get("log_scale"):
        encoding["y"]["scale"] = {"type": "symlog"}

    if overrides.get("ylims"):
        if "scale" not in encoding["y"]:
            encoding["y"]["scale"] = {}
        encoding["y"]["scale"]["domain"] = list(overrides["ylims"])

    if overrides.get("xlims"):
        if "scale" not in encoding["x"]:
            encoding["x"]["scale"] = {}
        encoding["x"]["scale"]["domain"] = list(overrides["xlims"])

    if overrides.get("xlabel"):
        encoding["x"]["title"] = overrides["xlabel"]

    if overrides.get("ylabel"):
        encoding["y"]["title"] = overrides["ylabel"]

    # Color encoding (first group_by key)
    if len(group_by) > 0:
        color_field = group_by[0]
        encoding["color"] = {
            "field": color_field,
            "type": "nominal",
        }
        # Use agent_color scale if it's the agent type field
        if color_field == "combo_agent.type":
            encoding["color"]["scale"] = "agent_color"

    # StrokeDash encoding (second group_by key if present)
    if len(group_by) > 1:
        encoding["strokeDash"] = {
            "field": group_by[1],
            "type": "ordinal",
        }

    result["encoding"] = encoding

    # Error band (always enabled for sensitivity plots)
    result["error_band"] = {
        "opacity": overrides.get("ci_alpha", 0.2),
    }

    # Faceting
    if facet_by:
        result["facet"] = {
            "column": {
                "field": facet_by,
                "type": "nominal",
            }
        }

    # Properties
    result["properties"] = {
        "width": 300,
        "height": 200,
    }

    # Handle individual_returns specific fields
    if plot_type == "individual_returns":
        result["_v1_type"] = "individual_returns"
        result["_v1_num_agents"] = plot.get("num_agents", 2)
        result["_v1_metric_prefix"] = plot.get("metric_prefix", "sep_returns:end")
        # For individual returns, we need to facet by agent_idx as well
        if facet_by:
            result["facet"]["row"] = {
                "field": "agent_idx",
                "type": "ordinal",
                "title": "Agent",
            }
        else:
            result["facet"] = {
                "row": {
                    "field": "agent_idx",
                    "type": "ordinal",
                    "title": "Agent",
                }
            }

    # Copy unsupported fields as comments
    if plot.get("unsupported"):
        result["_unsupported"] = plot["unsupported"]

    # Handle source_mode
    if plot.get("source_mode"):
        result["_v1_source_mode"] = plot["source_mode"]
        result["_v1_sort_key"] = plot.get("sort_key")
        result["_v1_ascending"] = plot.get("ascending")
        result["_v1_best_over"] = plot.get("best_over")

    # Handle filters
    if plot.get("filter_query"):
        result["_v1_filter_query"] = plot["filter_query"]
    if plot.get("filter_kwargs"):
        result["_v1_filter_kwargs"] = plot["filter_kwargs"]

    return result


def _get_default_scales() -> Dict[str, Dict[str, Any]]:
    """Get default Altair scale definitions."""
    return {
        "agent_color": {
            "domain": [
                "Centralized",
                "Decentralized",
                "DecentralizedJointObs",
                "DecentralizedSumRew",
                "DecentralizedJointObsSumRew",
            ],
            "range": [
                "#4477AA",
                "#EE6677",
                "#EE6677",
                "#228833",
                "#228833",
            ],
        },
        "agent_dash": {
            "domain": [
                "Centralized",
                "Decentralized",
                "DecentralizedJointObs",
                "DecentralizedSumRew",
                "DecentralizedJointObsSumRew",
            ],
            "range": [
                [1, 0],
                [4, 4],
                [1, 0],
                [4, 4],
                [1, 0],
            ],
        },
    }


def _get_default_labels() -> Dict[str, str]:
    """Get default field label mappings."""
    return {
        "combo_agent.type": "Agent Type",
        "env.envs": "Environment",
        "env.action_func.swap_prob": "Swap Probability",
        "base_dqn.q_network.width": "Network Width",
        "returns:avg_end": "Average Return",
        "returns:avg_all": "Average Return (All)",
        "returns:median_end": "Median Return",
    }


def infer_scale_definitions_from_data(
    df,
    field: str,
    colors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Infer Altair scale definition from data.

    Args:
        df: DataFrame to inspect
        field: Field name to create scale for
        colors: Optional color palette

    Returns:
        Scale definition dictionary
    """
    unique_values = sorted(df[field].unique())

    if colors is None:
        colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

    return {
        "domain": list(unique_values),
        "range": colors[: len(unique_values)],
    }


def print_migration_diff(v1_config: Dict, v2_config: Dict) -> str:
    """
    Generate a human-readable diff showing migration changes.

    Args:
        v1_config: Original v1 config
        v2_config: Migrated v2 config

    Returns:
        String describing the changes
    """
    lines = ["Migration Summary:", "=" * 50]

    lines.append(f"\nVersion: {v1_config.get('version', 1)} -> {v2_config.get('version', 2)}")
    lines.append(f"Backend: matplotlib -> {v2_config.get('plotting', {}).get('backend', 'altair')}")

    v1_plots = v1_config.get("plots", [])
    v2_plots = v2_config.get("plots", [])

    lines.append(f"\nPlots migrated: {len(v1_plots)}")

    for i, (v1, v2) in enumerate(zip(v1_plots, v2_plots)):
        lines.append(f"\n  Plot {i + 1}: {v1.get('name', 'unnamed')}")
        lines.append(f"    Type: {v1.get('type', 'sensitivity')} -> mark.type={v2.get('mark', {}).get('type')}")
        lines.append(f"    x_key: {v1.get('x_key')} -> encoding.x.field")
        lines.append(f"    metric: {v1.get('metric')} -> transform.metric")

        if v1.get("group_by"):
            lines.append(f"    group_by: {v1.get('group_by')} -> encoding.color/strokeDash")

        if v1.get("facet_by"):
            lines.append(f"    facet_by: {v1.get('facet_by')} -> facet.column.field")

        overrides = v1.get("config", {}).get("overrides", {})
        if overrides:
            lines.append(f"    config.overrides -> encoding.y.scale: {overrides}")

    return "\n".join(lines)
