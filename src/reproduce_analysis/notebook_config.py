"""
Shared plotting and data-cleaning defaults for marimo notebooks.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd

from .plotting import PlotConfig
from .experiment import Experiment


@dataclass(frozen=True)
class ColumnCleanSpec:
    """Declarative, YAML-friendly column cleaning specification."""

    replace: Optional[Dict[Any, Any]] = None
    cast: Optional[str] = None  # "int", "float", "str", "bool"
    filter_eq: Optional[Any] = None
    filter_in: Optional[Sequence[Any]] = None
    dropna: bool = False


@dataclass(frozen=True)
class DataCleaningConfig:
    """Configuration for cleaning experiment DataFrames in notebooks."""

    columns: Dict[str, ColumnCleanSpec] = None
    drop_columns: Sequence[str] = ("env.render_mode",)
    filter_query: Optional[str] = None

    @staticmethod
    def default() -> "DataCleaningConfig":
        """Default cleaning matching legacy notebook behavior."""
        return DataCleaningConfig(
            columns={
                "env.max_episode_steps": ColumnCleanSpec(
                    replace={np.nan: "None", 10000.0: 10000},
                    filter_eq=10000
                )
            },
            drop_columns=("env.render_mode",)
        )


def clean_experiment_df(
    df: pd.DataFrame,
    config: Optional[DataCleaningConfig] = None
) -> pd.DataFrame:
    """
    Clean a raw experiment DataFrame for notebook use.

    Applies max-episode-step normalization, optional filtering, and
    drops nuisance columns if present.
    """
    config = config or DataCleaningConfig.default()
    cleaned = df.copy()

    if config.columns:
        for col, spec in config.columns.items():
            if col not in cleaned.columns:
                continue

            if spec.replace:
                cleaned[col] = cleaned[col].replace(spec.replace)

            if spec.cast:
                if spec.cast == "int":
                    cleaned[col] = cleaned[col].astype("Int64")
                elif spec.cast == "float":
                    cleaned[col] = cleaned[col].astype(float)
                elif spec.cast == "str":
                    cleaned[col] = cleaned[col].astype(str)
                elif spec.cast == "bool":
                    cleaned[col] = cleaned[col].astype(bool)
                else:
                    raise ValueError(f"Unsupported cast type: {spec.cast}")

            if spec.dropna:
                cleaned = cleaned[cleaned[col].notna()]

            if spec.filter_eq is not None:
                cleaned = cleaned.query(f"`{col}` == @spec.filter_eq")

            if spec.filter_in is not None:
                cleaned = cleaned[cleaned[col].isin(list(spec.filter_in))]

    if config.filter_query:
        cleaned = cleaned.query(config.filter_query)

    drop_cols = [c for c in config.drop_columns if c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(drop_cols, axis=1)

    return cleaned


def clean_experiment(
    experiment: Experiment,
    config: Optional[DataCleaningConfig] = None,
    name_suffix: str = "_cleaned"
) -> Experiment:
    """Return a cleaned Experiment while preserving metadata."""
    cleaned_df = clean_experiment_df(experiment.df, config)
    return Experiment(cleaned_df, f"{experiment.name}{name_suffix}", experiment.metadata)


def notebook_plot_config() -> PlotConfig:
    """Default plot styling for marimo notebooks."""
    config = PlotConfig.for_agent_comparison()
    config.log_scale = False
    config.show_legend = True
    return config
